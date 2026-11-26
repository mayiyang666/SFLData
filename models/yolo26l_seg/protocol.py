"""YOLO26l-seg 协议对齐层：数据视图构建 + 受控增广 transform + 自定义 trainer。

ultralytics（8.4.x）默认训练配方含大量框架私有增广（mosaic/mixup/copy_paste/
affine 系/HSV 色相/erasing 等），与 benchmark 控制变量协议不一致。本模块将其
统一到 configs/protocol.yaml 口径：

受控项注入（train.py main 从 cfg.protocol.fixed 读取后传给 ultralytics）:
- seed=42 / imgsz=1024 / epochs=150 / amp=True / workers=4
- 增广: hflip 0.5 + vflip 0.5 + rot90 0.5（自定义 RandomRot90，ultralytics
  无原生参数）+ HSV 饱和度/亮度 ±0.2（近似协议 photometric saturation/
  brightness 0.2；色相与对比度无对应项）
- 其余增广全部关闭: mosaic/mixup/cutmix/copy_paste=0, degrees/translate/
  scale/shear/perspective=0, hsv_h=0, bgr=0, erasing=0, albumentations 未装
- grad_clip: 覆盖 optimizer_step 为协议 grad_clip_norm=0.1
  （ultralytics 默认 10.0）

模型固有差异:
- MuSGD 优化器 + auto lr + EMA（YOLO26 官方配方，optimizer 属 model_specific）
- YOLO26 端到端推理无 NMS（评估时 conf=0.001 阈值与 pycocotools 口径统一，
  但无 NMS 0.6 后处理；e2e 头每图最多 300 检测）
"""
from __future__ import annotations

import json
import os
import random
import shutil
from copy import copy
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _de_parallel(model):
    """等价于 ultralytics 的 `de_parallel`：DDP/DP 包裹时取 `.module`。

    为什么不用 import：本环境的 ultralytics 8.4.140 里
    `ultralytics.utils.torch_utils.de_parallel` **已被移除**（只剩 `is_parallel`），
    `ultralytics.utils` 与 `ultralytics.nn.tasks` 里也没有。2026-09-15 那次
    `ImportError: cannot import name 'de_parallel'` 就是踩了这个 —— rank 0 在
    build_dataset 里当场抛出，rank 1/2 卡在 barrier 上报 "Connection closed by peer"。
    语义与上游完全一致：`model.module if is_parallel(model) else model`。
    """
    try:
        from ultralytics.utils.torch_utils import is_parallel
        return model.module if is_parallel(model) else model
    except Exception:                             # noqa: BLE001
        return getattr(model, "module", model)


# 本模块依赖的全部 ultralytics 外部符号（name -> 候选模块列表，按序尝试）。
# 之所以集中声明：ultralytics 换版本经常搬符号，开训前 1 秒扫一遍比跑崩半天强。
_REQUIRED_SYMBOLS = {
    "Compose": ("ultralytics.data.augment",),
    "LetterBox": ("ultralytics.data.augment",),
    "RandomFlip": ("ultralytics.data.augment",),
    "RandomHSV": ("ultralytics.data.augment",),
    "YOLODataset": ("ultralytics.data.dataset",),
    "Instances": ("ultralytics.utils.instance",),
    "SegmentationTrainer": ("ultralytics.models.yolo.segment",),
    "get_torch_device_backend": ("ultralytics.utils.torch_utils",),
    "LOCAL_RANK": ("ultralytics.utils",),
    "RANK": ("ultralytics.utils",),
    "is_parallel": ("ultralytics.utils.torch_utils",),
}


def preflight_symbols() -> tuple:
    """逐个检查 _REQUIRED_SYMBOLS 是否存在。返回 (ok 列表, missing 列表)。

    只读、不占显存，1 秒内完成。任何 missing 都意味着开训必崩。
    """
    import importlib
    ok, missing = [], []
    for name, mods in _REQUIRED_SYMBOLS.items():
        hit = None
        for m in mods:
            try:
                mod = importlib.import_module(m)
            except Exception:                     # noqa: BLE001
                continue
            if hasattr(mod, name):
                hit = m
                break
        (ok if hit else missing).append("%s (%s)" % (name, hit or "/".join(mods)))
    return ok, missing


# =====================================================================
# 数据视图：按 COCO 缓存生成 ultralytics 可用的 symlink 树 + dataset.yaml
# =====================================================================
def prepare_yolo_dataset(cfg) -> dict:
    """从 data_cache/coco_{train,val}.json 构建 ultralytics 数据视图。

    - 图片/标签以 symlink 链接到源瓦片（源数据只读，视图可随时重建）
- 视图文件名加 {region}__ 前缀防跨区域重名；瓦片为正方形 1024，
  imgsz=1024 下 LetterBox 恒等
    - txt 标签中类别 id >= nc 的行（damage 残留，cid=2）被剔除后落副本，
      其余 symlink —— 保证训练类别严格为 landslide/road 2 类
    - train/val 瓦片集合与 COCO 缓存逐一对齐（后续统一评估同 val）

    返回 {"data_yaml": str, "train": int, "val": int,
          "filtered_labels": int, "filtered_lines": int}
    """
    root = str(cfg.dataset.root)
    cache = str(cfg.paths.cache_dir)
    view = os.path.join(cache, "yolo26l_seg")
    nc = len(list(cfg.benchmark.train_classes))

    # 重建视图（symlink 与过滤副本可安全重建，不影响源数据）
    if os.path.isdir(view):
        shutil.rmtree(view)

    stats = {"train": 0, "val": 0, "filtered_labels": 0, "filtered_lines": 0}
    for split in ("train", "val"):
        with open(os.path.join(cache, f"coco_{split}.json"), "r",
                  encoding="utf-8") as f:
            imgs = json.load(f)["images"]
        img_out = os.path.join(view, "images", split)
        lbl_out = os.path.join(view, "labels", split)
        os.makedirs(img_out)
        os.makedirs(lbl_out)
        for im in imgs:
            fn = im["file_name"]                        # 绝对路径
            rel = os.path.relpath(fn, root)
            region, sp, _, fname = rel.split(os.sep)    # {region}/{split}/images/{name}
            assert sp == split, f"split 不一致: {fn}"
            stem = os.path.splitext(fname)[0]
            # .../images/x.jpg -> .../labels/x.txt（源标签路径，扩展名大小写无关）
            lbl = os.path.join(os.path.dirname(fn), "..", "labels", stem + ".txt")
            lbl = os.path.normpath(lbl)
            os.symlink(fn, os.path.join(img_out, f"{region}__{fname}"))
            lbl_link = os.path.join(lbl_out, f"{region}__{stem}.txt")
            if os.path.isfile(lbl):
                with open(lbl, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                kept = [ln for ln in lines
                        if ln.strip() and int(ln.split()[0]) < nc]
                if len(kept) != len(lines):
                    stats["filtered_labels"] += 1
                    stats["filtered_lines"] += len(lines) - len(kept)
                    with open(lbl_link, "w", encoding="utf-8") as f:
                        f.writelines(kept)
                else:
                    os.symlink(lbl, lbl_link)
            else:
                # 无标签瓦片（负样本）：显式空 txt
                open(lbl_link, "w", encoding="utf-8").close()
            stats[split] += 1

    names = {i: n for i, n in enumerate(list(cfg.benchmark.train_classes))}
    data_yaml = os.path.join(view, "dataset.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {view}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, n in names.items():
            f.write(f"  {i}: {n}\n")

    stats["data_yaml"] = data_yaml
    return stats


# =====================================================================
# 受控增广：协议 rot90（ultralytics 无原生实现）
# =====================================================================
def _protocol_transforms(p_hflip: float, p_vflip: float, p_rot90: float,
                         hsv_s: float, hsv_v: float, imgsz: int):
    """按协议构建训练 transform 链（替代 ultralytics v8_transforms）。

    顺序与 benchmark.data.transforms.TrainAugmentation 语义一致:
    翻转 -> rot90 -> 光度（HSV 在 ultralytics 需在翻转前作用于图像，
    与翻转可交换，仅随机数消耗顺序不同，分布等价）。
    """
    from ultralytics.data.augment import (Compose, LetterBox, RandomFlip,
                                          RandomHSV)

    return Compose([
        RandomHSV(hgain=0.0, sgain=hsv_s, vgain=hsv_v),
        RandomFlip(direction="vertical", p=p_vflip),
        RandomFlip(direction="horizontal", p=p_hflip),
        RandomRot90(p=p_rot90),
        LetterBox(new_shape=(imgsz, imgsz), auto=False),
    ])


class RandomRot90:
    """协议 random_rotate90：p 概率随机 k∈{1,2,3} 次逆时针 90°。

    与 benchmark.data.transforms.TrainAugmentation 同口径（torch.rot90
    逆时针，k~U{1,2,3}）。仅支持归一化坐标（协议链无 mosaic/affine，
    标签自加载起即为归一化，恒成立）。归一化域逆时针 90°:
        (x, y) -> (y, 1 - x)
    与图像宽高无关；瓦片为正方形，np.rot90 后图像尺寸不变。

    实现为 ultralytics BaseTransform 兼容接口（get_params/apply_image/
    apply_instances），与 RandomFlip 同构。
    """

    def __init__(self, p: float = 0.5):
        assert 0 <= p <= 1.0, f"概率应在 [0,1]，got {p}"
        self.p = p

    def __call__(self, labels: dict) -> dict:
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        return labels

    def get_params(self, labels: dict) -> dict:
        return {"rot": random.random() < self.p, "k": random.randint(1, 3)}

    def apply_image(self, labels: dict, params: dict) -> dict:
        if params["rot"]:
            labels["img"] = np.ascontiguousarray(
                np.rot90(labels["img"], params["k"]))
        return labels

    def apply_instances(self, labels: dict, params: dict) -> dict:
        from ultralytics.utils.instance import Instances
        inst = labels.pop("instances")
        if params["rot"] and len(inst):
            if not inst.normalized:
                raise NotImplementedError(
                    "RandomRot90 仅支持归一化坐标（协议链无 mosaic/affine）")
            inst.convert_bbox(format="xyxy")
            b = inst.bboxes.copy()
            seg = inst.segments if inst.segments is not None else None
            for _ in range(params["k"]):
                x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
                b = np.stack([np.minimum(y1, y2), 1.0 - np.maximum(x1, x2),
                              np.maximum(y1, y2), 1.0 - np.minimum(x1, x2)],
                             axis=1)
                if seg is not None and len(seg):
                    seg = np.stack([seg[..., 1], 1.0 - seg[..., 0]], axis=-1)
            b = np.clip(b, 0.0, 1.0)
            if seg is not None and len(seg):
                seg = np.clip(seg, 0.0, 1.0)
            inst = Instances(bboxes=b, segments=seg, keypoints=None,
                             bbox_format="xyxy", normalized=True)
        labels["instances"] = inst
        return labels


# =====================================================================
# 自定义 Dataset / Trainer：注入协议链与 grad_clip
# =====================================================================
class ProtocolYOLODataset:
    """工厂：构建 build_transforms 按 protocol 对齐的 YOLODataset 子类。

    以工厂函数而非静态类实现，便于把协议参数（翻转/rot90/HSV 幅度、
    imgsz）作为闭包传入，避免依赖全局状态；cloudpickle 序列化
    trainer 时闭包随类一起被序列化，DDP 子进程可用。
    """

    def __init__(self, p_hflip: float, p_vflip: float, p_rot90: float,
                 hsv_s: float, hsv_v: float):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v

    def __call__(self, *args, **kwargs):
        from ultralytics.data.dataset import YOLODataset
        factory = self

        class _ProtocolYOLODataset(YOLODataset):
            def build_transforms(self, hyp=None):
                if self.augment:
                    transforms = _protocol_transforms(
                        factory.p_hflip, factory.p_vflip, factory.p_rot90,
                        factory.hsv_s, factory.hsv_v, self.imgsz)
                else:
                    from ultralytics.data.augment import Compose, LetterBox
                    transforms = Compose([
                        LetterBox(new_shape=(self.imgsz, self.imgsz),
                                  scaleup=False)])
                transforms.append(
                    self.format_class(
                        bbox_format="xywh", normalize=True,
                        return_mask=self.use_segments,
                        return_keypoint=self.use_keypoints,
                        return_obb=self.use_obb, batch_idx=True,
                        mask_ratio=hyp.mask_ratio,
                        mask_overlap=hyp.overlap_mask, bgr=0.0))
                return transforms

        return _ProtocolYOLODataset(*args, **kwargs)


def build_protocol_seg_trainer(p_hflip: float, p_vflip: float, p_rot90: float,
                               hsv_s: float, hsv_v: float,
                               grad_clip_norm: float,
                               ddp_backend: str = "gloo"):
    """构建协议对齐的 SegmentationTrainer 子类（工厂，闭包携带协议参数）。

    Select the distributed backend with ddp_backend (gloo or nccl).
    """

    class ProtocolSegTrainer(_ensure_base()):
        """build_dataset 用协议链数据集；optimizer_step 覆盖 grad_clip；
        _setup_ddp 把通信后端从硬编码的 nccl 改为可配置。"""

        def _setup_ddp(self):
            """复刻 BaseTrainer._setup_ddp，只把 init_process_group 的 backend
            从上游硬编码的 `"nccl" if is_nccl_available() else "gloo"` 改成
            ddp_backend。其余（device/accelerator/超时/rank）与上游逐字一致，
            避免与 ultralytics 的 DDP 假设产生偏差。
            """
            import os

            import torch
            from datetime import timedelta
            from torch import distributed as dist
            from ultralytics.utils import LOCAL_RANK, RANK
            from ultralytics.utils.torch_utils import get_torch_device_backend

            device_type = self.args.device.split(":", 1)[0]
            device_type = device_type if device_type in {"npu", "xpu"} else "cuda"
            devices = self.args.device.split(":", 1)[-1].split(",")
            index = int(devices[LOCAL_RANK])  # world_size > 1 保证是多卡串
            self.device = torch.device(device_type, index)
            self.accelerator = get_torch_device_backend(self.device)
            self.accelerator.set_device(index)

            backend = {"npu": "hccl", "xpu": "xccl"}.get(device_type, ddp_backend)
            if backend == "nccl":
                if not dist.is_nccl_available():
                    backend = "gloo"
                else:
                    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # 强制超时生效
            print("[protocol] init_process_group backend=%s rank=%d local_rank=%d "
                  "device=%s world=%d" % (backend, RANK, LOCAL_RANK, self.device,
                                          self.world_size), flush=True)
            dist.init_process_group(
                backend=backend,
                timeout=timedelta(seconds=10800),  # 3 小时，与上游一致
                rank=RANK,
                world_size=self.world_size,
            )
            print("[protocol] process group 就绪 backend=%s rank=%d" % (backend, RANK),
                  flush=True)

        def build_dataset(self, img_path: str, mode: str = "train",
                          batch: int | None = None):
            ds_factory = ProtocolYOLODataset(p_hflip, p_vflip, p_rot90,
                                            hsv_s, hsv_v)
            gs = max(int(_de_parallel(self.model).stride.max()), 32)
            return ds_factory(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch or self.args.batch,
                augment=mode == "train",
                hyp=copy(self.args),
                rect=mode == "val",
                stride=gs,
                pad=0.0 if mode == "train" else 0.5,
                task=self.args.task,
                data=self.data,
                # 与上游 build_yolo_dataset 对齐：fraction 只作用于 train。
                # 默认 1.0 时无行为变化；短冒烟可用 model.fraction=0.02 跑通全链路。
                fraction=float(getattr(self.args, "fraction", 1.0) or 1.0)
                if mode == "train" else 1.0,
            )

        def optimizer_step(self):
            import torch
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_norm=grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)

    return ProtocolSegTrainer


# 延迟解析的 SegmentationTrainer 基类缓存（避免导入顺序问题）
_SegmentationTrainer = None


def _ensure_base():
    """延迟解析并缓存 SegmentationTrainer 基类（避免导入顺序问题）。"""
    global _SegmentationTrainer
    if _SegmentationTrainer is None:
        from ultralytics.models.yolo.segment import SegmentationTrainer
        _SegmentationTrainer = SegmentationTrainer
    return _SegmentationTrainer
