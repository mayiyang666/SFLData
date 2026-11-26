"""共享实例分割数据集：从统一 COCO json 读取实例多边形，输出 torchvision
检测/实例分割模型所需的 (image, target) 格式。

供 Mask R-CNN / Cascade Mask R-CNN 等实例分割模型共用。
- 类别: COCO category_id 原样使用 (1=landslide, 2=road)，0 为背景
- 图像: float32 [3,H,W]，取值 [0,1]，**不做归一化**——torchvision 检测模型
  内部 GeneralizedRCNNTransform 以 ImageNet 统计量归一化，与协议一致
- 训练增广: 复用 benchmark.data.transforms.TrainAugmentation（翻转/90°旋转
  作用于图像与多边形，光度扰动仅作用于图像），与语义分割版随机序列对齐
- 尺寸统一: 右下 pad / 左上裁剪到 input_size（瓦片通常已是 1024）
- 空标注瓦片: 返回 N=0 的合法 target，保证背景瓦片可参与训练
"""
from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image, ImageFile
from pycocotools import mask as mask_util
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes
from torchvision.transforms import functional as TF

from .transforms import TrainAugmentation

# 数据集中存在少量尾部截断的瓦片（切图产物），容错加载避免训练中断
ImageFile.LOAD_TRUNCATED_IMAGES = True


def collate_fn(batch):
    """检测模型标准 collate：不堆叠，返回 (images_tuple, targets_tuple)。"""
    return tuple(zip(*batch))


class InstanceDataset(Dataset):
    """实例分割数据集（读取统一 COCO json）。

    返回 (image: float32 [3,s,s] ∈ [0,1], target: dict)：
        boxes:    float32 [N,4] (x1,y1,x2,y2)
        labels:   int64   [N]   COCO category_id (1=landslide, 2=road)
        masks:    uint8   [N,s,s]
        image_id: int64   标量，与 coco_{split}.json 中 id 一致（评估对齐用）
        area / iscrowd:   COCO 评估兼容字段
    """

    def __init__(self, coco_json: str, input_size: int, train: bool = True,
                 aug_cfg=None):
        with open(coco_json, "r", encoding="utf-8") as f:
            self.coco = json.load(f)
        self.input_size = int(input_size)
        self.train = bool(train)
        self.aug = TrainAugmentation(aug_cfg) if (train and aug_cfg is not None) else None
        self.images = {im["id"]: im for im in self.coco["images"]}
        self.img_ids = [im["id"] for im in self.coco["images"]]
        self.anns_by_img: dict[int, list] = {}
        for ann in self.coco["annotations"]:
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.img_ids)

    # ---------------- 内部工具 ----------------
    def _polygons_of(self, img_id: int) -> tuple[list[np.ndarray], list[int], list[float], list[int]]:
        """返回 (多边形列表 [M,2] 像素坐标, 类别, area, iscrowd)。"""
        polys, cats, areas, crowds = [], [], [], []
        for ann in self.anns_by_img.get(img_id, []):
            seg = ann["segmentation"]
            # 统一取 polygon 形式；RLE 形式的 GT 在本数据集中不出现
            plist = seg if isinstance(seg, list) else [seg]
            for p in plist:
                arr = np.asarray(p, dtype=np.float32).reshape(-1, 2)
                if arr.shape[0] >= 3:
                    polys.append(arr)
                    cats.append(int(ann["category_id"]))
                    areas.append(float(ann.get("area", 0.0)))
                    crowds.append(int(ann.get("iscrowd", 0)))
        return polys, cats, areas, crowds

    @staticmethod
    def _clip_polys(polys: list[np.ndarray], w: int, h: int) -> list[np.ndarray]:
        """坐标裁剪到图像范围内（旋转变换后可能越界 1px）。"""
        out = []
        for p in polys:
            q = p.copy()
            q[:, 0] = np.clip(q[:, 0], 0, max(w - 1, 0))
            q[:, 1] = np.clip(q[:, 1], 0, max(h - 1, 0))
            out.append(q)
        return out

    def _to_target(self, polys: list[np.ndarray], cats: list[int],
                   areas: list[float], crowds: list[int],
                   img_id: int, h: int, w: int) -> dict:
        """多边形 -> masks/boxes，过滤退化实例，组装 target。"""
        s = self.input_size
        keep_polys, keep = [], []
        for i, p in enumerate(polys):
            # 裁剪到最终画布（左上裁剪情形下越界点收边）
            q = p.copy()
            q[:, 0] = np.clip(q[:, 0], 0, s - 1)
            q[:, 1] = np.clip(q[:, 1], 0, s - 1)
            keep_polys.append(q)
            keep.append(i)

        if keep_polys:
            flats = [q.reshape(-1).tolist() for q in keep_polys]
            rles = mask_util.frPyObjects(flats, s, s)
            masks = torch.from_numpy(
                np.stack([mask_util.decode(r) for r in rles])).to(torch.uint8)
            boxes = masks_to_boxes(masks)
            areas_t = masks.flatten(1).sum(1).float()
            valid = (areas_t > 1.0) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            idx = valid.nonzero(as_tuple=False).flatten()
            masks, boxes, areas_t = masks[idx], boxes[idx], areas_t[idx]
            labels = torch.tensor([cats[keep[i]] for i in idx.tolist()], dtype=torch.int64)
            iscrowd = torch.tensor([crowds[keep[i]] for i in idx.tolist()], dtype=torch.int64)
        else:
            masks = torch.zeros((0, s, s), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            areas_t = torch.zeros((0,), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        return {"boxes": boxes, "labels": labels, "masks": masks,
                "image_id": torch.tensor(img_id, dtype=torch.int64),
                "area": areas_t, "iscrowd": iscrowd}

    # ---------------- 主流程 ----------------
    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        im_meta = self.images[img_id]
        pil = Image.open(im_meta["file_name"]).convert("RGB")
        w0, h0 = pil.size
        polys, cats, areas, crowds = self._polygons_of(img_id)
        s = self.input_size

        if self.aug is not None:
            img_u8, polys = self.aug(pil, polys)   # uint8 [3,h,w] + 同步变换多边形
            img = img_u8.to(torch.float32).div_(255.0)
        else:
            img = TF.to_tensor(pil)                # float32 [0,1]
        _, h, w = img.shape
        polys = self._clip_polys(polys, w, h)

        # 右下 pad 到 input_size（坐标不变）；左上裁剪在 _to_target 中收边
        if h < s or w < s:
            img = torch.nn.functional.pad(img, (0, max(0, s - w), 0, max(0, s - h)))
        img = img[:, :s, :s]

        target = self._to_target(polys, cats, areas, crowds, img_id, h0, w0)
        return img, target
