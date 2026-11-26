"""共享语义分割数据集：从统一 COCO json 渲染语义 mask。

供 U-Net / DeepLabV3+ / SegFormer 等语义分割模型共用。
- 类别映射: 0=background, 其余按 dataset.yaml classes 顺序 (1=landslide, 2=road)
- 训练增广: 与 protocol.yaml 完全一致（翻转/90°旋转作用于图像与 mask，
  光度扰动仅作用于图像），随机序列与实例分割版 TrainAugmentation 对齐
- 重叠实例按 category 升序绘制，保证 landslide(1) 最后覆盖，避免被 road 盖住
"""
from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image, ImageFile
from pycocotools import mask as mask_util
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

# 数据集中存在少量尾部截断的瓦片（切图产物），容错加载避免训练中断
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ImageNet 归一化使用的通道均值和标准差
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def img_to_device(img, device, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    """uint8 批次 [B,3,H,W] -> GPU 上归一化 float32 批次（与旧 worker 端归一化数值等价）。

    在 worker 中保留 uint8，可降低共享内存占用；传到设备后再转换为浮点数并归一化。
    """
    img = img.to(device, non_blocking=True).float().div_(255.0)
    return TF.normalize(img, mean=list(mean), std=list(std))


class SegDataset(Dataset):
    """语义分割数据集（读取统一 COCO json）。

    返回 (image: uint8 [3,H,W] 0-255 未归一化, mask: uint8 [H,W])。
    消费端: 图像用 img_to_device() 上 GPU 并归一化；mask 入损失前需 .long()。
    """

    def __init__(self, coco_json: str, input_size: int, train: bool = True,
                 aug_cfg=None, normalize=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        with open(coco_json, "r", encoding="utf-8") as f:
            self.coco = json.load(f)
        self.input_size = int(input_size)
        self.train = bool(train)
        self.aug_cfg = aug_cfg
        self.mean = list(normalize)
        self.std = list(std)
        self.images = {im["id"]: im for im in self.coco["images"]}
        self.img_ids = [im["id"] for im in self.coco["images"]]
        self.anns_by_img: dict[int, list] = {}
        for ann in self.coco["annotations"]:
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)
        # landslide(cat 1) 最后绘制，保证优先级
        for v in self.anns_by_img.values():
            v.sort(key=lambda a: -a["category_id"])

    def __len__(self) -> int:
        return len(self.img_ids)

    def _render_mask(self, img_id: int, h: int, w: int) -> np.ndarray:
        m = np.zeros((h, w), dtype=np.uint8)
        for ann in self.anns_by_img.get(img_id, []):
            seg = ann["segmentation"]
            polys = seg if isinstance(seg, list) else [seg]
            rles = mask_util.frPyObjects(polys, h, w)
            dec = mask_util.decode(mask_util.merge(rles))
            m[dec > 0] = ann["category_id"]
        return m

    def _augment(self, img: torch.Tensor, mask: torch.Tensor):
        """训练增广：随机序列与 TrainAugmentation 一致（hflip -> vflip -> rot90）。"""
        a = self.aug_cfg
        if torch.rand(1) < float(a.random_horizontal_flip):
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if torch.rand(1) < float(a.random_vertical_flip):
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if torch.rand(1) < float(a.random_rotate90):
            k = int(torch.randint(1, 4, (1,)))
            img = torch.rot90(img, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])
        ph = a.photometric
        img = TF.adjust_brightness(img, 1.0 + float(torch.empty(1).uniform_(
            -ph.brightness, ph.brightness)))
        img = TF.adjust_contrast(img, 1.0 + float(torch.empty(1).uniform_(
            -ph.contrast, ph.contrast)))
        img = TF.adjust_saturation(img, 1.0 + float(torch.empty(1).uniform_(
            -ph.saturation, ph.saturation)))
        return img, mask

    def _pad_or_crop(self, img: torch.Tensor, mask: torch.Tensor | None):
        """统一到 input_size（右下 pad / 左上裁剪），瓦片通常已是 1024。"""
        s = self.input_size
        _, h, w = img.shape
        if h < s or w < s:
            pad_h, pad_w = max(0, s - h), max(0, s - w)
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
            if mask is not None:
                mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h))
        img = img[:, :s, :s]
        if mask is not None:
            mask = mask[:s, :s]
        return img, mask

    def __getitem__(self, idx: int):
        im_meta = self.images[self.img_ids[idx]]
        img = Image.open(im_meta["file_name"]).convert("RGB")
        w, h = img.size
        img = TF.to_tensor(img)                                  # float32 [0,1]
        mask = torch.from_numpy(self._render_mask(im_meta["id"], h, w))  # uint8
        if self.train and self.aug_cfg is not None:
            img, mask = self._augment(img, mask)
        img, mask = self._pad_or_crop(img, mask)
        # uint8 出 worker 以压缩 /dev/shm 传输量；归一化移至 GPU 端（img_to_device）
        img = img.mul(255.0).round().clamp(0, 255).to(torch.uint8)
        return img, mask
