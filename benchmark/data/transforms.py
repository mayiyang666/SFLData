"""统一训练增广管线（所有模型共用，禁止修改协议以外的增广）。

实现基于 torch + torchvision，多边形与图像同步变换；
验证集固定为 no-aug（仅归一化），由协议保证。
"""
from __future__ import annotations

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image
import numpy as np


class TrainAugmentation:
    """协议规定的训练增广：翻转 + 90° 旋转 + 光度扰动（多边形同步）。"""

    def __init__(self, aug_cfg):
        self.hflip_p = float(aug_cfg.random_horizontal_flip)
        self.vflip_p = float(aug_cfg.random_vertical_flip)
        self.rot90_p = float(aug_cfg.random_rotate90)
        ph = aug_cfg.photometric
        self.color = T.ColorJitter(
            brightness=float(ph.brightness), contrast=float(ph.contrast),
            saturation=float(ph.saturation), hue=0.0)
        self.to_image = T.ToImage()

    def __call__(self, pil_img: Image.Image, polygons: list[np.ndarray]):
        """返回 (uint8 tensor [3,H,W], list[np.ndarray(N,2)] 像素坐标)。"""
        img = self.to_image(pil_img)
        w, h = pil_img.size
        polys = [p.copy() for p in polygons]

        if torch.rand(1) < self.hflip_p:
            img = F.horizontal_flip(img)
            polys = [np.stack([w - p[:, 0], p[:, 1]], axis=1) for p in polys]
        if torch.rand(1) < self.vflip_p:
            img = F.vertical_flip(img)
            polys = [np.stack([p[:, 0], h - p[:, 1]], axis=1) for p in polys]
        if torch.rand(1) < self.rot90_p:
            k = int(torch.randint(1, 4, (1,)))
            img = torch.rot90(img, k, dims=[-2, -1])  # 逆时针
            for _ in range(k):
                # 逆时针 90°: (x, y) -> (y, w - x)，且新尺寸 (w, h) -> (h, w)
                polys = [np.stack([p[:, 1], w - p[:, 0]], axis=1) for p in polys]
                w, h = h, w
        img = self.color(img)
        return img, polys
