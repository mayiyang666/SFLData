"""统一读取 DPT_region YOLO-seg 瓦片数据集。

布局: {root}/{region}/{split}/images/*.jpg + labels/*.{txt,json}
标签 txt: 每行 "cls x1 y1 x2 y2 ..."（归一化多边形）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class TileSample:
    """单个瓦片样本。polygons 为像素坐标 [[N,2] ...]。"""
    image_path: str
    region: str
    split: str
    tile_id: str                       # 不含扩展名的瓦片名
    polygons: list[np.ndarray] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)

    @property
    def size_wh(self) -> tuple[int, int]:
        with Image.open(self.image_path) as im:
            return im.size


def iter_split_samples(dataset_cfg, split: str):
    """遍历指定 split 的全部样本（惰性生成）。"""
    root = dataset_cfg.root
    for region in dataset_cfg.regions:
        img_dir = os.path.join(root, region, split, "images")
        lab_dir = os.path.join(root, region, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in IMG_EXTS:
                continue
            yield TileSample(
                image_path=os.path.join(img_dir, fn),
                region=region, split=split, tile_id=stem,
                **_parse_label(os.path.join(lab_dir, stem + ".txt"),
                               os.path.join(img_dir, fn)),
            )


def _parse_label(txt_path: str, img_path: str) -> dict:
    polygons, class_ids = [], []
    if not os.path.isfile(txt_path):
        return {"polygons": polygons, "class_ids": class_ids}
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return {"polygons": polygons, "class_ids": class_ids}
    w, h = Image.open(img_path).size
    for ln in lines:
        parts = ln.split()
        cls = int(parts[0])
        coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float64)
        pts = coords.reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        polygons.append(pts.astype(np.float32))
        class_ids.append(cls)
    return {"polygons": polygons, "class_ids": class_ids}
