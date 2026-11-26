"""YOLO-seg 瓦片 -> COCO 标注统一转换（共享）。

产物: {cache_dir}/coco_{split}.json  （COCO instance segmentation 格式）
用途: 1) 统一评估的 GT；2) Mask R-CNN / detectron2 / MMDetection 训练输入。
image_id 命名: "{region}__{tile_id}"，保证全局唯一。
"""
from __future__ import annotations

import json
import os

from .yolo_tiles import iter_split_samples


def _shoelace(pts) -> float:
    """Shoelace 公式计算多边形面积。"""
    import numpy as np
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def build_coco_annotation(dataset_cfg, split: str, out_path: str,
                          class_ids: list[int] | None = None) -> dict:
    """将指定 split 转换为 COCO json 并写出。返回 coco dict。"""
    classes = list(dataset_cfg.classes)
    use_ids = class_ids if class_ids is not None else list(range(len(classes)))

    coco = {
        "info": {"description": f"FlashLandslide {split} (unified)",
                 "gsd_m": dataset_cfg.gsd_m, "tile_size": dataset_cfg.tile_size},
        "categories": [{"id": i + 1, "name": classes[i]} for i in use_ids],
        "images": [], "annotations": [],
    }
    ann_id = 1
    for img_id, sample in enumerate(iter_split_samples(dataset_cfg, split), start=1):
        w, h = sample.size_wh
        coco["images"].append({
            "id": img_id,
            # Local training caches use loader paths. Rebuild the cache after moving the dataset.
            "file_name": sample.image_path,
            "width": w, "height": h,
            "region": sample.region, "tile_id": sample.tile_id,
        })
        for cls, pts in zip(sample.class_ids, sample.polygons):
            if cls not in use_ids:
                continue
            flat = pts.reshape(-1).round(3).tolist()
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            area = _shoelace(pts)
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cls + 1,
                "segmentation": [flat],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "area": area, "iscrowd": 0,
            })
            ann_id += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    return coco


def default_cache_path(cfg, split: str) -> str:
    return os.path.join(cfg.paths.cache_dir, f"coco_{split}.json")
