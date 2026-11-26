#!/usr/bin/env python
"""Export scored YOLO segmentation masks in COCO format. See --help for inputs."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.config import build_full_cfg

# 协议评估口径（conf 阈值；imgsz 与训练一致）
CONF = 0.001
IMGSZ = 1024


def predict_to_results(cfg, ckpt: str, split: str, batch: int,
                       device: str | None, out: str | None = None) -> str:
    from pycocotools import mask as mask_utils
    from ultralytics import YOLO

    cache = str(cfg.paths.cache_dir)
    gt_json = os.path.join(cache, f"coco_{split}.json")
    with open(gt_json, "r", encoding="utf-8") as f:
        gt = json.load(f)
    # file_name 为绝对路径，image_id 直接取 COCO id
    files = [im["file_name"] for im in gt["images"]]
    id_of = {im["file_name"]: im["id"] for im in gt["images"]}

    model = YOLO(ckpt)
    kwargs = dict(source=files, conf=CONF, imgsz=IMGSZ, stream=True,
                  batch=int(batch), verbose=False)
    if device:
        kwargs["device"] = device

    results_out = []
    n_imgs = 0
    for res in model.predict(**kwargs):
        n_imgs += 1
        iid = id_of[res.path]
        if res.masks is None or len(res.boxes) == 0:
            continue
        # masks.data 已映射回原图尺寸；与 COCO 记录宽高校验
        h, w = res.orig_shape
        for j in range(len(res.boxes)):
            cls = int(res.boxes.cls[j].item())
            score = float(res.boxes.conf[j].item())
            m = res.masks.data[j]
            assert tuple(m.shape) == (h, w), \
                f"mask 尺寸 {m.shape} != 原图 {(h, w)}"
            rle = mask_utils.encode(np.asfortranarray(
                m.cpu().numpy().astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            x1, y1, x2, y2 = res.boxes.xyxy[j].tolist()
            results_out.append({
                "image_id": iid,
                "category_id": cls + 1,   # YOLO 0/1 -> COCO 1/2
                "score": score,
                "segmentation": rle,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
            })

    run_name = str(cfg.model.get("name", "yolo26l_seg"))
    if not out:
        out = os.path.join(str(cfg.paths.work_dir), run_name,
                           f"results_{split}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results_out, f)
    print(f"[predict] {n_imgs} 张图 -> {len(results_out)} 个实例 "
          f"(conf={CONF}, imgsz={IMGSZ}) -> {out}")
    print(f"[predict] 统一评估: python tools/evaluate.py --results {out} "
          f"--by-region")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="权重路径，如 work_dirs/yolo26l_seg/weights/best.pt")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="如 0 或 0,1,2（默认自动）")
    ap.add_argument("--out", default=None,
                    help="results JSON 输出路径（默认 work_dirs/<model.name>/）")
    ap.add_argument("--overrides", nargs="*", default=None,
                    help='config 覆盖，如 model.name=yolo26l_seg_retrain')
    args = ap.parse_args()

    cfg = build_full_cfg("yolo26l_seg", overrides=args.overrides or None)
    predict_to_results(cfg, args.ckpt, args.split, args.batch, args.device,
                       args.out)


if __name__ == "__main__":
    main()
