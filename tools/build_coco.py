#!/usr/bin/env python
"""构建统一 COCO 标注缓存（训练 GT + 验证 GT）。

用法:
    python tools/build_coco.py                 # 构建 train + val
    python tools/build_coco.py --split val     # 只构建 val
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.config import load_dataset_cfg
from benchmark.data.coco_convert import build_coco_annotation, default_cache_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["train", "val", "all"])
    args = ap.parse_args()

    cfg = load_dataset_cfg()
    full = cfg  # dataset.yaml 顶层
    splits = ["train", "val"] if args.split == "all" else [args.split]
    for split in splits:
        out = default_cache_path(full, split)
        print(f"[build_coco] {split} -> {out}")
        coco = build_coco_annotation(cfg.dataset, split, out)
        n_inst = sum(1 for a in coco["annotations"]
                     if a["category_id"] == 1)
        print(f"  images={len(coco['images'])} annotations={len(coco['annotations'])} "
              f"landslide={n_inst}")


if __name__ == "__main__":
    main()
