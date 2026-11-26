#!/usr/bin/env python
"""Build the COCO ground truth paired with the saved SFLData paper predictions."""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.config import load_dataset_cfg
from benchmark.data.coco_convert import build_coco_annotation
from benchmark.data.yolo_tiles import IMG_EXTS


def check_pairs(folder):
    images = [p.stem for p in (folder / "images").iterdir()
              if p.is_file() and p.suffix.lower() in IMG_EXTS]
    labels = {p.stem for p in (folder / "labels").glob("*.txt")}
    if not images or len(set(images)) != len(images) or set(images) != labels:
        raise ValueError(f"Unpaired or duplicate tile stems in {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=["main", "heldout"])
    parser.add_argument("--root", required=True, type=Path,
                        help="Parent of the region folders containing paired images and YOLO labels.")
    parser.add_argument("--output", required=True, type=Path, help="Destination COCO JSON.")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.scope == "main":
        split, classes = "val", ["landslide", "road"]
        # The original COCO image IDs follow sorted folder names, then sorted tile names.
        regions = sorted(p.name for p in root.iterdir() if (p / split / "images").is_dir())
        if len(regions) != 18:
            raise ValueError(f"Expected 18 validation regions, found {len(regions)}")
        n_images, support = 7274, {1: 3676, 2: 3672}
    else:
        split, classes = "17val", ["landslide"]
        regions = ["Lixian_Shiziping"]
        n_images, support = 8883, {1: 3421}

    for region in regions:
        check_pairs(root / region / split)
    cfg = load_dataset_cfg().dataset
    cfg.root, cfg.regions, cfg.classes = root.as_posix(), regions, classes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        staged = Path(temp) / "coco_val.json"
        gt = build_coco_annotation(cfg, split, str(staged))
        images = gt["images"]
        sizes = {(im["width"], im["height"]) for im in images}
        counts = Counter(ann["category_id"] for ann in gt["annotations"])
        if len(images) != n_images or sizes != {(1024, 1024)} or counts != support:
            raise ValueError(f"Unexpected {args.scope} inventory: {len(images)} tiles, {counts}, {sizes}")
        os.replace(staged, args.output)
    print(f"{args.scope}: {n_images} images, category counts {dict(counts)} -> {args.output}")


if __name__ == "__main__":
    main()
