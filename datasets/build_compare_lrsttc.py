#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Assign LRSTTC images to stratified splits and export instance polygons and the split manifest."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

SRC_ROOT = "data/external/raw/LRSTTC_dataset"
OUT_ROOT = "data/Compare_dataset/LRSTTC"

# (subset dir name, filename prefix) -- prefixes disambiguate the colliding 000001.png names
SUBSETS = [("new_landslides", "new"), ("old_landslides", "old")]

SEED = 42
VAL_RATIO = 0.125
MIN_AREA = 10.0        # px^2, contours below this are dropped as noise
EPS_REL = 0.001        # approxPolyDP tolerance relative to contour perimeter

# instance-count buckets: (label, low, high)
BUCKETS = [
    ("0", 0, 0),
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4-6", 4, 6),
    ("7-10", 7, 10),
    ("11-20", 11, 20),
    ("21+", 21, 10 ** 9),
]


def bucket_of(n: int) -> str:
    for label, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return label
    return "21+"


# --------------------------------------------------------------------------- mask -> polygons
def poly_from_contour(cnt, eps_rel: float, min_area: float):
    """Simplify a contour; fall back to coarser tolerance, then to the raw contour."""
    peri = cv2.arcLength(cnt, True)
    for factor in (1.0, 0.25, 0.05):
        ap = cv2.approxPolyDP(cnt, max(0.5, eps_rel * factor * peri), True)
        pts = ap.reshape(-1, 2).astype(np.float64)
        if pts.shape[0] >= 3 and cv2.contourArea(ap) >= min_area:
            return pts
    pts = cnt.reshape(-1, 2).astype(np.float64)          # raw fallback
    if pts.shape[0] >= 3 and cv2.contourArea(cnt) >= min_area:
        return pts
    return None


def mask_to_polygons(mask: np.ndarray, min_area: float = MIN_AREA, eps_rel: float = EPS_REL):
    """Instance-id mask -> list of (instance_id, pts[N,2] pixel coords). Outer contours only.

    Outer-only matches ultralytics' polygons2masks (fillPoly per polygon), so holes are
    filled identically at training time -> the label is faithful to what the model sees.
    """
    h, w = mask.shape[:2]
    polys: list[tuple[int, np.ndarray]] = []
    stats = {"instances": 0, "contours": 0, "dropped": 0, "holes": 0, "multi_part": 0}
    vals = np.unique(mask)
    vals = vals[vals > 0]
    stats["instances"] = int(vals.size)
    for v in vals:
        binary = (mask == v).astype(np.uint8)
        # reporting only: instances whose region encloses a hole (RETR_CCOMP child contours)
        _all, hier = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is not None and len(_all):
            hier0 = hier[0]
            if int(np.count_nonzero(hier0[:, 3] != -1)) > 0:
                stats["holes"] += 1
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = 0
        for c in cnts:
            pts = poly_from_contour(c, eps_rel, min_area)
            if pts is None:
                stats["dropped"] += 1
                continue
            polys.append((int(v), pts))
            kept += 1
        stats["contours"] += kept
        if kept > 1:
            stats["multi_part"] += 1
    return polys, stats


def polygons_to_line(polys, w: int, h: int, cls: int = 0) -> str:
    lines = []
    for _v, pts in polys:
        norm = np.clip(pts / np.array([w, h], dtype=np.float64), 0.0, 1.0)
        flat = []
        for x, y in norm:
            flat.append(repr(float(x)))
            flat.append(repr(float(y)))
        lines.append(f"{cls} " + " ".join(flat))
    return "\n".join(lines)


def rasterize(polys, w: int, h: int) -> np.ndarray:
    """Reproduce what ultralytics sees: fill every polygon of the class into one mask."""
    canvas = np.zeros((h, w), dtype=np.uint8)
    for _v, pts in polys:
        cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
    return canvas


# --------------------------------------------------------------------------- scan / split
def scan_source():
    items = []
    for sub, prefix in SUBSETS:
        pic_dir = os.path.join(SRC_ROOT, sub, "pic")
        mask_dir = os.path.join(SRC_ROOT, sub, "mask")
        if not os.path.isdir(pic_dir) or not os.path.isdir(mask_dir):
            sys.exit(f"[ERR] missing source dir: {pic_dir} / {mask_dir}")
        pics = sorted(os.listdir(pic_dir))
        masks = sorted(os.listdir(mask_dir))
        if len(pics) != len(masks):
            sys.exit(f"[ERR] {sub}: pic={len(pics)} != mask={len(masks)}")
        stem_p = {os.path.splitext(p)[0] for p in pics}
        stem_m = {os.path.splitext(m)[0] for m in masks}
        if stem_p != stem_m:
            sys.exit(f"[ERR] {sub}: image/mask stems mismatch ({len(stem_p ^ stem_m)} diffs)")
        for p in pics:
            stem = os.path.splitext(p)[0]
            items.append({
                "subset": sub,
                "prefix": prefix,
                "stem": stem,
                "src_image": os.path.join(pic_dir, p),
                "src_mask": os.path.join(mask_dir, os.path.splitext(p)[0] + ".png"),
                "out_stem": f"{prefix}_{stem}",
            })
    return items


def largest_remainder_split(groups: dict[str, list], ratio: float, seed: int):
    """Allocate val counts per bucket by largest-remainder so the total hits round(N*ratio)."""
    total = sum(len(v) for v in groups.values())
    target = int(round(total * ratio))
    floor = {k: int(len(v) * ratio) for k, v in groups.items()}
    assigned = sum(floor.values())
    rem = sorted(
        ((len(v) * ratio - floor[k], k) for k, v in groups.items()),
        key=lambda t: (-t[0], t[1]),
    )
    i = 0
    while assigned < target and rem:
        k = rem[i % len(rem)][1]
        if floor[k] < len(groups[k]):
            floor[k] += 1
            assigned += 1
        i += 1
        if i > 10 * len(rem) * (target + 2):
            break
    rng = random.Random(seed)
    val, train = [], []
    for k in sorted(groups, key=lambda x: [b[0] for b in BUCKETS].index(x)):
        pool = sorted(groups[k], key=lambda it: it["out_stem"])
        n_val = floor[k]
        chosen = set(rng.sample(range(len(pool)), n_val)) if n_val else set()
        for idx, it in enumerate(pool):
            (val if idx in chosen else train).append(it)
    return train, val, target


# --------------------------------------------------------------------------- selftest
def selftest(n: int):
    items = scan_source()
    rng = random.Random(SEED)
    probe = rng.sample(items, min(n, len(items)))
    print(f"[selftest] {len(probe)} images, MIN_AREA={MIN_AREA}, EPS_REL={EPS_REL}\n")
    ious, drops, tot_inst = [], 0, 0
    for it in probe:
        img = Image.open(it["src_image"])
        w, h = img.size
        mask = np.array(Image.open(it["src_mask"]))
        polys, st = mask_to_polygons(mask)
        drops += st["dropped"]
        tot_inst += st["instances"]
        gt = (mask > 0).astype(np.uint8)
        pred = rasterize(polys, w, h)
        inter = int(np.logical_and(gt, pred).sum())
        union = int(np.logical_or(gt, pred).sum())
        iou = inter / union if union else 1.0
        ious.append(iou)
        print(f"  {it['out_stem']:<14} {w}x{h} inst={st['instances']} polys={st['contours']} "
              f"drop={st['dropped']} holes={st['holes']} IoU={iou:.4f}")
    print(f"\n  images={len(probe)} instances={tot_inst} dropped_contours={drops}")
    print(f"  IoU  min={min(ious):.4f}  mean={sum(ious)/len(ious):.4f}  max={max(ious):.4f}")
    print("  (IoU < 1 is expected: approxPolyDP simplification + outer-only holes)")


# --------------------------------------------------------------------------- main build
def main(force: bool):
    items = scan_source()
    print(f"[scan] {len(items)} images from {len(SUBSETS)} subsets")

    # phase A: instance counts
    for it in items:
        it["n_instances"] = int(np.count_nonzero(np.unique(np.array(Image.open(it["src_mask"])))))
        it["bucket"] = bucket_of(it["n_instances"])
    print("[scan] instance counts done")

    groups: dict[str, list] = {}
    for it in items:
        groups.setdefault(it["bucket"], []).append(it)

    train, val, target = largest_remainder_split(groups, VAL_RATIO, SEED)
    print(f"[split] target val={target} ({VAL_RATIO:.1%} of {len(items)}) "
          f"-> actual train={len(train)} val={len(val)} ({len(val)/len(items):.4%})")

    # phase B: fresh output tree
    if os.path.exists(OUT_ROOT):
        if not force:
            sys.exit(f"[ERR] output root exists, pass --force to overwrite: {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)
    dirs = {}
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = os.path.join(OUT_ROOT, split, kind)
            os.makedirs(d, exist_ok=True)
            dirs[(split, kind)] = d

    # phase C: convert
    manifest = []
    agg = []
    for split, bucket in (("train", train), ("val", val)):
        print(f"[convert] {split}: {len(bucket)} images")
        for i, it in enumerate(bucket, 1):
            img = Image.open(it["src_image"])
            w, h = img.size
            mask = np.array(Image.open(it["src_mask"]))
            polys, st = mask_to_polygons(mask)
            text = polygons_to_line(polys, w, h, cls=0)
            if st["contours"] == 0:
                print(f"  [warn] {it['out_stem']}: {st['instances']} instances but 0 polygons kept")
            out_img = os.path.join(dirs[(split, "images")], it["out_stem"] + ".png")
            out_lbl = os.path.join(dirs[(split, "labels")], it["out_stem"] + ".txt")
            try:
                os.link(it["src_image"], out_img)
                img_mode = "hardlink"
            except OSError:
                shutil.copy2(it["src_image"], out_img)
                img_mode = "copy"
            with open(out_lbl, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            rec = {
                "split": split,
                "subset": it["subset"],
                "out_stem": it["out_stem"],
                "width": w,
                "height": h,
                "n_instances": it["n_instances"],
                "bucket": it["bucket"],
                "n_polygons": st["contours"],
                "dropped_contours": st["dropped"],
                "multi_part_instances": st["multi_part"],
                "image_mode": img_mode,
                "source_image": os.path.relpath(it["src_image"], SRC_ROOT).replace(os.sep, "/"),
                "source_mask": os.path.relpath(it["src_mask"], SRC_ROOT).replace(os.sep, "/"),
            }
            manifest.append(rec)
            agg.append(rec)
            if i % 200 == 0:
                print(f"    ... {i}/{len(bucket)}")

    total_inst = sum(r["n_instances"] for r in agg)
    val_inst = sum(r["n_instances"] for r in agg if r["split"] == "val")

    summary = {
        "format": "compare_dataset_lrsttc_v1",
        "source_root": "<src>",
        "output_root": ".",
        "subsets": [{"dir": s, "prefix": p} for s, p in SUBSETS],
        "class_scheme": {"nc": 1, "names": {0: "landslide"}},
        "label_encoding": "yolo-seg polygon, one line per polygon, normalized xy, empty file = negative",
        "polygon_extraction": {
            "contour_mode": "RETR_EXTERNAL",
            "min_area_px2": MIN_AREA,
            "approx_eps_rel": EPS_REL,
        },
        "split": {
            "method": "per-image instance-count bucketing, largest-remainder allocation of 12.5%",
            "buckets": [b[0] for b in BUCKETS],
            "seed": SEED,
            "requested_val_ratio": VAL_RATIO,
            "target_val_images": target,
        },
        "totals": {
            "images": len(agg),
            "instances": total_inst,
            "train_images": len(train),
            "val_images": len(val),
            "actual_val_ratio": len(val) / len(agg),
            "train_instances": total_inst - val_inst,
            "val_instances": val_inst,
            "val_instance_ratio": val_inst / total_inst if total_inst else 0.0,
            "empty_label_images": sum(1 for r in agg if r["n_polygons"] == 0),
            "total_polygons": sum(r["n_polygons"] for r in agg),
            "dropped_contours": sum(r["dropped_contours"] for r in agg),
        },
        "per_bucket": [],
        "per_subset": [],
    }

    for label, _lo, _hi in BUCKETS:
        pool = [r for r in agg if r["bucket"] == label]
        if not pool:
            continue
        v = [r for r in pool if r["split"] == "val"]
        summary["per_bucket"].append({
            "bucket": label,
            "images": len(pool),
            "val_images": len(v),
            "actual_val_ratio": len(v) / len(pool),
            "instances": sum(r["n_instances"] for r in pool),
            "val_instances": sum(r["n_instances"] for r in v),
        })

    for sub, prefix in SUBSETS:
        pool = [r for r in agg if r["subset"] == sub]
        v = [r for r in pool if r["split"] == "val"]
        summary["per_subset"].append({
            "subset": sub,
            "prefix": prefix,
            "images": len(pool),
            "train_images": len(pool) - len(v),
            "val_images": len(v),
            "actual_val_ratio": len(v) / len(pool),
            "instances": sum(r["n_instances"] for r in pool),
            "val_instances": sum(r["n_instances"] for r in v),
        })

    # audit: pairing + coordinate sanity + instance conservation
    audit = {"train_images": 0, "train_labels": 0, "val_images": 0, "val_labels": 0,
             "unpaired": 0, "out_of_range": 0, "degenerate_polygons": 0,
             "label_instances_sum": 0, "mask_instances_sum": 0}
    for split in ("train", "val"):
        imgs = {os.path.splitext(f)[0] for f in os.listdir(dirs[(split, "images")])}
        lbls = {os.path.splitext(f)[0] for f in os.listdir(dirs[(split, "labels")])}
        audit[f"{split}_images"] = len(imgs)
        audit[f"{split}_labels"] = len(lbls)
        audit["unpaired"] += len(imgs ^ lbls)
        for name in sorted(lbls):
            with open(os.path.join(dirs[(split, "labels")], name + ".txt"), encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    coords = [float(x) for x in parts[1:]]
                    if len(coords) < 6 or len(coords) % 2:
                        audit["degenerate_polygons"] += 1
                        continue
                    for c in coords:
                        if not (0.0 <= c <= 1.0):
                            audit["out_of_range"] += 1
    audit["label_instances_sum"] = sum(r["n_instances"] for r in agg)
    audit["mask_instances_sum"] = total_inst
    summary["audit"] = audit

    with open(os.path.join(OUT_ROOT, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_ROOT, "split_manifest.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        wr.writeheader()
        wr.writerows(manifest)

    yaml_path = os.path.join(OUT_ROOT, "LRSTTC.yaml")
    with open(yaml_path, "w", encoding="utf-8", newline="\n") as f:
        # With no path override, Ultralytics resolves the splits from this YAML file.
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  0: landslide\n")

    print("\n================ SUMMARY ================")
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("totals", "per_bucket", "per_subset", "audit")},
                     ensure_ascii=False, indent=2))
    print(f"\n[ok] dataset -> {OUT_ROOT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", type=int, default=0, metavar="N")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--src", default=SRC_ROOT,
                    help=f"源数据根目录（默认 {SRC_ROOT}，请按需覆盖）")
    ap.add_argument("--out", default=OUT_ROOT,
                    help=f"输出根目录（默认 {OUT_ROOT}，请按需覆盖）")
    a = ap.parse_args()
    globals()["SRC_ROOT"] = a.src
    globals()["OUT_ROOT"] = a.out
    if a.selftest:
        selftest(a.selftest)
    else:
        main(a.force)
