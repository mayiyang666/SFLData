#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare labelled and negative Bijie patches using aspect-preserving letterboxing."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import statistics
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_compare_lrsttc import BUCKETS, EPS_REL, MIN_AREA, bucket_of, largest_remainder_split, mask_to_polygons  # noqa: E402
from build_compare_lrsttc_1024 import (S, letterbox, letterbox_image, letterbox_polys,  # noqa: E402
                                       polygons_to_line, rasterize, reference_fg)

SRC_ROOT = "data/external/raw/Bijie-landslide-dataset"
OUT_ROOT = "data/Compare_dataset_1024/Bijie"
SEED = 42
VAL_RATIO = 0.125
# (subset dir, has mask/annotation)
SUBSETS = [("landslide", True), ("non-landslide", False)]


# --------------------------------------------------------------------------- helpers
def region_prefix(stem: str) -> str:
    """文件名前缀（区域码）：df002 -> df, dzjwv05285 -> dzjwv。"""
    out = "".join(ch for ch in stem if not ch.isdigit())
    return out or stem


def read_polygon_txt(path: str):
    """解析 polygon_coordinate/*.txt：跳过表头，只取 'x y' 行。"""
    pts = []
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            q = ln.split()
            if len(q) != 2:
                continue
            try:
                pts.append((float(q[0]), float(q[1])))
            except ValueError:
                continue
    return np.asarray(pts, dtype=np.float64) if len(pts) >= 3 else None


# --------------------------------------------------------------------------- scan
def scan_source() -> list[dict]:
    items = []
    for sub, has_mask in SUBSETS:
        img_dir = os.path.join(SRC_ROOT, sub, "image")
        if not os.path.isdir(img_dir):
            sys.exit(f"[ERR] missing source dir: {img_dir}")
        names = sorted(os.listdir(img_dir))
        for p in names:
            stem = os.path.splitext(p)[0]
            it = {
                "subset": sub,
                "stem": stem,
                "region": region_prefix(stem),
                "out_stem": stem,
                "src_image": os.path.join(img_dir, p),
                "src_mask": os.path.join(SRC_ROOT, sub, "mask", stem + ".png") if has_mask else None,
                "src_poly": os.path.join(SRC_ROOT, sub, "polygon_coordinate", stem + ".txt") if has_mask else None,
                "has_mask": has_mask,
            }
            items.append(it)
    # stem 冲突检查（两个 subset 合并后必须唯一）
    dup = [s for s, n in collections.Counter(i["out_stem"] for i in items).items() if n > 1]
    if dup:
        sys.exit(f"[ERR] 合并后 stem 冲突 {len(dup)} 个: {dup[:10]}")
    return items


def instance_count(it: dict) -> int:
    if not it["has_mask"]:
        return 0
    m = np.array(Image.open(it["src_mask"]))
    return int(np.count_nonzero(np.unique(m)))


# --------------------------------------------------------------------------- selftest
def selftest(n: int):
    items = scan_source()
    print(f"[selftest] 源图 {len(items)} "
          f"(landslide {sum(1 for i in items if i['has_mask'])} / "
          f"non-landslide {sum(1 for i in items if not i['has_mask'])}) | S={S}\n")
    rng = random.Random(SEED)
    pos = rng.sample([i for i in items if i["has_mask"]], min(n, 770))
    neg = rng.sample([i for i in items if not i["has_mask"]], min(4, 2003))
    print(f"  {'stem':<14}{'src':>11}{'-> nw x nh':>13}{'up x':>7}{'fill':>7}"
          f"{'inst':>5}{'poly':>5}{'IoU':>8}{'nat':>8}{'txtIoU':>8}{'cc':>4}")
    ious, txt_ious, ups, fills, nat_ious = [], [], [], [], []
    tot_inst = 0
    for it in pos + neg:
        im = Image.open(it["src_image"]).convert("RGB")
        w, h = im.size
        geo = letterbox((w, h))
        _sx, _sy, nw, nh, dx, dy = geo
        up = min(nw / w, nh / h)
        ups.append(up)
        fills.append(nw * nh / (S * S))
        if it["has_mask"]:
            mask = np.array(Image.open(it["src_mask"]))
            polys_raw, st = mask_to_polygons(mask)
            gt = reference_fg(mask, geo)
            cc = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)[0] - 1
        else:
            mask, polys_raw, st, cc = None, [], {"instances": 0, "contours": 0, "dropped": 0}, 0
            gt = np.zeros((S, S), dtype=np.uint8)
        polys = letterbox_polys(polys_raw, geo)
        pred = rasterize(polys)
        inter = int(np.logical_and(gt, pred).sum())
        union = int(np.logical_or(gt, pred).sum())
        iou = inter / union if union else 1.0
        # mask vs 官方 polygon_coordinate 交叉校验（原生分辨率）
        ti = float("nan")
        if it["has_mask"]:
            tp = read_polygon_txt(it["src_poly"])
            if tp is not None:
                cv = np.zeros(gt.shape if False else (h, w), dtype=np.uint8)
                cv2.fillPoly(cv, [np.round(tp).astype(np.int32)], 1)
                g0 = (mask > 0).astype(np.uint8)
                a = int(np.logical_and(g0, cv).sum()); b = int(np.logical_or(g0, cv).sum())
                ti = a / b if b else 1.0
                txt_ious.append(ti)
        ious.append(iou)
        tot_inst += st["instances"]
        # Compare polygon rasterisation at the original mask resolution.
        nat = float("nan")
        if it["has_mask"]:
            g0 = (mask > 0).astype(np.uint8)
            pr0 = np.zeros((h, w), dtype=np.uint8)
            for _v, pts in polys_raw:
                cv2.fillPoly(pr0, [np.round(pts).astype(np.int32)], 1)
            a0 = int(np.logical_and(g0, pr0).sum()); b0 = int(np.logical_or(g0, pr0).sum())
            nat = a0 / b0 if b0 else 1.0
            nat_ious.append(nat)
        print(f"  {it['stem']:<14}{f'{w}x{h}':>11}{f'{nw}x{nh}':>13}{up:>7.2f}{fills[-1]:>7.3f}"
              f"{st['instances']:>5}{st['contours']:>5}{iou:>8.4f}{nat:>8.4f}{ti:>8.4f}{cc:>4}")
    print(f"\n  letterbox IoU  min={min(ious):.4f} mean={statistics.mean(ious):.4f}")
    if nat_ious:
        print(f"Native-resolution round-trip IoU: {statistics.mean(nat_ious):.4f}")
    if txt_ious:
        print(f"  vs 官方 polygon_coordinate  IoU  mean={statistics.mean(txt_ious):.4f} "
              f"min={min(txt_ious):.4f}")
    print(f"  上采样倍数 min/mean/max = {min(ups):.2f} / {statistics.mean(ups):.2f} / {max(ups):.2f}")
    print(f"  画布填充   mean={statistics.mean(fills):.3f}")
    print(f"  抽样实例数合计 = {tot_inst}")


# --------------------------------------------------------------------------- build
def main(force: bool):
    items = scan_source()
    print(f"[scan] {len(items)} images "
          f"(landslide {sum(1 for i in items if i['has_mask'])} / "
          f"non-landslide {sum(1 for i in items if not i['has_mask'])})")

    for it in items:
        it["n_instances"] = instance_count(it)
        it["bucket"] = bucket_of(it["n_instances"])
    print("[scan] instance counts done:",
          dict(sorted(collections.Counter(i["bucket"] for i in items).items(),
                      key=lambda kv: [b[0] for b in BUCKETS].index(kv[0]))))

    groups: dict[str, list] = {}
    for it in items:
        groups.setdefault(it["bucket"], []).append(it)
    train, val, target = largest_remainder_split(groups, VAL_RATIO, SEED)
    print(f"[split] target val={target} ({VAL_RATIO:.1%} of {len(items)}) "
          f"-> train={len(train)} val={len(val)} ({len(val)/len(items):.4%})")

    if os.path.exists(OUT_ROOT):
        if not force:
            sys.exit(f"[ERR] output root exists, pass --force: {OUT_ROOT}")
        import shutil
        shutil.rmtree(OUT_ROOT)
    dirs = {}
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = os.path.join(OUT_ROOT, split, kind)
            os.makedirs(d, exist_ok=True)
            dirs[(split, kind)] = d

    manifest, agg = [], []
    for split, bucket in (("train", train), ("val", val)):
        print(f"[convert] {split}: {len(bucket)} images")
        for i, it in enumerate(bucket, 1):
            im = Image.open(it["src_image"]).convert("RGB")
            w, h = im.size
            geo = letterbox((w, h))
            _sx, _sy, nw, nh, dx, dy = geo

            lb = letterbox_image(im, geo)
            assert lb.size == (S, S), lb.size
            lb.save(os.path.join(dirs[(split, "images")], it["out_stem"] + ".png"), optimize=True)

            if it["has_mask"]:
                mask = np.array(Image.open(it["src_mask"]))
                polys_raw, st = mask_to_polygons(mask)
                gt = reference_fg(mask, geo)
                cc = int(cv2.connectedComponents((mask > 0).astype(np.uint8),
                                                 connectivity=8)[0] - 1)
            else:
                polys_raw, st, gt, cc = [], {"instances": 0, "contours": 0,
                                             "dropped": 0, "multi_part": 0}, \
                    np.zeros((S, S), dtype=np.uint8), 0
            polys = letterbox_polys(polys_raw, geo)
            with open(os.path.join(dirs[(split, "labels")], it["out_stem"] + ".txt"),
                      "w", encoding="utf-8", newline="\n") as f:
                f.write(polygons_to_line(polys))

            pred = rasterize(polys)
            inter = int(np.logical_and(gt, pred).sum())
            union = int(np.logical_or(gt, pred).sum())
            rec = {
                "split": split, "subset": it["subset"], "region": it["region"],
                "out_stem": it["out_stem"],
                "src_width": w, "src_height": h,
                "nw": nw, "nh": nh, "dx": dx, "dy": dy,
                "upscale": nw / w,
                "out_width": S, "out_height": S,
                "canvas_fill": nw * nh / (S * S),
                "n_instances": st["instances"],
                "n_polygons": st["contours"],
                "connected_components": cc,
                "dropped_contours": st["dropped"],
                "clip_iou": inter / union if union else 1.0,
                "bucket": it["bucket"],
                "source_image": os.path.relpath(it["src_image"], SRC_ROOT).replace(os.sep, "/"),
                "source_mask": os.path.relpath(it["src_mask"], SRC_ROOT).replace(os.sep, "/") if it["src_mask"] else "",
            }
            manifest.append(rec); agg.append(rec)
            if i % 300 == 0:
                print(f"    ... {i}/{len(bucket)}")

    # ---------------------------------------------------------------- audit
    audit = {"unpaired": 0, "not_1024": 0, "out_of_range": 0, "degenerate_polygons": 0,
             "empty_labels": 0, "nonempty_labels": 0,
             "instances_positives": 0, "negatives": 0, "cc_gt1": 0}
    for split in ("train", "val"):
        imgs = sorted(os.path.splitext(f)[0] for f in os.listdir(dirs[(split, "images")]))
        lbls = sorted(os.path.splitext(f)[0] for f in os.listdir(dirs[(split, "labels")]))
        audit["unpaired"] += len(set(imgs) ^ set(lbls))
        for nm in imgs:
            with Image.open(os.path.join(dirs[(split, "images")], nm + ".png")) as im:
                if im.size != (S, S):
                    audit["not_1024"] += 1
        for nm in lbls:
            with open(os.path.join(dirs[(split, "labels")], nm + ".txt"), encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                audit["empty_labels"] += 1
                continue
            audit["nonempty_labels"] += 1
            for line in content.splitlines():
                parts = line.split()
                if not parts:
                    continue
                c = [float(x) for x in parts[1:]]
                if len(c) < 6 or len(c) % 2:
                    audit["degenerate_polygons"] += 1
                    continue
                for v in c:
                    if not (0.0 <= v <= 1.0):
                        audit["out_of_range"] += 1
    audit["instances_positives"] = sum(r["n_instances"] for r in agg)
    audit["negatives"] = sum(1 for r in agg if r["n_instances"] == 0 and r["subset"] == "non-landslide")
    audit["cc_gt1"] = sum(1 for r in agg if r["connected_components"] > 1)

    ups = [r["upscale"] for r in agg]
    fills = [r["canvas_fill"] for r in agg]
    ious = [r["clip_iou"] for r in agg if r["n_instances"] >= 0]
    pos_ious = [r["clip_iou"] for r in agg if r["subset"] == "landslide"]

    def q(v, p):
        return float(np.percentile(np.array(v), p))

    summary = {
        "format": "compare_dataset_bijie_v1_1024",
        "geometry": {
            "mode": "aspect-preserving letterbox, centred, black pad (same rule as LRSTTC v2)",
            "tile_size": S, "no_distortion": True, "no_content_loss": True,
            "no_instance_truncation": True,
            "resample": "LANCZOS when downscaling, BICUBIC when upscaling",
            "note": "⚠️ patch-based dataset: 'fill the canvas' UPSCALES strongly (median ~3.6x, "
                    "max 16.8x, 94.7% of images >2x) — see geometry_stats for exact values. "
                    "LRSTTC was the opposite (median 0.95x downscale).",
        },
        "source_root": "<src>",
        "output_root": ".",
        "subsets": [{"dir": s, "has_annotation": m} for s, m in SUBSETS],
        "class_scheme": {"nc": 1, "names": {0: "landslide"}},
        "label_encoding": "yolo-seg polygon, one line per polygon, normalized xy, "
                          "empty file = negative",
        "polygon_extraction": {
            "source": "instance mask (binary 0/255); one connected component per positive image",
            "contour_mode": "RETR_EXTERNAL", "min_area_px2": MIN_AREA, "approx_eps_rel": EPS_REL,
            "cross_check": "The native polygons are compared with the foreground mask.",
        },
        "split": {
            "method": "per-image instance-count bucketing, largest-remainder allocation of 12.5%",
            "buckets": [b[0] for b in BUCKETS], "seed": SEED,
            "requested_val_ratio": VAL_RATIO, "target_val_images": target,
        },
        "totals": {
            "images": len(agg),
            "train_images": sum(1 for r in agg if r["split"] == "train"),
            "val_images": sum(1 for r in agg if r["split"] == "val"),
            "actual_val_ratio": sum(1 for r in agg if r["split"] == "val") / len(agg),
            "landslide_images": sum(1 for r in agg if r["subset"] == "landslide"),
            "nonlandslide_images": sum(1 for r in agg if r["subset"] == "non-landslide"),
            "instances": audit["instances_positives"],
            "val_instances": sum(r["n_instances"] for r in agg if r["split"] == "val"),
            "negatives": audit["negatives"],
            "negative_ratio": audit["negatives"] / len(agg),
            "polygons": sum(r["n_polygons"] for r in agg),
        },
        "geometry_stats": {
            "upscale_min": min(ups), "upscale_median": q(ups, 50), "upscale_max": max(ups),
            "n_upscale_gt2": sum(1 for x in ups if x > 2),
            "n_upscale_gt4": sum(1 for x in ups if x > 4),
            "n_upscale_gt8": sum(1 for x in ups if x > 8),
            "canvas_fill_p05": q(fills, 5), "canvas_fill_median": q(fills, 50),
            "canvas_fill_mean": statistics.mean(fills),
            "clip_iou_mean": statistics.mean(pos_ious), "clip_iou_min": min(pos_ious),
        },
        "audit": audit,
        "per_bucket": [], "per_subset": [], "per_region": [],
    }

    for label, _lo, _hi in BUCKETS:
        pool = [r for r in agg if r["bucket"] == label]
        if not pool:
            continue
        v = [r for r in pool if r["split"] == "val"]
        summary["per_bucket"].append({
            "bucket": label, "images": len(pool), "val_images": len(v),
            "actual_val_ratio": len(v) / len(pool),
            "instances": sum(r["n_instances"] for r in pool),
            "val_instances": sum(r["n_instances"] for r in v)})

    for sub, _m in SUBSETS:
        pool = [r for r in agg if r["subset"] == sub]
        v = [r for r in pool if r["split"] == "val"]
        summary["per_subset"].append({
            "subset": sub, "images": len(pool),
            "train_images": len(pool) - len(v), "val_images": len(v),
            "actual_val_ratio": len(v) / len(pool),
            "instances": sum(r["n_instances"] for r in pool),
            "val_instances": sum(r["n_instances"] for r in v)})

    regions = sorted({r["region"] for r in agg})
    for reg in regions:
        pool = [r for r in agg if r["region"] == reg]
        v = [r for r in pool if r["split"] == "val"]
        summary["per_region"].append({
            "region": reg, "images": len(pool),
            "landslide_images": sum(1 for r in pool if r["subset"] == "landslide"),
            "nonlandslide_images": sum(1 for r in pool if r["subset"] == "non-landslide"),
            "val_images": len(v), "instances": sum(r["n_instances"] for r in pool)})

    with open(os.path.join(OUT_ROOT, "split_summary_1024.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_ROOT, "letterbox_manifest.csv"), "w",
              encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        wr.writeheader(); wr.writerows(manifest)
    with open(os.path.join(OUT_ROOT, "Bijie.yaml"), "w", encoding="utf-8", newline="\n") as f:
        # With no path override, Ultralytics resolves the splits from this YAML file.
        f.write("train: train/images\nval: val/images\nnc: 1\nnames:\n  0: landslide\n")

    print("\n================ SUMMARY ================")
    print(json.dumps({k: summary[k] for k in
                      ("totals", "geometry_stats", "audit",
                       "per_bucket", "per_subset")}, ensure_ascii=False, indent=2))
    print(f"\n  区域分布（前 12）: "
          f"{sorted(summary['per_region'], key=lambda x: -x['images'])[:12]}")
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
