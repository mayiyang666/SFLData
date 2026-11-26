#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Letterbox the LRSTTC image and mask on the same canvas, retaining the original split manifest."""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# reuse the v1 polygon semantics verbatim (RETR_EXTERNAL + approxPolyDP, EPS_REL=0.001)
from build_compare_lrsttc import BUCKETS, EPS_REL, MIN_AREA, mask_to_polygons  # noqa: E402

SRC_ROOT = "data/external/raw/LRSTTC_dataset"
V1_ROOT = "data/Compare_dataset/LRSTTC"
OUT_ROOT = "data/Compare_dataset_1024/LRSTTC"

S = 1024          # protocol input_size; every output tile is exactly this
PAD_FILL = (0, 0, 0)


# --------------------------------------------------------------------------- geometry
def letterbox(size: tuple[int, int], s: int = S):
    """(w,h) -> (sx, sy, nw, nh, dx, dy) for centred aspect-preserving letterbox."""
    w, h = size
    scale = min(s / w, s / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    nw, nh = min(nw, s), min(nh, s)
    dx, dy = (s - nw) // 2, (s - nh) // 2
    return nw / w, nh / h, nw, nh, dx, dy          # sx == sy == scale (up to rounding)


def letterbox_image(im: Image.Image, geo) -> Image.Image:
    sx, sy, nw, nh, dx, dy = geo
    resample = Image.LANCZOS if max(nw, nh) < max(im.size) else Image.BICUBIC
    canvas = Image.new("RGB", (S, S), PAD_FILL)
    canvas.paste(im.resize((nw, nh), resample), (dx, dy))
    return canvas


def letterbox_polys(polys, geo):
    sx, sy, nw, nh, dx, dy = geo
    out = []
    for inst, pts in polys:
        q = pts.copy()
        q[:, 0] = np.clip(q[:, 0] * sx + dx, 0, S - 1)
        q[:, 1] = np.clip(q[:, 1] * sy + dy, 0, S - 1)
        out.append((inst, q))
    return out


def polygons_to_line(polys, s: int = S, cls: int = 0) -> str:
    lines = []
    for _inst, pts in polys:
        norm = np.clip(pts / float(s), 0.0, 1.0)
        flat = []
        for x, y in norm:
            flat.append(repr(float(x)))
            flat.append(repr(float(y)))
        lines.append(f"{cls} " + " ".join(flat))
    return "\n".join(lines)


def rasterize(polys, s: int = S) -> np.ndarray:
    canvas = np.zeros((s, s), dtype=np.uint8)
    for _inst, pts in polys:
        cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
    return canvas


def reference_fg(mask: np.ndarray, geo) -> np.ndarray:
    """Letterboxed binary foreground = the image-space ground truth the model sees."""
    _sx, _sy, nw, nh, dx, dy = geo
    fg = (mask > 0).astype(np.uint8) * 255
    fg = cv2.resize(fg, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((S, S), dtype=np.uint8)
    canvas[dy:dy + nh, dx:dx + nw] = (fg > 0).astype(np.uint8)
    return canvas


# --------------------------------------------------------------------------- manifest
def load_v1_manifest() -> list[dict]:
    path = os.path.join(V1_ROOT, "split_manifest.csv")
    if not os.path.isfile(path):
        sys.exit(f"[ERR] v1 split manifest not found: {path}")
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        # The split and filenames are stable even when a legacy manifest refers to another machine.
        for field, folder in (("source_image", "pic"), ("source_mask", "mask")):
            name = r[field].replace("\\", "/").rsplit("/", 1)[-1]
            relative = "/".join((r["subset"], folder, name))
            candidate = os.path.join(SRC_ROOT, relative)
            # Older manifests may still point to a valid source outside the default root.
            if not os.path.isfile(candidate) and os.path.isfile(r[field]):
                candidate = r[field]
            r[field] = candidate
            r[field + "_relative"] = relative
        r["n_instances"] = int(r["n_instances"])
        r["n_polygons"] = int(r["n_polygons"])
    return rows


# --------------------------------------------------------------------------- selftest
def selftest(n: int):
    rows = load_v1_manifest()
    rows = rows[:: max(1, len(rows) // n)][:n]
    print(f"[selftest] {len(rows)} images | S={S} | EPS_REL={EPS_REL} | centred letterbox\n")
    print(f"  {'stem':<14}{'src w x h':>12}{'-> nw x nh':>13}{'dx,dy':>10}"
          f"{'area x':>8}{'fill':>7}{'inst':>6}{'poly':>6}{'IoU':>8}")
    ious, effs, seen = [], [], []
    tot_inst = tot_poly = 0
    for r in rows:
        im = Image.open(r["source_image"]).convert("RGB")
        w, h = im.size
        mask = np.array(Image.open(r["source_mask"]))
        geo = letterbox((w, h))
        _sx, _sy, nw, nh, dx, dy = geo

        polys_raw, st = mask_to_polygons(mask)
        polys = letterbox_polys(polys_raw, geo)

        gt = reference_fg(mask, geo)
        pred = rasterize(polys)
        inter = int(np.logical_and(gt, pred).sum())
        union = int(np.logical_or(gt, pred).sum())
        iou = inter / union if union else 1.0

        eff = (nw * nh) / (S * S)                 # effective (non-pad) fraction of the canvas
        vis = (nw * nh) / (w * h) if w * h else 1.0   # >1 = upsampled, <1 = downsampled
        ious.append(iou); effs.append(eff); seen.append(vis)
        tot_inst += st["instances"]; tot_poly += st["contours"]
        print(f"  {r['out_stem']:<14}{f'{w}x{h}':>12}{f'{nw}x{nh}':>13}"
              f"{f'{dx},{dy}':>10}{vis:>8.3f}{eff:>7.3f}{st['instances']:>6}"
              f"{st['contours']:>6}{iou:>8.4f}")
    print(f"\n  images={len(rows)} instances={tot_inst} polygons={tot_poly}")
    print(f"  letterbox IoU  min={min(ious):.4f}  mean={statistics.mean(ious):.4f}  max={max(ious):.4f}")
    print(f"  resample area  min={min(seen):.4f}  mean={statistics.mean(seen):.4f}"
          f"   (>1 upsampled / <1 downsampled; source content is never discarded)")
    print(f"  canvas fill    min={min(effs):.4f}  mean={statistics.mean(effs):.4f}   (1.0 = no black bars)")


# --------------------------------------------------------------------------- build
def main(force: bool):
    rows = load_v1_manifest()
    print(f"[load] v1 manifest: {len(rows)} images "
          f"(train {sum(1 for r in rows if r['split']=='train')} / "
          f"val {sum(1 for r in rows if r['split']=='val')})")

    if os.path.exists(OUT_ROOT):
        if not force:
            sys.exit(f"[ERR] output root exists, pass --force to overwrite: {OUT_ROOT}")
        import shutil
        shutil.rmtree(OUT_ROOT)

    dirs = {}
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = os.path.join(OUT_ROOT, split, kind)
            os.makedirs(d, exist_ok=True)
            dirs[(split, kind)] = d

    manifest, agg = [], []
    for split in ("train", "val"):
        bucket = [r for r in rows if r["split"] == split]
        print(f"[convert] {split}: {len(bucket)} images")
        for i, r in enumerate(bucket, 1):
            im = Image.open(r["source_image"]).convert("RGB")
            w, h = im.size
            mask = np.array(Image.open(r["source_mask"]))
            geo = letterbox((w, h))
            _sx, _sy, nw, nh, dx, dy = geo

            lb = letterbox_image(im, geo)
            assert lb.size == (S, S), lb.size
            out_img = os.path.join(dirs[(split, "images")], r["out_stem"] + ".png")
            lb.save(out_img, optimize=True)

            polys_raw, st = mask_to_polygons(mask)
            polys = letterbox_polys(polys_raw, geo)
            text = polygons_to_line(polys)
            with open(os.path.join(dirs[(split, "labels")], r["out_stem"] + ".txt"),
                      "w", encoding="utf-8", newline="\n") as f:
                f.write(text)

            gt = reference_fg(mask, geo)
            pred = rasterize(polys)
            inter = int(np.logical_and(gt, pred).sum())
            union = int(np.logical_or(gt, pred).sum())
            rec = {
                "split": split,
                "subset": r["subset"],
                "out_stem": r["out_stem"],
                "src_width": w, "src_height": h,
                "nw": nw, "nh": nh, "dx": dx, "dy": dy,
                "sx": nw / w, "sy": nh / h,
                "out_width": S, "out_height": S,
                "resample_area_ratio": (nw * nh) / (w * h) if w * h else 1.0,
                "canvas_fill": (nw * nh) / (S * S),
                "n_instances": st["instances"],
                "n_polygons": st["contours"],
                "dropped_contours": st["dropped"],
                "clip_iou": inter / union if union else 1.0,
                "bucket": r["bucket"],
                "source_image": r["source_image_relative"],
                "source_mask": r["source_mask_relative"],
            }
            manifest.append(rec); agg.append(rec)
            if i % 200 == 0:
                print(f"    ... {i}/{len(bucket)}")

    # ---------------------------------------------------------------- audit
    audit = {"unpaired": 0, "not_1024": 0, "out_of_range": 0, "degenerate_polygons": 0,
             "empty_labels": 0, "instances_manifest": 0, "instances_v1": 0,
             "polygons": 0, "dropped_contours": 0}
    for split in ("train", "val"):
        imgs = sorted(os.path.splitext(f)[0] for f in os.listdir(dirs[(split, "images")]))
        lbls = sorted(os.path.splitext(f)[0] for f in os.listdir(dirs[(split, "labels")]))
        audit["unpaired"] += len(set(imgs) ^ set(lbls))
        for nm in imgs:                                  # verify real on-disk size
            with Image.open(os.path.join(dirs[(split, "images")], nm + ".png")) as im:
                if im.size != (S, S):
                    audit["not_1024"] += 1
        for nm in lbls:
            with open(os.path.join(dirs[(split, "labels")], nm + ".txt"), encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                audit["empty_labels"] += 1
                continue
            for line in content.splitlines():
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
    audit["instances_manifest"] = sum(r["n_instances"] for r in agg)
    audit["instances_v1"] = sum(r["n_instances"] for r in rows)
    audit["polygons"] = sum(r["n_polygons"] for r in agg)
    audit["dropped_contours"] = sum(r["dropped_contours"] for r in agg)

    ious = [r["clip_iou"] for r in agg]
    kept = [r["resample_area_ratio"] for r in agg]
    fill = [r["canvas_fill"] for r in agg]

    def q(v, p):
        return float(np.percentile(np.array(v), p))

    summary = {
        "format": "compare_dataset_lrsttc_v2_1024",
        "geometry": {
            "mode": "aspect-preserving letterbox, centred, black pad",
            "tile_size": S,
            "no_distortion": True,
            "no_content_loss": True,
            "no_instance_truncation": True,
            "resample": "LANCZOS when downscaling, BICUBIC when upscaling",
            "note": "all source pixels are used; the pad is zero-filled background and "
                    "carries no GT instances",
        },
        "source_root": "<src>",
        "v1_manifest": "<v1-root>/split_manifest.csv",
        "output_root": ".",
        "class_scheme": {"nc": 1, "names": {0: "landslide"}},
        "label_encoding": "yolo-seg polygon, one line per polygon, normalized xy, empty file = negative",
        "polygon_extraction": {"contour_mode": "RETR_EXTERNAL",
                               "min_area_px2": MIN_AREA, "approx_eps_rel": EPS_REL},
        "split": {"reused_from": "v1 split_manifest.csv (per-image instance-count bucketing, "
                                 "largest-remainder 12.5%, seed=42)",
                  "buckets": [b[0] for b in BUCKETS]},
        "totals": {
            "images": len(agg),
            "train_images": sum(1 for r in agg if r["split"] == "train"),
            "val_images": sum(1 for r in agg if r["split"] == "val"),
            "actual_val_ratio": sum(1 for r in agg if r["split"] == "val") / len(agg),
            "instances": audit["instances_manifest"],
            "val_instances": sum(r["n_instances"] for r in agg if r["split"] == "val"),
            "polygons": audit["polygons"],
        },
        "geometry_stats": {
            "clip_iou_mean": statistics.mean(ious),
            "clip_iou_min": min(ious),
            "resample_area_ratio_mean": statistics.mean(kept),
            "resample_area_ratio_min": min(kept),
            "resample_area_ratio_max": max(kept),
            "canvas_fill_mean": statistics.mean(fill),
            "canvas_fill_p05": q(fill, 5),
            "canvas_fill_p25": q(fill, 25),
            "canvas_fill_median": q(fill, 50),
            "canvas_fill_p75": q(fill, 75),
            "n_upsampled": sum(1 for x in kept if x > 1.0),
            "n_downsampled": sum(1 for x in kept if x < 1.0),
            "n_same_scale": sum(1 for x in kept if x == 1.0),
        },
        "vs_v1": {
            "instances_v1": audit["instances_v1"],
            "instances_v2": audit["instances_manifest"],
            "image_count_v1": len(rows),
            "image_count_v2": len(agg),
        },
        "audit": audit,
        "per_bucket": [],
        "per_subset": [],
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

    for sub in ("new_landslides", "old_landslides"):
        pool = [r for r in agg if r["subset"] == sub]
        if not pool:
            continue
        v = [r for r in pool if r["split"] == "val"]
        summary["per_subset"].append({
            "subset": sub, "images": len(pool),
            "train_images": len(pool) - len(v), "val_images": len(v),
            "actual_val_ratio": len(v) / len(pool),
            "instances": sum(r["n_instances"] for r in pool),
            "val_instances": sum(r["n_instances"] for r in v)})

    with open(os.path.join(OUT_ROOT, "split_summary_1024.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_ROOT, "letterbox_manifest.csv"), "w",
              encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        wr.writeheader(); wr.writerows(manifest)
    with open(os.path.join(OUT_ROOT, "LRSTTC.yaml"), "w", encoding="utf-8", newline="\n") as f:
        # With no path override, Ultralytics resolves the splits from this YAML file.
        f.write("train: train/images\nval: val/images\nnc: 1\nnames:\n  0: landslide\n")

    print("\n================ SUMMARY ================")
    print(json.dumps({k: summary[k] for k in
                      ("totals", "geometry_stats", "vs_v1", "audit",
                       "per_bucket", "per_subset")}, ensure_ascii=False, indent=2))
    print(f"\n[ok] dataset -> {OUT_ROOT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", type=int, default=0, metavar="N")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--src", default=SRC_ROOT,
                    help=f"源数据根目录（默认 {SRC_ROOT}）")
    ap.add_argument("--v1-root", default=V1_ROOT,
                    help=f"v1（非 1024）构建产物目录，需含 split_manifest.csv"
                         f"（默认 {V1_ROOT}）；若不存在请先跑 build_compare_lrsttc.py")
    ap.add_argument("--out", default=OUT_ROOT,
                    help=f"输出根目录（默认 {OUT_ROOT}）")
    a = ap.parse_args()
    globals()["SRC_ROOT"] = a.src
    globals()["V1_ROOT"] = a.v1_root
    globals()["OUT_ROOT"] = a.out
    if a.selftest:
        selftest(a.selftest)
    else:
        main(a.force)
