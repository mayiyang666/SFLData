#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare GDCLD tiles and labels using the official split (Fang et al., doi:10.5281/zenodo.13612636)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_compare_lrsttc import (  # noqa: E402
    BUCKETS, EPS_REL, MIN_AREA, bucket_of, mask_to_polygons,
)
from build_compare_lrsttc_1024 import (  # noqa: E402
    S, letterbox, letterbox_image, letterbox_polys, polygons_to_line, rasterize,
)

SRC_ROOT = "data/external/raw/GDCLD"
OUT_ROOT = "data/Compare_dataset_1024/GDCLD"

# 官方划分，原样沿用
SPLITS = {
    "train": (os.path.join(SRC_ROOT, "train_data", "train_data"),
              os.path.join(SRC_ROOT, "train_data", "train_label"),
              "_t"),
    "val": (os.path.join(SRC_ROOT, "val_data", "val_data"),
            os.path.join(SRC_ROOT, "val_data", "val_label"),
            "_s"),
}

PNG_COMPRESS = 3        # optimize=True 会慢 3~5 倍，收益极小
DARK_MEAN_G = 15.0      # mean gray 低于此值视为「黑图」（nodata / 拼接失败片）

Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------------------------------- 扫描
def scan_source():
    items, meta = [], {}
    for split, (idir, ldir, suffix) in SPLITS.items():
        imgs = sorted(f for f in os.listdir(idir) if f.lower().endswith((".tif", ".tiff")))
        labs = sorted(f for f in os.listdir(ldir) if f.lower().endswith((".tif", ".tiff")))
        si = {os.path.splitext(f)[0] for f in imgs}
        sl = {os.path.splitext(f)[0] for f in labs}
        bad_suffix = [s for s in si if not s.endswith(suffix)]
        meta[split] = {
            "n_img": len(imgs), "n_label": len(labs),
            "img_only": len(si - sl), "label_only": len(sl - si),
            "paired": len(si & sl),
            "suffix_mismatch": len(bad_suffix),
            "suffix_mismatch_examples": bad_suffix[:5],
        }
        for st in sorted(si & sl):
            items.append({"split": split, "stem": st, "out_stem": st,
                          "src_img": os.path.join(idir, st + ".tif"),
                          "src_label": os.path.join(ldir, st + ".tif")})
    return items, meta


# --------------------------------------------------------------------------- pass A
def probe_one(it: dict) -> dict:
    """只读标签掩码：连通域数、前景像素。用于分桶统计 + 审计期望值。"""
    with Image.open(it["src_label"]) as im:
        lw, lh = im.size
        lm = im.mode
        a = np.array(im)
    if a.ndim == 3:
        a = a[..., 0]
    b = (a > 0).astype(np.uint8)
    n_cc, _ = cv2.connectedComponents(b, connectivity=8)
    n_cc -= 1
    return {**it, "label_size": (lw, lh), "label_mode": lm,
            "n_cc": int(n_cc), "fg_px": int(b.sum())}


# --------------------------------------------------------------------------- pass B
def _write_retry(fn, attempts: int = 6, delay: float = 0.25):
    last = None
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:          # noqa: BLE001
            last = e
            time.sleep(delay * (k + 1))
    raise last


def render_one(job: dict) -> dict:
    """letterbox(恒等) + 掩码->多边形 + 写 PNG/标签 + 往返 IoU。"""
    split, out_stem = job["split"], job["out_stem"]
    with Image.open(job["src_img"]) as im:
        im = im.convert("RGB")
        src_size = im.size
        geo = letterbox(src_size)                      # (sx, sy, nw, nh, dx, dy)
        canvas = letterbox_image(im, geo)
        g = np.asarray(canvas, dtype=np.float32).mean(2)
        dark = float(g.mean())

    with Image.open(job["src_label"]) as lm:
        lab = np.array(lm)
    if lab.ndim == 3:
        lab = lab[..., 0]
    lab = (lab > 0).astype(np.uint8)

    polys, lstats = mask_to_polygons(lab, MIN_AREA, EPS_REL)
    polys = letterbox_polys(polys, geo)

    # 往返 IoU：栅格化多边形 vs letterbox 后的真实前景
    ref = np.zeros((S, S), np.uint8)
    sx, sy, nw, nh, dx, dy = geo
    fg = cv2.resize((lab * 255).astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    ref[dy:dy + nh, dx:dx + nw] = (fg > 0).astype(np.uint8)
    got = rasterize(polys, S)
    inter = int(np.logical_and(ref, got).sum())
    union = int(np.logical_or(ref, got).sum())
    rt_iou = (inter / union) if union else 1.0
    ref_px, got_px = int(ref.sum()), int(got.sum())

    img_dir = os.path.join(OUT_ROOT, split, "images")
    lab_dir = os.path.join(OUT_ROOT, split, "labels")
    if not job.get("dry"):
        _write_retry(lambda: canvas.save(os.path.join(img_dir, out_stem + ".png"),
                                         format="PNG", compress_level=PNG_COMPRESS))
        _write_retry(lambda: open(os.path.join(lab_dir, out_stem + ".txt"), "w",
                                  encoding="utf-8").write(polygons_to_line(polys, S, 0)))

    return {**job,
            "src_size": src_size, "geo": geo, "dark_mean_g": dark,
            "n_poly": len(polys), "n_cc": job.get("n_cc", 0),
            "dropped": lstats["dropped"], "contours": lstats["contours"],
            "fg_px_ref": ref_px, "fg_px_got": got_px, "inter": inter, "union": union,
            "rt_iou": rt_iou}


# --------------------------------------------------------------------------- 自检
def selftest(n: int):
    items, meta = scan_source()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'suffix_mismatch_examples'}
                      for k, v in meta.items()}, ensure_ascii=False, indent=2))
    for split in SPLITS:
        sub = [it for it in items if it["split"] == split]
        pick = sub[:: max(1, len(sub) // n)][:n]
        scales, ious, losses, darks = [], [], [], []
        for it in pick:
            r = render_one({**it, "n_cc": 0, "dry": True})
            scales.append(r["geo"][0])                 # sx == 线性缩放
            ious.append(r["rt_iou"])
            losses.append(abs(r["fg_px_got"] - r["fg_px_ref"]) / r["fg_px_ref"]
                          if r["fg_px_ref"] else 0.0)
            darks.append(r["dark_mean_g"])
        print(f"\n===== {split} 自检 {len(pick)} 张 =====")
        print(f"  letterbox 线性缩放: min={min(scales):.6f} max={max(scales):.6f} "
              f"（应为 1.0 —— 源图已是 1024²）")
        print(f"  往返 IoU: 均值={statistics.mean(ious):.4f} 中位={statistics.median(ious):.4f} "
              f"最小={min(ious):.4f}")
        print(f"  前景面积偏差: 均值={statistics.mean(losses)*100:.3f}% "
              f"中位={statistics.median(losses)*100:.3f}% 最大={max(losses)*100:.3f}%")
        print(f"  图像 meanG: 均值={statistics.mean(darks):.1f} 最小={min(darks):.1f} "
              f"（黑图 {sum(1 for d in darks if d < DARK_MEAN_G)} 张）")


# --------------------------------------------------------------------------- 正式构建
def main(force: bool, workers: int, write_outputs: bool):
    t0 = time.time()
    items, meta = scan_source()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'suffix_mismatch_examples'}
                      for k, v in meta.items()}, ensure_ascii=False, indent=2))

    # 父进程预建目录（worker 内绝不 makedirs —— Windows 并发建目录会竞态报 PermissionError）
    if write_outputs:
        for split in SPLITS:
            for sub in ("images", "labels"):
                d = os.path.join(OUT_ROOT, split, sub)
                if force and os.path.isdir(d):
                    import shutil
                    shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)

    tA = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        probed = list(ex.map(probe_one, items, chunksize=32))
    print(f"\n[pass A] {len(probed)} 张，{time.time()-tA:.1f} s")

    tB = time.time()
    if write_outputs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            done = list(ex.map(render_one, probed, chunksize=16))
        print(f"[pass B] 落盘 {len(done)} 张，{time.time()-tB:.1f} s")
    else:
        done = []
        for it in probed:
            it.setdefault("n_cc", it.get("n_cc", 0))
        print("[pass B] 跳过（--stats-only）")

    # 只算统计：rerender 一遍但丢弃写入（用 render_one 会写盘，故只在 write_outputs 时可用）
    if not write_outputs:
        print("\n（--stats-only：几何/保真度统计未复算，用已落盘的 split_manifest.csv）")
        for split in SPLITS:
            sub = [it for it in probed if it["split"] == split]
            n_fg = [it for it in sub if it["fg_px"] > 0]
            print(f"  {split}: 图 {len(sub)}  有前景 {len(n_fg)}  负样本 {len(sub)-len(n_fg)}  "
                  f"前景像素合计 {sum(it['fg_px'] for it in sub)}")

    # ---------------------------------------------------------------- 汇总
    agg = done if done else probed
    summary = {
        "format": "compare_dataset_gdcld_v1_1024_semantic",
        "task": "semantic segmentation (DeepLabV3+) —— 原论文明确 GDCLD 更适用语义而非实例分割",
        "source_root": "<src>",
        "output_root": ".",
        "citation": ("Fang, C., Fan, X., Wang, X., Nava, L., Zhong, H., Dong, X., Qi, J., "
                     "Catani, F. (2024) A globally distributed dataset of coseismic landslide "
                     "mapping via multi-source high-resolution remote sensing images, ESSD, "
                     "doi:10.5194/essd-2024-239 ; data: 10.5281/zenodo.13612636"),
        "split_policy": {
            "used": "OFFICIAL — GDCLD 自带的 75/25 划分，原样沿用，未重切",
            "why": ("官方划分本身是「按 patch 随机 75/25」，不是按事件/空间留出，"
                    "因此换成我们的 12.5% 分桶并不降低泄漏风险；选官方划分是为了与原文数字可比。"),
            "caveat": "Validation follows the official split.",
        },
        "source_recon": meta,
        "geometry": {
            "mode": "letterbox（恒等：源图已是 1024x1024）",
            "tile_size": 1024,
            "source_size": "ALL images are exactly 1024x1024 -> scale = 1.0, dx = dy = 0, "
                           "canvas fill = 1.0, ZERO black border, NO resampling",
            "resample": "none (identity)",
            "no_distortion": True,
            "no_content_loss": True,
            "advantage": ("四个数据集里唯一「几何零改动」的：无重采样 => 标签保真度只受"
                          "多边形编码影响，不受缩放影响"),
        },
        "class_scheme": {"nc": 1, "names": {"0": "landslide"},
                         "note": "语义: 0=background, 1=landslide；YOLO 标签类索引 0"},
        "label_encoding": "yolo-seg polygon，一行一个多边形，归一化 xy；每个 8 连通域一个多边形",
        "ground_truth_source": {"used": "{split}_label/*.tif（L 模式二值 {0,1}，1 = 滑坡）"},
        "polygon_extraction": {"contours": "RETR_EXTERNAL + approxPolyDP",
                               "min_area_px2": MIN_AREA, "approx_eps_rel": EPS_REL,
                               "rule_parity": "与 LRSTTC/Bijie/CAS 逐字同构"},
        "instance_count_definition": {
            "used": "滑坡前景的 8 连通域个数（仅用于分桶统计，不参与切分）",
            "buckets": [b[0] for b in BUCKETS],          # BUCKETS 是 (label, lo, hi) 元组列表
            "note": "官方划分固定，故不做最大余数法重切"},
        "totals": {}, "geometry_stats": {}, "per_bucket": {}, "audit": {},
        "timing": {"pass_a_s": round(time.time() - tA, 1),
                   "pass_b_s": round(time.time() - tB, 1),
                   "workers": workers,
                   "total_s": round(time.time() - t0, 1)},
    }

    for split in SPLITS:
        sub = [r for r in agg if r["split"] == split]
        n_neg = sum(1 for r in sub if r.get("fg_px", r.get("fg_px_ref", 0)) == 0)
        summary["totals"][split] = {
            "images": len(sub),
            "polygons": sum(r.get("n_poly", 0) for r in sub),
            "negatives": n_neg,
            "all_foreground": sum(1 for r in sub if r.get("fg_px_ref", 0) == S * S),
            "dropped_components": sum(r.get("dropped", 0) for r in sub),
            "images_with_dropped": sum(1 for r in sub if r.get("dropped", 0) > 0),
            "dark_images": sum(1 for r in sub if r.get("dark_mean_g", 999) < DARK_MEAN_G),
        }
        bk = {}
        for r in probed:
            if r["split"] != split:
                continue
            lab = bucket_of(r["n_cc"])
            bk[lab] = bk.get(lab, 0) + 1
        summary["per_bucket"][split] = {b[0]: bk.get(b[0], 0) for b in BUCKETS}

        if done:
            ious = [r["rt_iou"] for r in sub]
            losses = [(abs(r["fg_px_got"] - r["fg_px_ref"]) / r["fg_px_ref"])
                      if r["fg_px_ref"] else 0.0 for r in sub]
            ref_tot = sum(r["fg_px_ref"] for r in sub)
            pw = sum(abs(r["fg_px_got"] - r["fg_px_ref"]) for r in sub) / ref_tot if ref_tot else 0.0
            summary["geometry_stats"][split] = {
                "scale_min": min(r["geo"][0] for r in sub),
                "scale_max": max(r["geo"][0] for r in sub),
                "canvas_fill_median": statistics.median(r["geo"][2] / S for r in sub),
                "rt_iou_mean": statistics.mean(ious),
                "rt_iou_median": statistics.median(ious),
                "rt_iou_p05": float(np.percentile(ious, 5)),
                "rt_iou_min": min(ious),
                "fg_area_loss_mean": statistics.mean(losses),
                "fg_area_loss_median": statistics.median(losses),
                "fg_area_loss_p95": float(np.percentile(losses, 95)),
                "fg_area_loss_max": max(losses),
                "fg_area_loss_pixel_weighted": pw,
            }

    # 审计
    summary["audit"] = {
        "write_outputs": write_outputs,
        "source_img_only": sum(m["img_only"] for m in meta.values()),
        "source_label_only": sum(m["label_only"] for m in meta.values()),
        "source_suffix_mismatch": sum(m["suffix_mismatch"] for m in meta.values()),
    }
    if done:
        summary["audit"].update({
            "verify_bad_size": sum(1 for r in done if r["src_size"] != (S, S)),
            "verify_scale_not_1": sum(1 for r in done if abs(r["geo"][0] - 1.0) > 1e-9),
            "verify_expected_empty_but_filled":
                sum(1 for r in done if r.get("n_cc", 0) == 0 and r.get("n_poly", 0) > 0),
            "verify_expected_filled_but_empty":
                sum(1 for r in done if r.get("n_cc", 0) > 0 and r.get("n_poly", 0) == 0),
        })
        # 文件数校验
        bad = 0
        for split in SPLITS:
            n_img = len(os.listdir(os.path.join(OUT_ROOT, split, "images")))
            n_lab = len(os.listdir(os.path.join(OUT_ROOT, split, "labels")))
            exp = sum(1 for r in done if r["split"] == split)
            if n_img != exp or n_lab != exp:
                bad += 1
            print(f"  {split}: images={n_img} labels={n_lab} 期望={exp}")
        summary["audit"]["verify_count_mismatch"] = bad

    if write_outputs:
        with open(os.path.join(OUT_ROOT, "split_manifest.csv"), "w", newline="",
                  encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["split", "out_stem", "n_cc", "n_poly", "n_dropped",
                         "fg_px_ref", "fg_px_got", "inter", "union",
                         "rt_iou", "dark_mean_g"])
            for r in done:
                wr.writerow([r["split"], r["out_stem"], r.get("n_cc", 0), r.get("n_poly", 0),
                             r.get("dropped", 0), r.get("fg_px_ref", 0), r.get("fg_px_got", 0),
                             r.get("inter", 0), r.get("union", 0),
                             f"{r.get('rt_iou', 0):.6f}", f"{r.get('dark_mean_g', 0):.2f}"])
        with open(os.path.join(OUT_ROOT, "split_summary_1024.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n已写 split_manifest.csv + split_summary_1024.json")

    print("\n===== 汇总 =====")
    print(json.dumps({"totals": summary["totals"],
                      "geometry_stats": summary["geometry_stats"],
                      "audit": summary["audit"]}, ensure_ascii=False, indent=2))
    print(f"\n总耗时 {summary['timing']['total_s']} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--src", default=SRC_ROOT,
                    help=f"源数据根目录（默认 {SRC_ROOT}）")
    ap.add_argument("--out", default=OUT_ROOT,
                    help=f"输出根目录（默认 {OUT_ROOT}）")
    a = ap.parse_args()
    globals()["SRC_ROOT"] = a.src
    globals()["OUT_ROOT"] = a.out
    if a.selftest:
        selftest(a.selftest)
    else:
        main(a.force, a.workers, not a.stats_only)
