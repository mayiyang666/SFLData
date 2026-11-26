#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert Landslide4Sense HDF5 bands B4, B3 and B2 to RGB; use --calib before --build."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from PIL import Image

SRC = "data/external/raw/landslide4sense"
OUT = "data/external/preconverted/landslide4sense_png"
CALIB = os.path.join(OUT, "ls4s_calib.json")
PREVIEW = os.path.join(OUT, "preview")

# Sentinel-2 -> 通道索引（0-based）。rgb=(红,绿,蓝)
CH_RGB = (3, 2, 1)
SPLITS = {"train": "train", "val": "validation", "test": "test"}   # 输出名 -> 源目录名

PNG_COMPRESS = 3
BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9",
              "B11", "B12", "slope", "DEM"]


# ------------------------------------------------------------------ IO
def h5_read(path, kind):
    with h5py.File(path, "r") as f:
        k = list(f.keys())[0]
        return np.asarray(f[k][()])


def img_path(src_split, i):
    return os.path.join(SRC, "images", src_split, "image_%d.h5" % i)


def msk_path(src_split, i):
    return os.path.join(SRC, "annotations", src_split, "mask_%d.h5" % i)


def list_ids(src_split):
    d = os.path.join(SRC, "images", src_split)
    ids = []
    for f in os.listdir(d):
        stem = os.path.splitext(f)[0]
        ids.append(int(stem.split("_")[1]))
    return sorted(ids)


# ------------------------------------------------------------------ calib
# 用「全局像素池化」直方图求分位（而不是逐图分位的均值/中位数）。
HIST_RANGE = (0.0, 16.0)
HIST_BINS = 512
HIST_W = (HIST_RANGE[1] - HIST_RANGE[0]) / HIST_BINS


def calib_worker(image_file):
    im = h5_read(image_file, "img")
    if im.ndim == 2:
        im = im[:, :, None]
    hists = []
    for c in CH_RGB:
        h, _ = np.histogram(np.clip(im[:, :, c], *HIST_RANGE), bins=HIST_BINS,
                            range=HIST_RANGE)
        hists.append(h.astype(np.int64))
    return hists


def _pooled_quantile(hist, qs):
    """hist: (HIST_BINS,) 计数 -> 指定百分位的值。"""
    c = np.cumsum(hist)
    tot = c[-1]
    out = []
    for q in qs:
        target = tot * q / 100.0
        idx = int(np.searchsorted(c, target))
        idx = min(idx, HIST_BINS - 1)
        out.append(HIST_RANGE[0] + (idx + 0.5) * HIST_W)
    return out


def run_calib(workers, qs):
    agg_hist = [np.zeros(HIST_BINS, np.int64) for _ in CH_RGB]
    jobs = []
    for out_sp, src_sp in SPLITS.items():
        for i in list_ids(src_sp):
            # Resolve inputs in the parent so spawned workers honour --src on every platform.
            jobs.append(img_path(src_sp, i))
    print("[calib] %d 张，逐通道**全局池化** p%s 全量统计..." % (len(jobs), qs), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, r in enumerate(ex.map(calib_worker, jobs, chunksize=32), 1):
            for j in range(len(CH_RGB)):
                agg_hist[j] += r[j]
            if k % 1000 == 0:
                print("   ... %d/%d (%.0fs)" % (k, len(jobs), time.time() - t0), flush=True)

    probe_q = sorted(set([0.1, 0.5, 1.0, 2.0, 5.0, 50.0, 95.0, 98.0, 99.0, 99.5] + list(qs)))
    out = {}
    for j, c in enumerate(CH_RGB):
        vq = _pooled_quantile(agg_hist[j], probe_q)
        out[str(c)] = {"band": BAND_NAMES[c],
                       "n_px": int(agg_hist[j].sum()),
                       "probe_q": probe_q,
                       "probe_v": vq,
                       "q": [vq[probe_q.index(q)] for q in qs]}
    out["joint"] = {"lo": min(out[str(c)]["q"][0] for c in CH_RGB),
                    "hi": max(out[str(c)]["q"][-1] for c in CH_RGB)}
    out["_note"] = ("q = 全局像素池化分位（所有 4844 张图的所有像素合并后求分位），"
                    "hist range %s bins %d。" % (HIST_RANGE, HIST_BINS))
    with open(CALIB, "w", encoding="utf-8") as f:
        json.dump({"quantiles": qs, "n_images": len(jobs), "channels": out,
                   "ch_rgb": list(CH_RGB)}, f, ensure_ascii=False, indent=2)
    print("[calib] -> %s" % CALIB)
    for c in CH_RGB:
        print("   ch%02d %-5s  选中 %s | 全部分位 %s" % (
            c, BAND_NAMES[c],
            " ".join("p%g=%.4f" % (qs[q], out[str(c)]["q"][q]) for q in range(len(qs))),
            " ".join("p%g=%.2f" % (q, v) for q, v in zip(probe_q, out[str(c)]["probe_v"]))))
    print("   joint   lo=%.4f hi=%.4f" % (out["joint"]["lo"], out["joint"]["hi"]))
    return out


# ------------------------------------------------------------------ stretch
def load_calib():
    with open(CALIB, "r", encoding="utf-8") as f:
        return json.load(f)


def to_u8(im, mode, cal):
    """im:(H,W,3) float64 按 CH_RGB 顺序 -> uint8。"""
    chans = cal["channels"]
    if mode == "perchan":
        out = np.empty(im.shape[:2] + (3,), np.uint8)
        for j, c in enumerate(CH_RGB):
            lo, hi = chans[str(c)]["q"][0], chans[str(c)]["q"][-1]
            v = (im[:, :, j] - lo) / max(hi - lo, 1e-9)
            out[:, :, j] = np.clip(v * 255.0, 0, 255).astype(np.uint8)
        return out
    if mode == "joint":
        lo, hi = chans["joint"]["lo"], chans["joint"]["hi"]
        v = (im - lo) / max(hi - lo, 1e-9)
        return np.clip(v * 255.0, 0, 255).astype(np.uint8)
    if mode == "perimg":
        out = np.empty(im.shape[:2] + (3,), np.uint8)
        for j in range(3):
            a = im[:, :, j]
            lo, hi = float(a.min()), float(a.max())
            out[:, :, j] = np.clip((a - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
        return out
    raise SystemExit("未知 stretch: %s" % mode)


# ------------------------------------------------------------------ preview
def run_preview(cal, ids=(191, 303, 202, 351, 1, 2), up=6):
    os.makedirs(PREVIEW, exist_ok=True)
    for i in ids:
        im = h5_read(img_path("train", i), "img")
        m = h5_read(msk_path("train", i), "mask")
        panels = [("mask", (m > 0).astype(np.uint8) * 255)]
        for mode in ("perchan", "joint", "perimg"):
            rgb = im[:, :, list(CH_RGB)]
            panels.append((mode, to_u8(rgb, mode, cal)))
        gap, x = 6, 0
        ups = []
        for k, p in panels:
            u = np.asarray(Image.fromarray(p).resize(
                (p.shape[1] * up, p.shape[0] * up), Image.NEAREST))
            ups.append(np.repeat(u[:, :, None], 3, axis=2) if u.ndim == 2 else u)
        W = sum(u.shape[1] for u in ups) + gap * (len(ups) - 1)
        canvas = np.zeros((ups[0].shape[0], W, 3), np.uint8)
        for u in ups:
            canvas[:, x:x + u.shape[1]] = u
            x += u.shape[1] + gap
        fn = os.path.join(PREVIEW, "stretch_cmp_train_%04d.png" % i)
        Image.fromarray(canvas).save(fn)
        print("[preview] ->", fn, " 列序:", [p[0] for p in panels])


# ------------------------------------------------------------------ build
def _write_retry(fn, attempts=6, delay=0.25):
    last = None
    for k in range(attempts):
        try:
            return fn()
        except OSError as ex:
            last = ex
            time.sleep(delay * (k + 1))
    raise last


def build_worker(job):
    out_sp, image_file, mask_file, i, out_root, mode, calib_d, write = job
    im = h5_read(image_file, "img")
    m = h5_read(mask_file, "mask")
    if im.shape[:2] != m.shape[:2]:
        return {"err": "size mismatch %s" % i, "i": i}

    rgb = to_u8(im[:, :, list(CH_RGB)].astype(np.float64), mode, calib_d)
    mb = (m > 0).astype(np.uint8) * 255

    stem = "LS4S__%s_%06d" % (out_sp, i)
    if write:
        di = os.path.join(out_root, out_sp, "images")
        dm = os.path.join(out_root, out_sp, "masks")
        _write_retry(lambda: Image.fromarray(rgb).save(
            os.path.join(di, stem + ".png"), compress_level=PNG_COMPRESS))
        _write_retry(lambda: Image.fromarray(mb).save(
            os.path.join(dm, stem + ".png"), compress_level=PNG_COMPRESS))

    fg = int(np.count_nonzero(mb))
    return {"err": None, "i": i, "split": out_sp, "stem": stem,
            "src_shape": list(im.shape), "src_dtype": str(im.dtype),
            "fg_px_128": fg, "fg_ratio_128": fg / float(mb.size),
            "img_mean": float(rgb.mean()), "img_std": float(rgb.std()),
            "sat_frac": float(np.mean(rgb >= 254)),
            "black_frac": float(np.mean(rgb <= 1)),
            "rgb_md5": hashlib.md5(rgb.tobytes()).hexdigest()}


def run_build(cal, mode, workers, write=True):
    out_root = OUT
    if write:
        for sp in SPLITS:
            for kind in ("images", "masks"):
                os.makedirs(os.path.join(out_root, sp, kind), exist_ok=True)

    jobs = []
    for out_sp, src_sp in SPLITS.items():
        for i in list_ids(src_sp):
            jobs.append((out_sp, img_path(src_sp, i), msk_path(src_sp, i),
                         i, out_root, mode, cal, write))
    print("[build] %d 张，stretch=%s，%s" % (len(jobs), mode,
          "落盘" if write else "仅统计"), flush=True)

    agg, errs = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, r in enumerate(ex.map(build_worker, jobs, chunksize=32), 1):
            (errs if r.get("err") else agg).append(r)
            if k % 1000 == 0:
                el = time.time() - t0
                print("   ... %d/%d (%.0fs, %.1f/s)" % (k, len(jobs), el, k / el), flush=True)
    print("[build] 完成 %.0fs，错误 %d" % (time.time() - t0, len(errs)))
    if errs:
        print("   !! %s" % errs[:3])

    per = collections.defaultdict(lambda: {"n": 0, "fg": 0, "neg": 0, "sat": [], "blk": []})
    for r in agg:
        p = per[r["split"]]
        p["n"] += 1
        p["fg"] += r["fg_px_128"]
        p["neg"] += 1 if r["fg_px_128"] == 0 else 0
        p["sat"].append(r["sat_frac"])
        p["blk"].append(r["black_frac"])

    report = {"mode": mode, "write": write, "n": len(agg),
              "dupe_rgb": len(agg) - len({r["rgb_md5"] for r in agg})}
    print("\n[build] 逐 split：")
    for sp in SPLITS:
        p = per[sp]
        tot = p["n"] * 128 * 128
        report[sp] = {
            "images": p["n"], "negatives": p["neg"],
            "neg_ratio": p["neg"] / p["n"], "fg_ratio": p["fg"] / tot,
            "sat254_mean": statistics.mean(p["sat"]),
            "black_mean": statistics.mean(p["blk"]),
        }
        print("   %-5s n=%4d 负样本=%4d(%.1f%%) 前景占比=%.4f "
              "饱和(>=254)均值=%.5f 全黑(<=1)均值=%.5f" %
              (sp, p["n"], p["neg"], 100 * p["neg"] / p["n"], p["fg"] / tot,
               statistics.mean(p["sat"]), statistics.mean(p["blk"])))
    print("   RGB 重复图 = %d" % report["dupe_rgb"])
    dst = os.path.join(os.path.dirname(CALIB),
                       "preconvert_summary_%s%s.json" % (mode, "" if write else "_stats"))
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[build] -> %s" % dst)
    return report


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--stretch", default="perchan",
                    choices=["perchan", "joint", "perimg"])
    ap.add_argument("--quantiles", default="1,99")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--src", default=SRC,
                    help=f"Landslide4Sense 原始 h5 根目录（默认 {SRC}）")
    ap.add_argument("--out", default=OUT,
                    help=f"PNG 输出根目录（默认 {OUT}）；"
                         f"标定文件与预览目录一并落在该目录下")
    a = ap.parse_args()
    globals()["SRC"] = a.src
    globals()["OUT"] = a.out
    globals()["CALIB"] = os.path.join(a.out, "ls4s_calib.json")
    globals()["PREVIEW"] = os.path.join(a.out, "preview")
    os.makedirs(a.out, exist_ok=True)

    qs = [float(x) for x in a.quantiles.split(",")]
    if a.calib:
        run_calib(a.workers, qs)
    if a.preview:
        if not os.path.exists(CALIB):
            sys.exit("先跑 --calib")
        run_preview(load_calib())
    if a.build:
        if not os.path.exists(CALIB):
            sys.exit("先跑 --calib")
        run_build(load_calib(), a.stretch, a.workers, write=not a.stats_only)
    if not (a.calib or a.preview or a.build):
        ap.print_help()
