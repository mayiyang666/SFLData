#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare Landslide4Sense tiles from converted RGB images and masks. Use --split-mode official for the manuscript split."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 与 LRSTTC/Bijie/CAS/GDCLD/RLZX **同一套** letterbox、多边形、分桶与切分实现
from build_compare_lrsttc_1024 import (  # noqa: E402
    S, letterbox, letterbox_image, letterbox_polys, polygons_to_line, rasterize,
)
from build_compare_lrsttc import (  # noqa: E402
    BUCKETS, EPS_REL, MIN_AREA, bucket_of, largest_remainder_split, mask_to_polygons,
)

SRC_ROOT = "data/external/preconverted/landslide4sense_png"
OUT_ROOT = "data/Compare_dataset_1024/LS4S"

SEED, VAL_RATIO = 42, 0.125
PNG_COMPRESS = 3
MIN_AREA_USE = MIN_AREA
EPS_REL_USE = EPS_REL

REGION = "LS4S"
GSD_M = 10.0          # Sentinel-2 可见光/近红外原生 10 m（B5/B6/B7/B8A/B11/B12 为 20 m）
SENSOR = "Sentinel-2 MSI (10 m), true colour B4/B3/B2 of a 14-layer stack"

SRC_SPLITS = ["train", "val", "test"]      # 预转换输出名（= 源 train/validation/test）


# --------------------------------------------------------------------------- IO
def read_rgb_u8(path: str):
    """预转换产物已是 uint8 RGB PNG —— 直接读，保证与其它数据集同一入口语义。"""
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    a = np.asarray(im, dtype=np.uint8)
    return a, {"src_shape": list(a.shape), "src_dtype": "uint8"}


def load_mask(path: str) -> np.ndarray:
    """读 128x128 mask PNG。值域 {0,255}，**255 = 滑坡**（极性已实测确认，无需反相）。"""
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError("掩膜读取失败: %s" % path)
    if m.ndim == 3:
        m = m[:, :, 0]
    return ((m > 0).astype(np.uint8) * 255)


# --------------------------------------------------------------------------- scan
def scan_source():
    items, regions_meta, missing = [], [], []
    for sp in SRC_SPLITS:
        img_d = os.path.join(SRC_ROOT, sp, "images")
        msk_d = os.path.join(SRC_ROOT, sp, "masks")
        if not (os.path.isdir(img_d) and os.path.isdir(msk_d)):
            sys.exit("[ERR] 缺 images/masks: %s" % os.path.join(SRC_ROOT, sp))
        imgs = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(img_d))}
        msks = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(msk_d))}
        only_i, only_m = sorted(set(imgs) - set(msks)), sorted(set(msks) - set(imgs))
        common = sorted(set(imgs) & set(msks))
        if only_i or only_m:
            missing.append({"split": sp, "img_only": only_i[:5], "mask_only": only_m[:5]})
        for stem in common:
            items.append({
                "region": REGION, "src_split": sp, "gsd_m": GSD_M, "sensor": SENSOR,
                "src_stem": stem,
                "img": os.path.join(img_d, imgs[stem]),
                "mask": os.path.join(msk_d, msks[stem]),
                "out_stem": "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in stem),
            })
        regions_meta.append({"region": REGION, "src_split": sp, "gsd_m": GSD_M,
                             "sensor": SENSOR, "n_img": len(imgs), "n_mask": len(msks),
                             "img_only": len(only_i), "mask_only": len(only_m)})
    return items, regions_meta, missing


# --------------------------------------------------------------------------- pass A
def probe_one(it):
    m = load_mask(it["mask"])
    h, w = m.shape[:2]
    fg = m > 0
    fg_px = int(np.count_nonzero(fg))
    n_cc = (int(cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)[0] - 1)
            if fg_px else 0)
    return {"out_stem": it["out_stem"], "n_cc": n_cc, "fg_src_px": fg_px,
            "src_w": w, "src_h": h}


# --------------------------------------------------------------------------- pass B
def _write_retry(fn, attempts: int = 6, delay: float = 0.25):
    last = None
    for k in range(attempts):
        try:
            return fn()
        except OSError as ex:
            last = ex
            time.sleep(delay * (k + 1))
    raise last


def render_one(job):
    it, split, out_root, min_area, eps_rel, write, poly_res = job
    rgb, meta = read_rgb_u8(it["img"])
    m = load_mask(it["mask"])
    h0, w0 = m.shape[:2]
    if (w0, h0) != (rgb.shape[1], rgb.shape[0]):
        return {"err": "size_mismatch img=%dx%d mask=%dx%d" % (rgb.shape[1], rgb.shape[0], w0, h0),
                "out_stem": it["out_stem"]}

    geo = letterbox((w0, h0))
    sx, sy, nw, nh, dx, dy = geo

    # 参考前景：源掩码 NEAREST 放大后贴进 letterbox 画布（= 模型本该看到的真值）
    ref = np.zeros((S, S), dtype=np.uint8)
    fg = cv2.resize((m > 0).astype(np.uint8) * 255, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ref[dy:dy + nh, dx:dx + nw] = (fg > 0).astype(np.uint8)

    if poly_res == "tile":
        # Extract contours after resizing the mask to retain the tile geometry.
        polys, pstats = mask_to_polygons(ref * 255, min_area=min_area, eps_rel=eps_rel)
        lp = letterbox_polys(polys, (1.0, 1.0, S, S, 0, 0))
    else:
        # 与 LRSTTC/Bijie/CAS/GDCLD/RLZX 逐字同构的规则：源分辨率提多边形再映射
        polys, pstats = mask_to_polygons(m, min_area=min_area, eps_rel=eps_rel)
        lp = letterbox_polys(polys, geo)

    got = rasterize(lp, S)
    inter = int(np.count_nonzero(ref & got))
    union = int(np.count_nonzero(ref | got))
    rt_iou = inter / union if union else 1.0

    line = polygons_to_line(lp, S, cls=0)
    if write:
        d_img = os.path.join(out_root, it["region"], split, "images")
        d_lab = os.path.join(out_root, it["region"], split, "labels")

        def _save_img():
            letterbox_image(Image.fromarray(rgb), geo).save(
                os.path.join(d_img, it["out_stem"] + ".png"), compress_level=PNG_COMPRESS)

        def _save_lab():
            with open(os.path.join(d_lab, it["out_stem"] + ".txt"), "w",
                      encoding="utf-8", newline="\n") as f:
                if line:
                    f.write(line + "\n")

        _write_retry(_save_img)
        _write_retry(_save_lab)

    nodata_px = int(np.count_nonzero(np.all(rgb == 0, axis=2)))

    return {
        "out_stem": it["out_stem"], "region": it["region"], "src_split": it["src_split"],
        "gsd_m": it["gsd_m"], "sensor": it["sensor"], "src_stem": it["src_stem"],
        "src_w": w0, "src_h": h0, "src_shape": meta["src_shape"],
        "n_cc": pstats["contours"] + pstats["dropped"], "n_poly": len(lp),
        "n_dropped": pstats["dropped"],
        "fg_src_px": int(np.count_nonzero(m > 0)),
        "fg_ref_px": int(np.count_nonzero(ref)), "fg_got_px": int(np.count_nonzero(got)),
        "rt_iou": rt_iou, "scale": sx, "fill": (nw * nh) / float(S * S),
        "nodata_px": nodata_px,
    }


# --------------------------------------------------------------------------- selftest
def selftest(n: int, poly_res: str = "tile"):
    items, regions_meta, missing = scan_source()
    print("[scan] 源 split %d 个，配对成功 %d 张" % (len(regions_meta), len(items)))
    for r in regions_meta:
        print("       %-8s img=%5d mask=%5d only_img=%d only_mask=%d" %
              (r["src_split"], r["n_img"], r["n_mask"], r["img_only"], r["mask_only"]))
    rng = random.Random(SEED)
    sample = rng.sample(items, min(n, len(items)))
    print("\n[selftest] 抽样 %d 张（eps_rel=%s min_area=%s poly_res=%s）" %
          (len(sample), EPS_REL_USE, MIN_AREA_USE, poly_res))
    ious, drops, ccs, scales, fills, loss = [], 0, [], [], [], []
    for it in sample:
        r = render_one((it, "train", "", MIN_AREA_USE, EPS_REL_USE, False, poly_res))
        if "err" in r:
            print("  !! %s: %s" % (it["out_stem"], r["err"]))
            continue
        ious.append(r["rt_iou"]); drops += r["n_dropped"]; ccs.append(r["n_cc"])
        scales.append(r["scale"]); fills.append(r["fill"])
        if r["fg_ref_px"]:
            loss.append(1.0 - r["fg_got_px"] / r["fg_ref_px"])
        print("  %-22s src=%dx%d cc=%3d poly=%3d drop=%3d scale=%.3f fg=%6d "
              "nodata=%5d rtIoU=%.4f" %
              (r["src_stem"], r["src_w"], r["src_h"], r["n_cc"], r["n_poly"], r["n_dropped"],
               r["scale"], r["fg_src_px"], r["nodata_px"], r["rt_iou"]))
    print("\n[selftest] 往返 IoU 均值 %.4f 中位 %.4f 最小 %.4f" %
          (statistics.mean(ious), statistics.median(ious), min(ious)))
    print("[selftest] scale 中位 %.4f 画布填充率 中位 %.4f" %
          (statistics.median(scales), statistics.median(fills)))
    print("[selftest] 连通域 中位 %.0f 最大 %d；被 MIN_AREA 丢弃 %d 个" %
          (statistics.median(ccs), max(ccs), drops))
    if loss:
        print("[selftest] 前景面积损失 均值 %.5f 中位 %.5f 最大 %.5f" %
              (statistics.mean(loss), statistics.median(loss), max(loss)))


# --------------------------------------------------------------------------- main
def main(force, workers, split_mode, poly_res, write_outputs=True):
    if write_outputs and os.path.isdir(OUT_ROOT) and os.listdir(OUT_ROOT) and not force:
        sys.exit("[ERR] 输出目录已存在且非空，如要覆盖请加 --force: %s" % OUT_ROOT)

    items, regions_meta, missing = scan_source()
    print("[scan] 配对 %d 张（源 split：%s）" %
          (len(items), ", ".join("%s=%d" % (r["src_split"], r["n_img"]) for r in regions_meta)))
    if missing:
        print("       ⚠️ 未配对文件：%s" % missing)

    t0 = time.time()
    print("[pass A] 统计滑坡连通域数（%d 进程）..." % workers, flush=True)
    probes = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, p in enumerate(ex.map(probe_one, items, chunksize=32), 1):
            probes[p["out_stem"]] = p
            if i % 1000 == 0:
                print("   ... %d/%d (%.0fs)" % (i, len(items), time.time() - t0), flush=True)
    t_a = time.time() - t0
    print("[pass A] 完成 %.0fs" % t_a, flush=True)

    for it in items:
        p = probes[it["out_stem"]]
        it["n_cc"] = p["n_cc"]
        it["fg_src_px"] = p["fg_src_px"]
        it["bucket"] = bucket_of(p["n_cc"])

    if split_mode == "bucket":
        groups = collections.defaultdict(list)
        for it in items:
            groups[it["bucket"]].append(it)
        train, val, target = largest_remainder_split(groups, VAL_RATIO, SEED)
    else:
        # 官方划分：train / val 直接沿用源 split；源 test 归入 val（保持 val 含跨区域样本）
        train = [it for it in items if it["src_split"] == "train"]
        val = [it for it in items if it["src_split"] in ("val", "test")]
        groups = collections.defaultdict(list)
        for it in items:
            groups[it["bucket"]].append(it)
        target = len(val)

    split_of = {it["out_stem"]: "train" for it in train}
    split_of.update({it["out_stem"]: "val" for it in val})
    print("[split] mode=%s train=%d val=%d (target=%d) buckets=%s" %
          (split_mode, len(train), len(val), target,
           {k: len(v) for k, v in sorted(groups.items())}))

    all_regions = sorted({it["region"] for it in items})
    if write_outputs:
        for reg in all_regions:
            for sp in ("train", "val"):
                for kind in ("images", "labels"):
                    os.makedirs(os.path.join(OUT_ROOT, reg, sp, kind), exist_ok=True)
        print("[pass B] 预建 %d×2×2 个目录完成" % len(all_regions), flush=True)

    print("[pass B] %s（%d 进程）..." % ("转换并落盘" if write_outputs else "仅计算", workers),
          flush=True)
    jobs = [(it, split_of[it["out_stem"]], OUT_ROOT, MIN_AREA_USE, EPS_REL_USE, write_outputs,
             poly_res) for it in items]
    agg, errs = [], []
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(render_one, jobs, chunksize=8), 1):
            (errs if "err" in r else agg).append(r)
            if i % 500 == 0:
                el = time.time() - t1
                print("   ... %d/%d (%.0fs, %.1f/s, eta %.1f min)" %
                      (i, len(items), el, i / el, ((len(items) - i) / max(i / el, 1e-9)) / 60),
                      flush=True)
    t_b = time.time() - t1
    print("[pass B] 完成 %.0fs；错误 %d" % (t_b, len(errs)), flush=True)
    if errs:
        print("   !! 错误样例: %s" % errs[:3])

    # ---------------- 复验 ----------------
    bad_size = bad_lab = empty_expected = stale = 0
    if write_outputs:
        for r in agg:
            p = os.path.join(OUT_ROOT, r["region"], split_of[r["out_stem"]], "images",
                             r["out_stem"] + ".png")
            with Image.open(p) as im:
                if im.size != (S, S):
                    bad_size += 1
            lp = os.path.join(OUT_ROOT, r["region"], split_of[r["out_stem"]], "labels",
                              r["out_stem"] + ".txt")
            sz = os.path.getsize(lp)
            if r["n_poly"] and not sz:
                empty_expected += 1
            if not r["n_poly"] and sz:
                bad_lab += 1
        print("[verify] 非 1024² 图像=%d；应为非空却为空=%d；应为空却非空=%d" %
              (bad_size, empty_expected, bad_lab))
        expect = collections.Counter()
        for r in agg:
            expect[(r["region"], split_of[r["out_stem"]], "images")] += 1
            expect[(r["region"], split_of[r["out_stem"]], "labels")] += 1
        for (reg, sp, kind), n in sorted(expect.items()):
            actual = len(os.listdir(os.path.join(OUT_ROOT, reg, sp, kind)))
            if actual != n:
                print("  !! %s/%s/%s: 磁盘 %d != 应有 %d" % (reg, sp, kind, actual, n))
                stale += abs(actual - n)
        print("[verify] 目录文件数校验：%s" % ("PASS" if stale == 0 else "不一致 %d 个" % stale))

    # ---------------- 统计 ----------------
    rt = [r["rt_iou"] for r in agg]
    fg_loss = [1.0 - r["fg_got_px"] / r["fg_ref_px"] for r in agg if r["fg_ref_px"]]
    scales = [r["scale"] for r in agg]
    fills = [r["fill"] for r in agg]
    ccs = [r["n_cc"] for r in agg]

    def q(a, p):
        a = sorted(a)
        return a[min(len(a) - 1, max(0, int(round((p / 100.0) * (len(a) - 1)))))]

    tot_ref = sum(r["fg_ref_px"] for r in agg)
    tot_got = sum(r["fg_got_px"] for r in agg)
    w_loss = 1.0 - tot_got / tot_ref if tot_ref else 0.0
    inter_w = sum(r["fg_ref_px"] * r["rt_iou"] / (1 + r["rt_iou"]) for r in agg)
    union_w = sum(r["fg_ref_px"] / (1 + r["rt_iou"]) for r in agg)
    pix_iou_w = inter_w / union_w if union_w else 1.0

    summary = {
        "format": "compare_dataset_ls4s_v1_1024",
        "task": "instance segmentation (YOLO-seg), single class landslide",
        "source_root": "<src>", "output_root": ".",
        "dataset_citation": "Ghorbanzadeh, O., Xu, Y., Ghamisi, P. et al. Landslide4Sense: "
                            "Reference Benchmark Data and Deep Learning Models for Landslide "
                            "Detection. arXiv:2206.00515 / ISPRS (2022).",
        "input_encoding": {
            "source_format": "one h5 per patch; dataset 'img' (128,128,14) float64 + "
                             "'mask' (128,128) uint8",
            "!! not_rgb": "the 14 layers are a Sentinel-2 MSI stack, NOT an 8-bit RGB image; "
                          "PIL/normal image loaders cannot interpret it directly",
            "band_interpretation": ["B1(443,60m)", "B2(490)", "B3(560)", "B4(665)", "B5(705)",
                                    "B6(740)", "B7(783)", "B8(842)", "B8A(865)", "B9(940,60m)",
                                    "B11(1610)", "B12(2190)", "slope", "DEM"],
            "band_order_evidence": "RGB uses bands B4, B3 and B2.",
            "rgb_used": "ch03,ch02,ch01 = B4(red),B3(green),B2(blue)",
            "dropped_layers": "11 of 14 layers, INCLUDING slope and DEM "
                              "(the benchmark protocol is 3-channel; this is the single biggest "
                              "information loss for this dataset)",
            "value_range": "Scale the floating-point inputs before conversion to 8-bit RGB.",
            "stretch": "Map each channel from its calibrated percentile range to [0,255].",
            "nodata": "See the current input audit for coverage.",
        },
        "ground_truth_source": {
            "used": "annotations/**/mask_*.h5 -> (m>0) -> 255",
            "polarity": "1 = landslide; 0 = background.",
            "handling": "no inversion needed (unlike RLZX-LDD)",
        },
        "class_scheme": {"nc": 1, "names": {0: "landslide"}},
        "label_encoding": "yolo-seg polygon, one line per polygon, normalized xy; one polygon "
                          "per 8-connected component of the landslide mask; empty file = negative",
        "polygon_extraction": {
            "contours": "RETR_EXTERNAL + approxPolyDP", "mode": "per 8-connected component",
            "min_area_px2": MIN_AREA_USE, "approx_eps_rel": EPS_REL_USE,
            "resolution": poly_res,
        },
        "geometry": {
            "mode": "aspect-preserving letterbox, centred, black pad (identical rule to the "
                    "other Compare_dataset builds)",
            "tile_size": S,
            "sources": "LS4S 128x128 -> 1024x1024, i.e. exactly 8x upscale (BICUBIC); the source "
                       "is square so canvas fill = 1.0 and there is no black pad",
            "no_distortion": True, "no_content_loss": True,
        },
        "instance_count_definition": {
            "used": "number of 8-connected components of the landslide mask",
            "why": "the mask is semantic/binary ({0,255}); CC count is the canonical analogue of "
                   "'instances per image' and keeps BUCKETS unchanged",
            "buckets": [b[0] for b in BUCKETS], "seed": SEED,
            "requested_val_ratio": VAL_RATIO, "target_val_images": target,
        },
        "split_provenance": {
            "official_split_available": True,
            "official_split": "The official mode combines validation and test for evaluation.",
            "official_split_is_geographic": True,
            "official_split_evidence": "The official split uses the source subset assignments.",
            "used_mode": split_mode,
            "val_composition_official": {
                "official_train": len([it for it in items if it["src_split"] == "train"]),
                "official_validation": len([it for it in items if it["src_split"] == "val"]),
                "official_test": len([it for it in items if it["src_split"] == "test"]),
                "note": ("the benchmark's Compare_dataset layout only has {train,val} "
                         "(tools/build_coco.py hard-codes choices=['train','val','all']), so the "
                         "official test split is folded into val. Use the manifest's `src_split` "
                         "column to recover the official three-way split at any time "
                         "(src_split in {val,test} => official hold-out)."),
            },
            "switch": "python build_compare_ls4s.py --split-mode official --force",
        },
        "totals": {}, "geometry_stats": {}, "audit": {}, "timing": {},
        "regions": regions_meta, "per_bucket": [], "per_region": [],
    }

    agg_by_stem = {r["out_stem"]: r for r in agg}
    summary["totals"] = {
        "images": len(agg), "train_images": len(train), "val_images": len(val),
        "actual_val_ratio": len(val) / len(agg),
        "instances": sum(r["n_poly"] for r in agg),
        "val_instances": sum(agg_by_stem.get(it["out_stem"], {}).get("n_poly", 0) for it in val),
        "negatives": sum(1 for r in agg if r["n_cc"] == 0),
        "negative_ratio": sum(1 for r in agg if r["n_cc"] == 0) / len(agg),
        "all_foreground_images": sum(1 for r in agg if r["fg_ref_px"] == S * S),
        "dropped_components": sum(r["n_dropped"] for r in agg),
        "images_with_dropped": sum(1 for r in agg if r["n_dropped"]),
        "fg_px_total": tot_ref,
        "fg_ratio_overall": tot_ref / float(len(agg) * S * S) if agg else 0.0,
        "src_negative_ratio": {sp: sum(1 for r in agg if r["src_split"] == sp and r["n_cc"] == 0)
                               / max(1, sum(1 for r in agg if r["src_split"] == sp))
                               for sp in SRC_SPLITS},
    }
    summary["geometry_stats"] = {
        "scale_min": min(scales), "scale_median": statistics.median(scales), "scale_max": max(scales),
        "canvas_fill_median": statistics.median(fills),
        "cc_median": statistics.median(ccs), "cc_max": max(ccs),
        "rt_iou_mean": statistics.mean(rt), "rt_iou_median": statistics.median(rt),
        "rt_iou_p05": q(rt, 5), "rt_iou_min": min(rt),
        "fg_area_loss_mean": statistics.mean(fg_loss) if fg_loss else 0.0,
        "fg_area_loss_median": statistics.median(fg_loss) if fg_loss else 0.0,
        "fg_area_loss_p95": q(fg_loss, 95) if fg_loss else 0.0,
        "fg_area_loss_max": max(fg_loss) if fg_loss else 0.0,
        "fg_area_loss_pixel_weighted": w_loss,
        "fg_mask_iou_pixel_weighted": pix_iou_w,
        "nodata_px_total": sum(r["nodata_px"] for r in agg),
    }
    summary["audit"] = {"write_outputs": write_outputs, "errors": len(errs),
                        "error_samples": errs[:5], "verify_bad_size": bad_size,
                        "verify_expected_empty_but_filled": empty_expected,
                        "verify_expected_filled_but_empty": bad_lab,
                        "verify_count_mismatch": stale, "unpaired": missing}
    summary["timing"] = {"pass_a_s": round(t_a, 1), "pass_b_s": round(t_b, 1),
                         "workers": workers, "total_s": round(time.time() - t0, 1)}

    for label, _lo, _hi in BUCKETS:
        pool = groups.get(label, [])
        if not pool:
            continue
        v = [it for it in pool if split_of[it["out_stem"]] == "val"]
        summary["per_bucket"].append({
            "bucket": label, "images": len(pool), "val_images": len(v),
            "actual_val_ratio": len(v) / len(pool),
            "images_with_landslide": sum(1 for it in pool if it["n_cc"]),
            "val_images_with_landslide": sum(1 for it in v if it["n_cc"])})

    for sp in SRC_SPLITS:
        pool = [r for r in agg if r["src_split"] == sp]
        if not pool:
            continue
        v = [r for r in pool if split_of[r["out_stem"]] == "val"]
        summary["per_region"].append({
            "region": REGION, "src_split": sp, "gsd_m": pool[0]["gsd_m"],
            "sensor": pool[0]["sensor"], "images": len(pool),
            "train_images": len(pool) - len(v), "val_images": len(v),
            "n_instances": sum(r["n_poly"] for r in pool),
            "fg_px": sum(r["fg_ref_px"] for r in pool),
            "fg_ratio": sum(r["fg_ref_px"] for r in pool) / float(len(pool) * S * S),
            "negatives": sum(1 for r in pool if r["n_cc"] == 0),
            "cc_median": statistics.median([r["n_cc"] for r in pool]),
            "rt_iou_mean": statistics.mean([r["rt_iou"] for r in pool]),
            "dropped_components": sum(r["n_dropped"] for r in pool),
            "src_size": "%dx%d" % (pool[0]["src_w"], pool[0]["src_h"]),
        })

    with open(os.path.join(OUT_ROOT, "split_summary_1024.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_ROOT, "split_manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region", "split", "src_split", "out_stem", "src_stem", "gsd_m",
                    "n_cc", "n_poly", "n_dropped", "fg_px", "rt_iou", "nodata_px"])
        for r in sorted(agg, key=lambda x: (x["region"], split_of[x["out_stem"]], x["out_stem"])):
            w.writerow([r["region"], split_of[r["out_stem"]], r["src_split"], r["out_stem"],
                        r["src_stem"], r["gsd_m"], r["n_cc"], r["n_poly"], r["n_dropped"],
                        r["fg_ref_px"], "%.6f" % r["rt_iou"], r["nodata_px"]])

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("regions", "per_region", "per_bucket")},
                     ensure_ascii=False, indent=2))
    print("\nper_bucket:")
    for b in summary["per_bucket"]:
        print("  ", b)
    print("\nper_region（按源 split）:")
    for b in summary["per_region"]:
        print("   %-6s src=%-9s n=%5d tr=%5d va=%4d inst=%6d fg=%.4f cc_med=%.0f "
              "rtIoU=%.4f neg=%4d" %
              (b["src_split"], b["src_size"], b["images"], b["train_images"], b["val_images"],
               b["n_instances"], b["fg_ratio"], b["cc_median"], b["rt_iou_mean"], b["negatives"]))
    print("\n[OK] 写出到 %s  总耗时 %.0fs" % (OUT_ROOT, time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", type=int, default=0, metavar="N")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--eps-rel", type=float, default=EPS_REL)
    ap.add_argument("--min-area", type=float, default=MIN_AREA)
    ap.add_argument("--split-mode", default="bucket", choices=["bucket", "official"])
    ap.add_argument("--poly-resolution", default="tile", choices=["source", "tile"],
                    help="Extract polygons at source or tile resolution.")
    ap.add_argument("--stats-only", action="store_true",
                    help="不写图像/标签，只重算审计统计与清单")
    ap.add_argument("--src", default=SRC_ROOT,
                    help=f"源数据根目录（默认 {SRC_ROOT}）；"
                         f"需先跑 preconvert_ls4s_h5.py 生成 PNG")
    ap.add_argument("--out", default=OUT_ROOT,
                    help=f"输出根目录（默认 {OUT_ROOT}）")
    a = ap.parse_args()
    globals()["SRC_ROOT"] = a.src
    globals()["OUT_ROOT"] = a.out
    MIN_AREA_USE, EPS_REL_USE = a.min_area, a.eps_rel
    print("[cfg] eps_rel=%s min_area=%s split_mode=%s poly_res=%s workers=%d "
          "(基线 eps_rel=%s min_area=%s)" %
          (EPS_REL_USE, MIN_AREA_USE, a.split_mode, a.poly_resolution, a.workers,
           EPS_REL, MIN_AREA))
    if a.selftest:
        selftest(a.selftest, a.poly_resolution)
    else:
        main(a.force, a.workers, a.split_mode, a.poly_resolution,
             write_outputs=not a.stats_only)
