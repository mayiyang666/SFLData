#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert CAS images and foreground masks into paired RGB tiles and polygon labels."""
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

# 与 LRSTTC/Bijie **同一个** letterbox 与多边形实现，以及同一套分桶/切分
from build_compare_lrsttc_1024 import (  # noqa: E402
    S, letterbox, letterbox_image, letterbox_polys, polygons_to_line, rasterize,
)
from build_compare_lrsttc import (  # noqa: E402
    BUCKETS, EPS_REL, MIN_AREA, bucket_of, largest_remainder_split, mask_to_polygons,
)

SRC_ROOT = "data/external/raw/CASDataset"
OUT_ROOT = "data/Compare_dataset_1024/CAS"

SEED, VAL_RATIO = 42, 0.125
PNG_COMPRESS = 3          # optimize=True 会让 PNG 编码慢 3~5 倍；3 已接近体积收益上限

# 多边形编码容差：默认与 LRSTTC/Bijie 完全一致（规则不变）；留 CLI 覆盖口用于保真度对比
MIN_AREA_USE = MIN_AREA
EPS_REL_USE = EPS_REL

# (源子集目录, 源区域目录, 输出区域名, GSD m/px, 传感器/来源)
# GSD 取自数据集自带 README 的 Subdataset 表；Wenchuan 未列入该表，标 None。
CAS_REGIONS = [
    ("Satllite Low Resolution", "Lombok",                       "Lombok",              5.0,  "SAT WorldView2/3"),
    ("Satllite Low Resolution", "Longxi River(SAT)",            "LongxiRiver_SAT",     0.5,  "SAT GF-1"),
    ("Satllite Low Resolution", "Mengdong Township",            "Mengdong",            0.5,  "SAT SuperView-1"),
    ("Satllite Low Resolution", "Moxitaidi (UAV-1m)",           "Moxitaidi_UAV1m",     1.0,  "UAV"),
    ("Satllite Low Resolution", "Tiburon Peninsula（planet）",   "Tiburon_planet",      4.0,  "SAT Planet"),
    ("Satllite Low Resolution", "Wenchuan",                     "Wenchuan",            None, "unknown"),
    ("Satllite Low Resolution", "palu",                         "Palu",                5.0,  "SAT WorldView2/3"),
    ("UAV and High Resolution", "Jiuzhai valley (UAV-0.2m)",    "Jiuzhai_UAV0.2m",     0.2,  "UAV"),
    ("UAV and High Resolution", "Jiuzhai valley (UAV-0.5m)",    "Jiuzhai_UAV0.5m",     0.5,  "UAV"),
    ("UAV and High Resolution", "Longxi River（UAV）",           "LongxiRiver_UAV",     0.5,  "UAV"),
    ("UAV and High Resolution", "Moxi town（UAV-0.2m）",         "MoxiTown_UAV0.2m",    0.2,  "UAV"),
    ("UAV and High Resolution", "Moxitaidi (UAV-0.6m)",         "Moxitaidi_UAV0.6m",   0.6,  "UAV"),
]

SUBSET_TAG = {"Satllite Low Resolution": "satellite_lowres",
              "UAV and High Resolution": "uav_highres"}


def out_stem_of(region_key: str, stem: str) -> str:
    """输出文件名 = 区域键 + 源 stem，净化成 [A-Za-z0-9._-]。"""
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_"
                   for ch in f"{region_key}__{stem}")


# --------------------------------------------------------------------------- scan
def scan_source():
    """遍历 12 个区域，建立清单并查 img/mask 配对。"""
    items, pairs = [], {"unpaired": 0, "img_only": 0, "mask_only": 0}
    regions_meta = []
    for subset, src_region, key, gsd, sensor in CAS_REGIONS:
        d = os.path.join(SRC_ROOT, subset, src_region)
        img_d, msk_d = os.path.join(d, "img"), os.path.join(d, "mask")
        if not (os.path.isdir(img_d) and os.path.isdir(msk_d)):
            sys.exit(f"[ERR] 缺 img/mask 目录: {d}")
        imgs = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(img_d))}
        msks = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(msk_d))}
        only_i, only_m = set(imgs) - set(msks), set(msks) - set(imgs)
        pairs["unpaired"] += len(only_i) + len(only_m)
        pairs["img_only"] += len(only_i)
        pairs["mask_only"] += len(only_m)
        for stem in sorted(set(imgs) & set(msks)):
            items.append({
                "region": key, "subset": subset, "subset_tag": SUBSET_TAG[subset],
                "gsd_m": gsd, "sensor": sensor, "src_stem": stem,
                "img": os.path.join(img_d, imgs[stem]),
                "mask": os.path.join(msk_d, msks[stem]),
                "out_stem": out_stem_of(key, stem),
            })
        regions_meta.append({"region": key, "src_region": src_region, "subset": subset,
                             "subset_tag": SUBSET_TAG[subset], "gsd_m": gsd, "sensor": sensor,
                             "n_img": len(imgs), "n_mask": len(msks),
                             "img_only": len(only_i), "mask_only": len(only_m)})
    return items, pairs, regions_meta


# --------------------------------------------------------------------------- 第一遍：轻量
def probe_one(it):
    """只读掩码，拿到连通域数与前景像素数（用于分桶切分）。"""
    m = np.array(Image.open(it["mask"]))
    h, w = m.shape[:2]
    fg = int(np.count_nonzero(m > 0))
    if fg == 0:
        n_cc = 0
    else:
        n_cc = int(cv2.connectedComponents((m > 0).astype(np.uint8), connectivity=8)[0] - 1)
    return {"out_stem": it["out_stem"], "n_cc": n_cc, "fg_src_px": fg,
            "src_w": w, "src_h": h}


# --------------------------------------------------------------------------- 第二遍：重活
def _write_retry(fn, attempts: int = 6, delay: float = 0.25):
    """Windows 上并发写盘偶发 PermissionError（杀软/索引器瞬时占用）——重试即可。"""
    last = None
    for k in range(attempts):
        try:
            return fn()
        except OSError as ex:
            last = ex
            time.sleep(delay * (k + 1))
    raise last


def render_one(job):
    """letterbox 图像 + 生成多边形标签 + 落盘，返回审计记录。

    目录由父进程预先创建 —— worker 里绝不做 makedirs（并发建目录在 Windows 上会竞态）。
    """
    it, split, out_root, min_area, eps_rel, write = job
    rgb = Image.open(it["img"]).convert("RGB")
    m = np.array(Image.open(it["mask"]))
    h0, w0 = m.shape[:2]
    if (w0, h0) != rgb.size:
        return {"err": f"size_mismatch img={rgb.size} mask=({w0},{h0})", "out_stem": it["out_stem"]}

    geo = letterbox((w0, h0))
    sx, sy, nw, nh, dx, dy = geo

    polys, pstats = mask_to_polygons(m, min_area=min_area, eps_rel=eps_rel)

    # 参考前景：源掩码 NN 放大后贴进 letterbox 画布（= 模型本该看到的真值）
    ref = np.zeros((S, S), dtype=np.uint8)
    fg = cv2.resize((m > 0).astype(np.uint8) * 255, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ref[dy:dy + nh, dx:dx + nw] = (fg > 0).astype(np.uint8)

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
            letterbox_image(rgb, geo).save(os.path.join(d_img, it["out_stem"] + ".png"),
                                           compress_level=PNG_COMPRESS)

        def _save_lab():
            with open(os.path.join(d_lab, it["out_stem"] + ".txt"), "w", encoding="utf-8") as f:
                if line:
                    f.write(line + "\n")

        _write_retry(_save_img)
        _write_retry(_save_lab)

    return {
        "out_stem": it["out_stem"], "region": it["region"], "subset_tag": it["subset_tag"],
        "gsd_m": it["gsd_m"], "sensor": it["sensor"], "src_stem": it["src_stem"],
        "src_w": w0, "src_h": h0,
        "n_cc": pstats["contours"] + pstats["dropped"], "n_poly": len(lp),
        "n_dropped": pstats["dropped"],
        "fg_src_px": int(np.count_nonzero(m > 0)),
        "fg_ref_px": int(np.count_nonzero(ref)), "fg_got_px": int(np.count_nonzero(got)),
        "rt_iou": rt_iou, "scale": sx, "fill": (nw * nh) / float(S * S),
    }


# --------------------------------------------------------------------------- selftest
def selftest(n: int):
    items, pairs, regions_meta = scan_source()
    print(f"[scan] 区域 {len(regions_meta)} 个，配对成功 {len(items)} 张")
    print(f"[scan] 未配对: {pairs['unpaired']} (仅img {pairs['img_only']} / 仅mask {pairs['mask_only']})")
    for r in regions_meta:
        print(f"       {r['region']:<22} img={r['n_img']:>5} mask={r['n_mask']:>5} "
              f"gsd={r['gsd_m']} {r['sensor']}")

    rng = random.Random(SEED)
    sample = rng.sample(items, min(n, len(items)))
    print(f"\n[selftest] 抽样 {len(sample)} 张（eps_rel={EPS_REL_USE} min_area={MIN_AREA_USE}）：")
    ious, drops, ccs, scales, fills, loss = [], 0, [], [], [], []
    for it in sample:
        r = render_one((it, "train", "", MIN_AREA_USE, EPS_REL_USE, False))
        if "err" in r:
            print(f"  !! {it['out_stem']}: {r['err']}")
            continue
        ious.append(r["rt_iou"]); drops += r["n_dropped"]; ccs.append(r["n_cc"])
        scales.append(r["scale"]); fills.append(r["fill"])
        if r["fg_ref_px"]:
            loss.append(1.0 - r["fg_got_px"] / r["fg_ref_px"])
        print(f"  {r['region']:<22} {r['src_stem'][:26]:<28} cc={r['n_cc']:>4} "
              f"poly={r['n_poly']:>4} drop={r['n_dropped']:>3} scale={r['scale']:.3f} "
              f"fill={r['fill']:.3f} fg={r['fg_src_px']:>6} rtIoU={r['rt_iou']:.4f}")
    print(f"\n[selftest] 语义往返 IoU  均值 {statistics.mean(ious):.4f} "
          f"中位 {statistics.median(ious):.4f} 最小 {min(ious):.4f}")
    print(f"[selftest] letterbox scale 中位 {statistics.median(scales):.4f} "
          f"画布填充率 中位 {statistics.median(fills):.4f}")
    print(f"[selftest] 连通域数 中位 {statistics.median(ccs):.0f} 最大 {max(ccs)}；"
          f"被 MIN_AREA 丢弃的小连通域共 {drops} 个")
    if loss:
        print(f"[selftest] 前景面积损失 均值 {statistics.mean(loss):.5f} "
              f"中位 {statistics.median(loss):.5f} 最大 {max(loss):.5f}")


# --------------------------------------------------------------------------- main
def main(force: bool, workers: int, write_outputs: bool = True):
    if write_outputs and os.path.isdir(OUT_ROOT) and os.listdir(OUT_ROOT) and not force:
        sys.exit(f"[ERR] 输出目录已存在且非空，如要覆盖请加 --force: {OUT_ROOT}")
    if not write_outputs:
        print("[mode] --stats-only：不写图像/标签，只重算审计统计与清单")
        if not os.path.isdir(OUT_ROOT):
            sys.exit(f"[ERR] 输出目录不存在，无法只统计: {OUT_ROOT}")

    items, pairs, regions_meta = scan_source()
    print(f"[scan] 配对 {len(items)} 张；未配对 {pairs['unpaired']} "
          f"(仅img {pairs['img_only']} / 仅mask {pairs['mask_only']})")
    if pairs["unpaired"]:
        print("       ⚠️ 存在未配对文件，已跳过；详见 summary 的 pairs 字段")

    t0 = time.time()
    # ---- 第一遍：连通域数（并行，轻量）----
    print(f"[pass A] 统计连通域数（{workers} 进程）...", flush=True)
    probes = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, p in enumerate(ex.map(probe_one, items, chunksize=64), 1):
            probes[p["out_stem"]] = p
            if i % 4000 == 0:
                print(f"   ... {i}/{len(items)}  ({time.time()-t0:.0f}s)", flush=True)
    t_a = time.time() - t0
    print(f"[pass A] 完成，{t_a:.0f}s", flush=True)

    for it in items:
        p = probes[it["out_stem"]]
        it["n_cc"] = p["n_cc"]
        it["fg_src_px"] = p["fg_src_px"]
        it["bucket"] = bucket_of(p["n_cc"])

    groups = collections.defaultdict(list)
    for it in items:
        groups[it["bucket"]].append(it)
    train, val, target = largest_remainder_split(groups, VAL_RATIO, SEED)
    split_of = {it["out_stem"]: "train" for it in train}
    split_of.update({it["out_stem"]: "val" for it in val})
    print(f"[split] train={len(train)} val={len(val)} (target={target}) "
          f"buckets={ {k: len(v) for k, v in sorted(groups.items())} }")

    # ---- 第二遍：letterbox + 多边形 + 落盘（并行，重活）----
    # 目录一律在父进程预建：worker 并发 makedirs 在 Windows 上会竞态报 PermissionError
    all_regions = sorted({it["region"] for it in items})
    if write_outputs:
        for reg in all_regions:
            for sp in ("train", "val"):
                for kind in ("images", "labels"):
                    os.makedirs(os.path.join(OUT_ROOT, reg, sp, kind), exist_ok=True)
        print(f"[pass B] 预建 {len(all_regions)}×2×2 个目录完成", flush=True)

    print(f"[pass B] {'转换并落盘' if write_outputs else '仅计算（不落盘）'}（{workers} 进程）...", flush=True)
    jobs = [(it, split_of[it["out_stem"]], OUT_ROOT, MIN_AREA_USE, EPS_REL_USE, write_outputs)
            for it in items]
    agg, errs = [], []
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(render_one, jobs, chunksize=16), 1):
            if "err" in r:
                errs.append(r)
            else:
                agg.append(r)
            if i % 2000 == 0:
                el = time.time() - t1
                print(f"   ... {i}/{len(items)}  ({el:.0f}s, {i/el:.1f}/s, "
                      f"eta {((len(items)-i)/max(i/el,1e-9))/60:.1f} min)", flush=True)
    t_b = time.time() - t1
    print(f"[pass B] 完成，{t_b:.0f}s；错误 {len(errs)}", flush=True)

    # ---- 复验（仅落盘模式）----
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
        print(f"[verify] 非 1024² 图像={bad_size}；应为非空却为空={empty_expected}；"
              f"应为空却非空={bad_lab}")

        # 落盘文件数 vs 应有数（能抓出残留/漏写）
        expect = collections.Counter()
        for r in agg:
            expect[(r["region"], split_of[r["out_stem"]], "images")] += 1
            expect[(r["region"], split_of[r["out_stem"]], "labels")] += 1
        for (reg, sp, kind), n in sorted(expect.items()):
            actual = len(os.listdir(os.path.join(OUT_ROOT, reg, sp, kind)))
            if actual != n:
                print(f"  !! {reg}/{sp}/{kind}: 磁盘 {actual} != 应有 {n}")
                stale += abs(actual - n)
        print(f"[verify] 目录文件数校验：{'PASS' if stale == 0 else f'不一致 {stale} 个'}")

    # ---- 统计 ----
    rt = [r["rt_iou"] for r in agg]
    fg_loss = [1.0 - r["fg_got_px"] / r["fg_ref_px"] for r in agg if r["fg_ref_px"]]
    scales = [r["scale"] for r in agg]
    fills = [r["fill"] for r in agg]
    ccs = [r["n_cc"] for r in agg]

    def q(a, p):
        a = sorted(a)
        return a[min(len(a) - 1, max(0, int(round((p / 100.0) * (len(a) - 1)))))]

    # 面积加权的全局前景偏差（更贴近 mIoU 的像素口径）
    tot_ref = sum(r["fg_ref_px"] for r in agg)
    tot_got = sum(r["fg_got_px"] for r in agg)
    w_loss = 1.0 - tot_got / tot_ref if tot_ref else 0.0
    # 全局像素口径的「原始掩码 vs 多边形标签」IoU（近似）：用面积偏差与往返 IoU 合成
    inter_w = sum(r["fg_ref_px"] * r["rt_iou"] / (1 + r["rt_iou"]) for r in agg)
    union_w = sum(r["fg_ref_px"] / (1 + r["rt_iou"]) for r in agg)
    pix_iou_w = inter_w / union_w if union_w else 1.0

    # val/train 里装的是 scan 阶段的 item，逐图审计字段只在 agg（render 结果）里
    agg_by_stem = {r["out_stem"]: r for r in agg}

    summary = {
        "format": "compare_dataset_cas_v1_1024_semantic",
        "task": "semantic segmentation (DeepLabV3+)",
        "source_root": "<src>", "output_root": ".",
        "geometry": {
            "mode": "aspect-preserving letterbox, centred, black pad (identical rule to LRSTTC/Bijie)",
            "tile_size": S,
            "source_size": "ALL images are exactly 512x512 -> scale exactly 2.0, dx=dy=0, "
                           "canvas fill = 1.0, ZERO black border",
            "resample": "BICUBIC (upscaling)",
            "no_distortion": True, "no_content_loss": True,
        },
        "class_scheme": {"nc": 1, "names": {0: "landslide"},
                         "note": "semantic: 0=background, 1=landslide; the YOLO label class "
                                 "index is 0 because COCO category_id 1 maps to training class 0"},
        "label_encoding": "yolo-seg polygon, one line per polygon, normalized xy; "
                          "one polygon per connected component of the landslide mask",
        "ground_truth_source": {
            "used": "mask/*.tif (L mode, values {0,1}, 1 = landslide)",
            "discarded": "label/*.tif (RGB {0,255}) is the EXACT INVERSE of mask "
                         "(0 = landslide, 255 = background), and its filenames do not match "
                         "img stems in the Longxi River（UAV）region -> unusable",
            "evidence": "The source label raster and foreground mask use opposite polarities.",
        },
        "polygon_extraction": {
            "contours": "RETR_EXTERNAL + approxPolyDP", "mode": "per 8-connected component",
            "min_area_px2": MIN_AREA_USE, "approx_eps_rel": EPS_REL_USE,
            "rule_parity": "identical to LRSTTC/Bijie",
            "fidelity_note": "See the computed round-trip statistics for polygon fidelity.",
        },
        "instance_count_definition": {
            "used": "number of 8-connected components of the landslide mask",
            "why": "CAS is semantic (no instance id); CC count is the canonical analogue of "
                   "'instances per image' and keeps BUCKETS unchanged",
            "buckets": [b[0] for b in BUCKETS], "seed": SEED,
            "requested_val_ratio": VAL_RATIO, "target_val_images": target,
        },
        "totals": {
            "images": len(agg), "train_images": len(train), "val_images": len(val),
            "actual_val_ratio": len(val) / len(agg),
            "instances": sum(r["n_poly"] for r in agg),
            "val_instances": sum(agg_by_stem.get(it["out_stem"], {}).get("n_poly", 0) for it in val),
            "negatives": sum(1 for r in agg if r["n_cc"] == 0),
            "all_foreground_images": sum(1 for r in agg if r["fg_ref_px"] == S * S),
            "dropped_components": sum(r["n_dropped"] for r in agg),
            "images_with_dropped": sum(1 for r in agg if r["n_dropped"]),
            "fg_px_total": tot_ref,
        },
        "geometry_stats": {
            "scale_min": min(scales), "scale_median": statistics.median(scales),
            "scale_max": max(scales), "canvas_fill_median": statistics.median(fills),
            "cc_median": statistics.median(ccs), "cc_max": max(ccs),
            "semantic_rt_iou_mean": statistics.mean(rt),
            "semantic_rt_iou_median": statistics.median(rt),
            "semantic_rt_iou_p05": q(rt, 5), "semantic_rt_iou_min": min(rt),
            "fg_area_loss_mean": statistics.mean(fg_loss) if fg_loss else 0.0,
            "fg_area_loss_median": statistics.median(fg_loss) if fg_loss else 0.0,
            "fg_area_loss_p95": q(fg_loss, 95) if fg_loss else 0.0,
            "fg_area_loss_max": max(fg_loss) if fg_loss else 0.0,
            "fg_area_loss_pixel_weighted": w_loss,
            "fg_mask_iou_pixel_weighted": pix_iou_w,
        },
        "pairs": pairs,
        "audit": {"write_outputs": write_outputs, "errors": len(errs),
                  "verify_bad_size": bad_size,
                  "verify_expected_empty_but_filled": empty_expected,
                  "verify_expected_filled_but_empty": bad_lab,
                  "verify_count_mismatch": stale},
        "timing": {"pass_a_s": round(t_a, 1), "pass_b_s": round(t_b, 1), "workers": workers,
                   "total_s": round(time.time() - t0, 1)},
        "regions": regions_meta, "per_bucket": [], "per_region": [],
    }

    for label, _lo, _hi in BUCKETS:
        pool = groups.get(label, [])
        if not pool:
            continue
        v = [it for it in pool if split_of[it["out_stem"]] == "val"]
        summary["per_bucket"].append({
            "bucket": label, "images": len(pool), "val_images": len(v),
            "actual_val_ratio": len(v) / len(pool),
            "instances": sum(1 for it in pool if it["n_cc"]),
            "val_instances": sum(1 for it in v if it["n_cc"])})

    for reg in sorted({r["region"] for r in agg}):
        pool = [r for r in agg if r["region"] == reg]
        v = [r for r in pool if split_of[r["out_stem"]] == "val"]
        summary["per_region"].append({
            "region": reg, "subset": pool[0]["subset_tag"], "gsd_m": pool[0]["gsd_m"],
            "sensor": pool[0]["sensor"],
            "images": len(pool), "train_images": len(pool) - len(v), "val_images": len(v),
            "n_instances": sum(r["n_poly"] for r in pool),
            "val_instances": sum(r["n_poly"] for r in v),
            "fg_px": sum(r["fg_ref_px"] for r in pool),
            "fg_ratio": sum(r["fg_ref_px"] for r in pool) / float(len(pool) * S * S),
            "cc_median": statistics.median([r["n_cc"] for r in pool]),
            "rt_iou_mean": statistics.mean([r["rt_iou"] for r in pool]),
            "dropped_components": sum(r["n_dropped"] for r in pool),
            "images_with_dropped": sum(1 for r in pool if r["n_dropped"]),
        })

    with open(os.path.join(OUT_ROOT, "split_summary_1024.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_ROOT, "split_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region", "split", "out_stem", "src_stem", "gsd_m", "sensor",
                    "n_cc", "n_poly", "n_dropped", "fg_px", "rt_iou"])
        for r in sorted(agg, key=lambda x: (x["region"], split_of[x["out_stem"]], x["out_stem"])):
            w.writerow([r["region"], split_of[r["out_stem"]], r["out_stem"], r["src_stem"],
                        r["gsd_m"], r["sensor"], r["n_cc"], r["n_poly"], r["n_dropped"],
                        r["fg_ref_px"], f"{r['rt_iou']:.6f}"])

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("regions", "per_region", "per_bucket")},
                     ensure_ascii=False, indent=2))
    print("\nper_bucket:")
    for b in summary["per_bucket"]:
        print("  ", b)
    print("\nper_region:")
    for b in summary["per_region"]:
        print(f"   {b['region']:<22} n={b['images']:>5} tr={b['train_images']:>5} "
              f"va={b['val_images']:>4} inst={b['n_instances']:>5} "
              f"gsd={str(b['gsd_m']):<5} fg={b['fg_ratio']:.4f} cc_med={b['cc_median']:.0f} "
              f"rtIoU={b['rt_iou_mean']:.4f} drop={b['images_with_dropped']}")
    print(f"\n[OK] 写出到 {OUT_ROOT}  总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", type=int, default=0, metavar="N")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--eps-rel", type=float, default=EPS_REL,
                    help=f"approxPolyDP 相对周长容差（默认 {EPS_REL}，与 LRSTTC/Bijie 一致）")
    ap.add_argument("--min-area", type=float, default=MIN_AREA,
                    help=f"连通域最小面积 px^2（默认 {MIN_AREA}，与 LRSTTC/Bijie 一致）")
    ap.add_argument("--stats-only", action="store_true",
                    help="不写图像/标签，只重算审计统计与清单（用于补摘要，约 5 min 而非 13 min）")
    ap.add_argument("--src", default=SRC_ROOT,
                    help=f"源数据根目录（默认 {SRC_ROOT}）")
    ap.add_argument("--out", default=OUT_ROOT,
                    help=f"输出根目录（默认 {OUT_ROOT}）")
    a = ap.parse_args()
    globals()["SRC_ROOT"] = a.src
    globals()["OUT_ROOT"] = a.out
    MIN_AREA_USE, EPS_REL_USE = a.min_area, a.eps_rel
    print(f"[cfg] eps_rel={EPS_REL_USE} min_area={MIN_AREA_USE} "
          f"(规则基线 eps_rel={EPS_REL} min_area={MIN_AREA}) workers={a.workers}")
    if a.selftest:
        selftest(a.selftest)
    else:
        main(a.force, a.workers, write_outputs=not a.stats_only)
