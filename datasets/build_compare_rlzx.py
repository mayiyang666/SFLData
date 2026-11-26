#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare the satellite and UAV subsets of RLZX-LDD (Fu et al., doi:10.1038/s41597-025-05670-w)."""
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
import tifffile
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 与 LRSTTC/Bijie/CAS **同一套** letterbox、多边形、分桶与切分实现
from build_compare_lrsttc_1024 import (  # noqa: E402
    S, letterbox, letterbox_image, letterbox_polys, polygons_to_line, rasterize,
)
from build_compare_lrsttc import (  # noqa: E402
    BUCKETS, EPS_REL, MIN_AREA, bucket_of, largest_remainder_split, mask_to_polygons,
)

SRC_ROOT = "data/external/raw/RLZX-LDD"
OUT_ROOT = "data/Compare_dataset_1024/RLZX"

SEED, VAL_RATIO = 42, 0.125
PNG_COMPRESS = 3

MIN_AREA_USE = MIN_AREA
EPS_REL_USE = EPS_REL

# (源子集目录, 文件名前缀过滤(None=全部), 输出区域名, GSD m/px, 传感器)
# GSD 取自 Scientific Data 论文 Table 2（Fu et al. 2025, doi:10.1038/s41597-025-05670-w）：
#   SAT-LDD 2 m（Gaofen 全色 2 m 与多光谱 8 m 经 NNDiffuse 融合后的 2 m 多光谱），
#   UAV-LDD 0.3 m（四张无人机镶嵌图，RGB）。
RLZX_REGIONS = [
    ("SAT-LDD", None,    "SAT",  2.0, "Gaofen (SAT) 2 m, RGB+NIR"),
    ("UAV-LDD", "UAV1",  "UAV1", 0.3, "UAV mosaic 1, 0.3 m"),
    ("UAV-LDD", "UAV2",  "UAV2", 0.3, "UAV mosaic 2, 0.3 m"),
    ("UAV-LDD", "UAV3",  "UAV3", 0.3, "UAV mosaic 3, 0.3 m"),
    ("UAV-LDD", "UAV4",  "UAV4", 0.3, "UAV mosaic 4, 0.3 m"),
]


# --------------------------------------------------------------------------- IO helpers
def read_rgb_u8(path: str) -> tuple[np.ndarray, dict]:
    """float32 多波段 TIFF -> uint8 RGB（取前 3 通道）。返回 (rgb, meta)。"""
    a = tifffile.imread(path)
    if a.ndim == 3:
        rgb = a[:, :, :3]
    elif a.ndim == 2:
        rgb = np.repeat(a[:, :, None], 3, axis=2)
    else:
        raise RuntimeError(f"意外维度 {a.shape}: {path}")
    amax = float(a.max())
    if rgb.dtype != np.uint8:
        if amax > 1.0:                                   # 已是 0-255 量级
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        else:                                            # [0,1] -> [0,255]
            rgb = np.clip(rgb.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)
    meta = {"src_shape": list(a.shape), "src_dtype": str(a.dtype), "src_max": amax}
    return rgb, meta


def load_mask_inverted(path: str) -> np.ndarray:
    """读掩膜并反相：源 `0=滑坡 / 255=背景` -> 返回 `255=滑坡 / 0=背景`。

    反相后与 `mask_to_polygons` 的 `vals[vals>0]` 前景语义一致（见模块 docstring 坑 1）。
    """
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"掩膜读取失败: {path}")
    if m.ndim == 3:
        m = m[:, :, 0]
    return np.where(m == 0, np.uint8(255), np.uint8(0))


# --------------------------------------------------------------------------- scan
def scan_source():
    items, regions_meta = [], []
    missing = []
    for subset, prefix, key, gsd, sensor in RLZX_REGIONS:
        img_d = os.path.join(SRC_ROOT, subset, "images")
        lab_d = os.path.join(SRC_ROOT, subset, "labels")
        if not (os.path.isdir(img_d) and os.path.isdir(lab_d)):
            sys.exit(f"[ERR] 缺 images/labels: {os.path.join(SRC_ROOT, subset)}")
        imgs = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(img_d))}
        labs = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(lab_d))}
        if prefix:
            imgs = {k: v for k, v in imgs.items() if k.startswith(prefix + "_")}
            labs = {k: v for k, v in labs.items() if k.startswith(prefix + "_")}
        only_i, only_l = sorted(set(imgs) - set(labs)), sorted(set(labs) - set(imgs))
        common = sorted(set(imgs) & set(labs))
        if only_i or only_l:
            missing.append({"region": key, "img_only": only_i[:5], "lab_only": only_l[:5]})
        for stem in common:
            items.append({
                "region": key, "subset": subset, "gsd_m": gsd, "sensor": sensor,
                "src_stem": stem,
                "img": os.path.join(img_d, imgs[stem]),
                "mask": os.path.join(lab_d, labs[stem]),
                "out_stem": "".join(ch if (ch.isalnum() or ch in "._-") else "_"
                                    for ch in f"{key}__{stem}"),
            })
        regions_meta.append({"region": key, "subset": subset, "gsd_m": gsd, "sensor": sensor,
                             "n_img": len(imgs), "n_mask": len(labs),
                             "img_only": len(only_i), "lab_only": len(only_l)})
    return items, regions_meta, missing


# --------------------------------------------------------------------------- split modes
OFFICIAL_TRAIN_RATIO = 0.7
OFFICIAL_SEED = 2024


def official_split(items, train_ratio: float = OFFICIAL_TRAIN_RATIO,
                   seed: int = OFFICIAL_SEED):
    """复现官方 `tools.ipynb::split_dataset` 的划分：**按子集各自** sorted -> shuffle -> 7:3。

    官方原文：
        image_files = sorted(os.listdir(images_dir))
        random.seed(random_seed)          # random_seed = 2024，每次调用都重设
        random.shuffle(image_files)
        train_size = int(len(image_files) * train_ratio)   # train_ratio = 0.7
        train_files = image_files[:train_size] ; test_files = image_files[train_size:]

    并且 `split_dataset(SAT)` 与 `split_dataset(UAV)` **分别调用**（两次都从同一种子重设），
    所以这里也按 subset 独立洗牌。

    ⚠️ 官方是 **patch 级随机**（非空间/事件留出）⇒ 同一景的相邻瓦片会横跨 train/test，
    空间自相关泄漏不可量化。论文引用时需说明"绝对指标有向上偏置，跨数据集比较应看相对排序"。
    """
    train, val = [], []
    for subset in ("SAT-LDD", "UAV-LDD"):
        pool = sorted([it for it in items if it["subset"] == subset],
                      key=lambda it: it["src_stem"])
        names = [it["src_stem"] for it in pool]
        random.Random(seed).shuffle(names)     # 等价 random.seed(seed); random.shuffle(...)
        n_train = int(len(names) * train_ratio)
        keep = set(names[:n_train])
        print(f"       [{subset}] n={len(names)} -> int(n*{train_ratio})={n_train} train / "
              f"{len(names)-n_train} test", flush=True)
        for it in pool:
            (train if it["src_stem"] in keep else val).append(it)
    return train, val


# --------------------------------------------------------------------------- pass A
def probe_one(it):
    """只读掩膜：连通域数 + 前景像素（用于分桶切分）。"""
    m_inv = load_mask_inverted(it["mask"])
    h, w = m_inv.shape[:2]
    fg = m_inv > 0
    fg_px = int(np.count_nonzero(fg))
    n_cc = (int(cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)[0] - 1)
            if fg_px else 0)
    return {"out_stem": it["out_stem"], "n_cc": n_cc, "fg_src_px": fg_px,
            "src_w": w, "src_h": h}


# --------------------------------------------------------------------------- pass B
def _write_retry(fn, attempts: int = 6, delay: float = 0.25):
    """Windows 并发写盘偶发 PermissionError（杀软/索引器瞬时占用）—— 重试即可。"""
    last = None
    for k in range(attempts):
        try:
            return fn()
        except OSError as ex:
            last = ex
            time.sleep(delay * (k + 1))
    raise last


def render_one(job):
    """letterbox + 多边形标签 + 落盘。目录由父进程预建（worker 不做 makedirs）。"""
    it, split, out_root, min_area, eps_rel, write = job
    rgb, meta = read_rgb_u8(it["img"])
    m_inv = load_mask_inverted(it["mask"])
    h0, w0 = m_inv.shape[:2]
    if (w0, h0) != (rgb.shape[1], rgb.shape[0]):
        return {"err": f"size_mismatch img={rgb.shape[1]}x{rgb.shape[0]} "
                       f"mask={w0}x{h0}", "out_stem": it["out_stem"]}

    geo = letterbox((w0, h0))
    sx, sy, nw, nh, dx, dy = geo

    # 反相后掩膜 -> 多边形（规则与 LRSTTC/Bijie/CAS 逐字一致）
    polys, pstats = mask_to_polygons(m_inv, min_area=min_area, eps_rel=eps_rel)

    # 参考前景：源掩码 NN 放大后贴进 letterbox 画布（= 模型本该看到的真值）
    ref = np.zeros((S, S), dtype=np.uint8)
    fg = cv2.resize((m_inv > 0).astype(np.uint8) * 255, (nw, nh),
                    interpolation=cv2.INTER_NEAREST)
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
            letterbox_image(Image.fromarray(rgb), geo).save(
                os.path.join(d_img, it["out_stem"] + ".png"), compress_level=PNG_COMPRESS)

        def _save_lab():
            with open(os.path.join(d_lab, it["out_stem"] + ".txt"), "w",
                      encoding="utf-8", newline="\n") as f:
                if line:
                    f.write(line + "\n")

        _write_retry(_save_img)
        _write_retry(_save_lab)

    # UAV 镶嵌边界 nodata（源图三通道全 0），与 letterbox 黑边无关
    nodata_px = int(np.count_nonzero(np.all(rgb == 0, axis=2)))

    return {
        "out_stem": it["out_stem"], "region": it["region"], "subset": it["subset"],
        "gsd_m": it["gsd_m"], "sensor": it["sensor"], "src_stem": it["src_stem"],
        "src_w": w0, "src_h": h0, "src_shape": meta["src_shape"],
        "n_cc": pstats["contours"] + pstats["dropped"], "n_poly": len(lp),
        "n_dropped": pstats["dropped"],
        "fg_src_px": int(np.count_nonzero(m_inv > 0)),
        "fg_ref_px": int(np.count_nonzero(ref)), "fg_got_px": int(np.count_nonzero(got)),
        "rt_iou": rt_iou, "scale": sx, "fill": (nw * nh) / float(S * S),
        "nodata_px": nodata_px,
    }


# --------------------------------------------------------------------------- selftest
def selftest(n: int):
    items, regions_meta, missing = scan_source()
    print(f"[scan] 区域 {len(regions_meta)} 个，配对成功 {len(items)} 张")
    for r in regions_meta:
        print(f"       {r['region']:<8} img={r['n_img']:>5} mask={r['n_mask']:>5} "
              f"only_img={r['img_only']} only_mask={r['lab_only']} {r['sensor']}")

    rng = random.Random(SEED)
    per = max(1, n // len(regions_meta))
    sample = []
    for reg in [r["region"] for r in regions_meta]:
        pool = [it for it in items if it["region"] == reg]
        sample += rng.sample(pool, min(per, len(pool)))

    print(f"\n[selftest] 抽样 {len(sample)} 张（eps_rel={EPS_REL_USE} min_area={MIN_AREA_USE}）")
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
        print(f"  {r['region']:<6} {r['src_stem'][:20]:<22} src={r['src_w']}x{r['src_h']} "
              f"cc={r['n_cc']:>3} poly={r['n_poly']:>3} drop={r['n_dropped']:>2} "
              f"scale={r['scale']:.3f} fg={r['fg_src_px']:>7} nodata={r['nodata_px']:>6} "
              f"rtIoU={r['rt_iou']:.4f}")
    print(f"\n[selftest] 往返 IoU   均值 {statistics.mean(ious):.4f} "
          f"中位 {statistics.median(ious):.4f} 最小 {min(ious):.4f}")
    print(f"[selftest] letterbox scale 中位 {statistics.median(scales):.4f} "
          f"画布填充率 中位 {statistics.median(fills):.4f}")
    print(f"[selftest] 连通域 中位 {statistics.median(ccs):.0f} 最大 {max(ccs)}；"
          f"被 MIN_AREA 丢弃 {drops} 个")
    if loss:
        print(f"[selftest] 前景面积损失 均值 {statistics.mean(loss):.5f} "
              f"中位 {statistics.median(loss):.5f} 最大 {max(loss):.5f}")


# --------------------------------------------------------------------------- main
def main(force: bool, workers: int, write_outputs: bool = True, split_mode: str = "bucket"):
    if write_outputs and os.path.isdir(OUT_ROOT) and os.listdir(OUT_ROOT) and not force:
        sys.exit(f"[ERR] 输出目录已存在且非空，如要覆盖请加 --force: {OUT_ROOT}")

    items, regions_meta, missing = scan_source()
    print(f"[scan] 配对 {len(items)} 张；区域 {len(regions_meta)} 个")
    for r in regions_meta:
        print(f"       {r['region']:<8} img={r['n_img']:>5} mask={r['n_mask']:>5} "
              f"| {r['sensor']}")
    if missing:
        print(f"       ⚠️ 存在未配对文件：{missing}")

    t0 = time.time()
    print(f"[pass A] 统计滑坡连通域数（{workers} 进程）...", flush=True)
    probes = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, p in enumerate(ex.map(probe_one, items, chunksize=32), 1):
            probes[p["out_stem"]] = p
            if i % 1000 == 0:
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
    if split_mode == "official":
        print(f"[split] 官方划分（tools.ipynb: sorted -> shuffle(seed={OFFICIAL_SEED}) -> "
              f"{OFFICIAL_TRAIN_RATIO:.0%}/{1 - OFFICIAL_TRAIN_RATIO:.0%}，SAT 与 UAV 各自独立）")
        train, val = official_split(items)
        target = len(val)
    else:
        train, val, target = largest_remainder_split(groups, VAL_RATIO, SEED)
        print(f"[split] 分桶 12.5%（seed={SEED}，target={target}）")
    split_of = {it["out_stem"]: "train" for it in train}
    split_of.update({it["out_stem"]: "val" for it in val})
    print(f"[split] train={len(train)} val={len(val)} "
          f"actual_val_ratio={len(val) / len(items):.4f} "
          f"buckets={ {k: len(v) for k, v in sorted(groups.items())} }")

    all_regions = sorted({it["region"] for it in items})
    if write_outputs:
        for reg in all_regions:
            for sp in ("train", "val"):
                for kind in ("images", "labels"):
                    os.makedirs(os.path.join(OUT_ROOT, reg, sp, kind), exist_ok=True)
        print(f"[pass B] 预建 {len(all_regions)}×2×2 个目录完成", flush=True)

    print(f"[pass B] {'转换并落盘' if write_outputs else '仅计算'}（{workers} 进程）...",
          flush=True)
    jobs = [(it, split_of[it["out_stem"]], OUT_ROOT, MIN_AREA_USE, EPS_REL_USE, write_outputs)
            for it in items]
    agg, errs = [], []
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(render_one, jobs, chunksize=8), 1):
            (errs if "err" in r else agg).append(r)
            if i % 500 == 0:
                el = time.time() - t1
                print(f"   ... {i}/{len(items)}  ({el:.0f}s, {i/el:.1f}/s, "
                      f"eta {((len(items)-i)/max(i/el,1e-9))/60:.1f} min)", flush=True)
    t_b = time.time() - t1
    print(f"[pass B] 完成，{t_b:.0f}s；错误 {len(errs)}", flush=True)
    if errs:
        print(f"   !! 错误样例: {errs[:3]}")

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
        print(f"[verify] 非 1024² 图像={bad_size}；应为非空却为空={empty_expected}；"
              f"应为空却非空={bad_lab}")

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

    agg_by_stem = {r["out_stem"]: r for r in agg}

    summary = {
        "format": "compare_dataset_rlzx_v1_1024",
        "task": "instance segmentation (YOLO-seg), single class landslide",
        "source_root": "<src>", "output_root": ".",
        "dataset_citation": "Fu, Z., Wang, F., Ma, H. et al. Records of shallow landslides "
                            "triggered by extreme rainfall in July 2024 in Zixing, China. "
                            "Sci Data 12, 1364 (2025). doi:10.1038/s41597-025-05670-w",
        "ground_truth_source": {
            "used": "labels/*.png (L mode, values {0,255})",
            "!! polarity": "0 = LANDSLIDE (foreground), 255 = BACKGROUND  -> the mask is "
                           "INVERTED relative to the usual convention",
            "handling": "the mask is inverted at load time (0->255, 255->0) so that "
                        "mask_to_polygons()' `vals[vals>0]` foreground semantics hold",
        },
        "geometry": {
            "mode": "aspect-preserving letterbox, centred, black pad (identical rule to "
                    "LRSTTC/Bijie/CAS)",
            "tile_size": S,
            "sources": "SAT-LDD 256x256 -> 1024x1024 (4x upscale, BICUBIC); "
                       "UAV-LDD 1024x1024 -> identity (scale=1.0, dx=dy=0)",
            "resample": "LANCZOS when downscaling, BICUBIC when upscaling",
            "no_distortion": True, "no_content_loss": True,
        },
        "input_encoding": {
            "source_format": "float32 multi-band TIFF (uncompressed), PIL cannot read it",
            "SAT-LDD": "256x256x4, band order (R,G,B,NIR) AS STORED; NIR (band 4) dropped, "
                       "first 3 bands used as RGB with NO channel swap",
            "UAV-LDD": "1024x1024x3, band order (R,G,B) as stored",
            "band_order_evidence": "rendering bands (0,1,2) yields natural true colour "
                                   "(green vegetation, brown bare soil); swapping to (2,1,0) "
                                   "turns the whole scene cyan/blue. The upstream repo's "
                                   "dataloader swaps gaofen data, but THIS released copy is "
                                   "already in RGB order.",
            "value_range": "[0,1] float -> x255 -> uint8",
            "nodata": "UAV source tiles carry black nodata wedges at the mosaic boundary; "
                      "they lie in the background (255) of the mask and are carried through "
                      "unchanged as non-landslide area",
        },
        "class_scheme": {"nc": 1, "names": {0: "landslide"}},
        "label_encoding": "yolo-seg polygon, one line per polygon, normalized xy; "
                          "one polygon per 8-connected component of the landslide mask; "
                          "empty file = negative",
        "polygon_extraction": {
            "contours": "RETR_EXTERNAL + approxPolyDP",
            "mode": "per 8-connected component",
            "min_area_px2": MIN_AREA_USE, "approx_eps_rel": EPS_REL_USE,
            "rule_parity": "identical to LRSTTC/Bijie/CAS",
        },
        "instance_count_definition": {
            "used": "number of 8-connected components of the landslide (inverted) mask",
            "why": "RLZX-LDD labels are semantic (the only non-zero value is 255, no instance "
                   "ids); CC count is the canonical analogue of 'instances per image' and "
                   "keeps BUCKETS unchanged",
            "buckets": [b[0] for b in BUCKETS], "seed": SEED,
            "requested_val_ratio": VAL_RATIO, "target_val_images": target,
        },
        "split_scheme": {
            "mode": split_mode,
            "official_source": "tools.ipynb :: split_dataset() @ github.com/klaus2023/"
                               "RLZX-landslide-inventory-and-landslide-detection-datasets",
            "official_rule": "sorted(os.listdir(images)) -> random.seed(2024) -> shuffle -> "
                             "int(n*0.7) train / rest test; applied SEPARATELY to SAT-LDD and "
                             "UAV-LDD (each call re-seeds to 2024)",
            "official_train_ratio": OFFICIAL_TRAIN_RATIO, "official_seed": OFFICIAL_SEED,
            "deterministic_reproducible": True,
            "our_bucket_rule": "per-bucket (landslide CC count) largest-remainder 12.5% val, "
                               "seed=42 -- kept available for cross-dataset comparability",
        },
        "split_provenance": {
            "official_split_available": True,
            "note": "the official split is a deterministic function of the sorted filename list, "
                    "so it is fully reproducible without any extra download. THIS local copy only "
                    "has flat images/ + labels/ (no train/ or test/ subfolders), hence we "
                    "recompute it by replaying the exact rule above.",
            "caveat": "it is a PATCH-LEVEL RANDOM split, not a spatial/event hold-out: tiles "
                      "cut from the same orthophoto neighbour each other across train/val, so "
                      "spatial autocorrelation leakage cannot be quantified. Absolute mAP is "
                      "therefore optimistically biased and cross-dataset comparison should lean "
                      "on relative ranking.",
        },
        "totals": {
            "images": len(agg), "train_images": len(train), "val_images": len(val),
            "actual_val_ratio": len(val) / len(agg),
            "instances": sum(r["n_poly"] for r in agg),
            "val_instances": sum(agg_by_stem.get(it["out_stem"], {}).get("n_poly", 0)
                                 for it in val),
            "negatives": sum(1 for r in agg if r["n_cc"] == 0),
            "all_foreground_images": sum(1 for r in agg if r["fg_ref_px"] == S * S),
            "dropped_components": sum(r["n_dropped"] for r in agg),
            "images_with_dropped": sum(1 for r in agg if r["n_dropped"]),
            "fg_px_total": tot_ref,
            "fg_ratio_overall": tot_ref / float(len(agg) * S * S) if agg else 0.0,
        },
        "geometry_stats": {
            "scale_min": min(scales), "scale_median": statistics.median(scales),
            "scale_max": max(scales), "canvas_fill_median": statistics.median(fills),
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
        },
        "audit": {"write_outputs": write_outputs, "errors": len(errs),
                  "error_samples": errs[:5],
                  "verify_bad_size": bad_size,
                  "verify_expected_empty_but_filled": empty_expected,
                  "verify_expected_filled_but_empty": bad_lab,
                  "verify_count_mismatch": stale,
                  "unpaired": missing},
        "timing": {"pass_a_s": round(t_a, 1), "pass_b_s": round(t_b, 1),
                   "workers": workers, "total_s": round(time.time() - t0, 1)},
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
            "images_with_landslide": sum(1 for it in pool if it["n_cc"]),
            "val_images_with_landslide": sum(1 for it in v if it["n_cc"])})

    for reg in sorted({r["region"] for r in agg}):
        pool = [r for r in agg if r["region"] == reg]
        v = [r for r in pool if split_of[r["out_stem"]] == "val"]
        summary["per_region"].append({
            "region": reg, "subset": pool[0]["subset"], "gsd_m": pool[0]["gsd_m"],
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
            "nodata_px": sum(r["nodata_px"] for r in pool),
            "src_size": f"{pool[0]['src_w']}x{pool[0]['src_h']}",
        })

    # summary / manifest 无条件写出（--stats-only 也要刷新清单，与 CAS 脚本一致）
    with open(os.path.join(OUT_ROOT, "split_summary_1024.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_ROOT, "split_manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region", "split", "out_stem", "src_stem", "gsd_m", "sensor",
                    "n_cc", "n_poly", "n_dropped", "fg_px", "rt_iou", "nodata_px"])
        for r in sorted(agg, key=lambda x: (x["region"], split_of[x["out_stem"]],
                                            x["out_stem"])):
            w.writerow([r["region"], split_of[r["out_stem"]], r["out_stem"], r["src_stem"],
                        r["gsd_m"], r["sensor"], r["n_cc"], r["n_poly"], r["n_dropped"],
                        r["fg_ref_px"], f"{r['rt_iou']:.6f}", r["nodata_px"]])

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
        print(f"   {b['region']:<8} src={b['src_size']:<10} n={b['images']:>5} "
              f"tr={b['train_images']:>5} va={b['val_images']:>4} inst={b['n_instances']:>6} "
              f"fg={b['fg_ratio']:.4f} cc_med={b['cc_median']:.0f} "
              f"rtIoU={b['rt_iou_mean']:.4f} drop={b['images_with_dropped']}")
    print(f"\n[OK] 写出到 {OUT_ROOT}  总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", type=int, default=0, metavar="N")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--eps-rel", type=float, default=EPS_REL)
    ap.add_argument("--min-area", type=float, default=MIN_AREA)
    ap.add_argument("--stats-only", action="store_true",
                    help="不写图像/标签，只重算审计统计与清单")
    ap.add_argument("--split", choices=["bucket", "official"], default="bucket",
                    help="bucket=按实例数分桶 12.5%% val（与 LRSTTC/Bijie/CAS 可比，默认）；"
                         "official=复现论文/官方 7:3（sorted->shuffle(seed=2024)，SAT/UAV 各自独立）")
    ap.add_argument("--src", default=SRC_ROOT,
                    help=f"源数据根目录（默认 {SRC_ROOT}）")
    ap.add_argument("--out", default=OUT_ROOT,
                    help=f"输出根目录（默认 {OUT_ROOT}）")
    a = ap.parse_args()
    globals()["SRC_ROOT"] = a.src
    globals()["OUT_ROOT"] = a.out
    MIN_AREA_USE, EPS_REL_USE = a.min_area, a.eps_rel
    print(f"[cfg] eps_rel={EPS_REL_USE} min_area={MIN_AREA_USE} "
          f"(基线 eps_rel={EPS_REL} min_area={MIN_AREA}) workers={a.workers}")
    if a.selftest:
        selftest(a.selftest)
    else:
        main(a.force, a.workers, write_outputs=not a.stats_only, split_mode=a.split)
