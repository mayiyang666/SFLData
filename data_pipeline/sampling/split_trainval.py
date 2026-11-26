"""按滑坡实例数分层抽样，把每个区域划成 train / val。

这是"采样"这一步：**不是随机切**，而是按「每张图的滑坡实例数」分桶后逐桶抽样，
保证 val 的实例/影像比与 train 一致 —— 滑坡数据极度不平衡（大量无目标图 +
少数密集图），纯随机切会让 val 的实例密度随机漂移，指标不可比。

分桶（与论文 §2.3 一致）::

    0 / 1 / 2 / 3 / 4-6 / 7-10 / 11-20 / 21+

其它行为：
  1. **孤儿 JSON**（无对应图像的标注）移到 ``--backup`` 备份目录，不直接删；
  2. 清理非法标签（仅保留 landslide / road / damage —— damage 在后续
     ``labelme2yolo.py`` 转 YOLO 时才丢弃，这里保留是为了溯源）；
  3. 输出 ``train.txt`` / ``val.txt`` / ``split_summary.csv``，并可物理复制成
     ``{region}/{split}/{images,labels}`` 结构。

用法::

    python data_pipeline/sampling/split_trainval.py \
        --root ./data/labelme_new --out ./data/labelme_trainval \
        --val-frac 0.125 --seed 42

复现论文口径就用上面的参数（val 12.5% / seed 42）；``--no-copy`` 只出清单不复制。
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VALID_LABELS = {"landslide", "road", "damage"}
GSD_PAT = re.compile(r"-UAV-(\d+(?:\.\d+)?)m-", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="输入根目录：{root}/{region}/ 下平铺 images + json")
    p.add_argument("--out", required=True,
                   help="输出根目录：{out}/{region}/{train,val}/{images,labels} + 清单")
    p.add_argument("--backup", default=None,
                   help="孤儿 JSON 备份目录（默认 {out}/_orphan_backup）")
    p.add_argument("--val-frac", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-copy", action="store_true",
                   help="仅生成划分清单，不做物理复制")
    return p.parse_args()


def short_name(region):
    """去掉 -UAV-x.xm-labelme-xxx 后缀得到区域短名"""
    idx = region.find("-UAV-")
    return region[:idx] if idx > 0 else region


def parse_gsd(region):
    """从源文件夹名 -UAV-<gsd>m- 解析原始地面分辨率（m/px）；解析不到返回 None。

    会把结果写进 ``{out}/{region}/meta.json``，供下游 ``build_tiles.py`` 直接读取
    （否则它只能看到被改成短名的目录，无法得知源 GSD）。
    """
    m = GSD_PAT.search(region)
    return float(m.group(1)) if m else None


def ls_bucket(n):
    """滑坡实例数分桶（分层抽样用）"""
    if n == 0:
        return "0"
    if n <= 3:
        return str(n)
    if n <= 6:
        return "4-6"
    if n <= 10:
        return "7-10"
    if n <= 20:
        return "11-20"
    return "21+"


def main():
    args = parse_args()
    if not args.backup:
        args.backup = os.path.join(args.out, "_orphan_backup")
    rng = random.Random(args.seed)
    regions = sorted(d for d in os.listdir(args.root)
                     if os.path.isdir(os.path.join(args.root, d)))
    print(f"[INFO] {len(regions)} regions | val_frac={args.val_frac} seed={args.seed}")

    summary_rows = []
    total_orphans = 0
    total_bad_shapes = 0

    for region in regions:
        rdir = os.path.join(args.root, region)
        sname = short_name(region)
        files = os.listdir(rdir)
        imgs = {}
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXTS:
                imgs[stem.lower()] = f
        jsons = sorted(f for f in files if f.lower().endswith(".json"))

        # ---------- 1) 清理孤儿 JSON ----------
        orphans = [j for j in jsons
                   if os.path.splitext(j)[0].lower() not in imgs]
        if orphans:
            bdir = os.path.join(args.backup, sname)
            os.makedirs(bdir, exist_ok=True)
            for j in orphans:
                shutil.move(os.path.join(rdir, j), os.path.join(bdir, j))
            total_orphans += len(orphans)

        # ---------- 2) 删除异常标签 + 统计滑坡实例 ----------
        img_ls_count = {}
        for j in jsons:
            stem = os.path.splitext(j)[0]
            if stem.lower() not in imgs:
                continue  # 孤儿已移走
            jpath = os.path.join(rdir, j)
            with open(jpath, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            shapes = data.get("shapes", []) or []
            kept = [s for s in shapes if s.get("label", "") in VALID_LABELS]
            n_removed = len(shapes) - len(kept)
            if n_removed:
                total_bad_shapes += n_removed
                data["shapes"] = kept
                with open(jpath, "w", encoding="utf-8") as fp:
                    json.dump(data, fp, ensure_ascii=False, indent=2)
            n_ls = sum(1 for s in kept if s.get("label") == "landslide")
            img_ls_count[imgs[stem.lower()]] = n_ls

        # ---------- 3) 分层伪随机划分 ----------
        buckets = defaultdict(list)
        for img, n in img_ls_count.items():
            buckets[ls_bucket(n)].append(img)

        val_imgs, train_imgs = [], []
        for b in sorted(buckets.keys()):
            pool = buckets[b][:]
            rng.shuffle(pool)
            k = round(len(pool) * args.val_frac)
            val_imgs.extend(pool[:k])
            train_imgs.extend(pool[k:])

        n_tot = len(img_ls_count)
        ls_tot = sum(img_ls_count.values())
        ls_val = sum(img_ls_count[i] for i in val_imgs)
        ls_tr = ls_tot - ls_val
        n_val, n_tr = len(val_imgs), len(train_imgs)
        r_tr = ls_tr / n_tr if n_tr else 0
        r_val = ls_val / n_val if n_val else 0

        summary_rows.append({
            "region": sname, "folder": region,
            "n_images": n_tot, "ls_total": ls_tot,
            "train_imgs": n_tr, "train_ls": ls_tr,
            "val_imgs": n_val, "val_ls": ls_val,
            "val_img_pct": round(100 * n_val / n_tot, 1),
            "val_ls_pct": round(100 * ls_val / ls_tot, 1) if ls_tot else 0,
            "inst_per_img_train": round(r_tr, 2),
            "inst_per_img_val": round(r_val, 2),
        })

        # ---------- 4) 写出清单 ----------
        region_out = os.path.join(args.out, sname)
        os.makedirs(region_out, exist_ok=True)
        with open(os.path.join(region_out, "train.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(train_imgs)))
        with open(os.path.join(region_out, "val.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(val_imgs)))

        # 供 build_tiles.py 读取：源 GSD（短名目录里已丢失该信息）+ 划分参数
        with open(os.path.join(region_out, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"region": sname, "source_folder": region,
                       "gsd": parse_gsd(region),
                       "val_frac": args.val_frac, "seed": args.seed,
                       "n_images": n_tot, "n_train": n_tr, "n_val": n_val},
                      f, ensure_ascii=False, indent=2)

        # ---------- 5) 物理复制 ----------
        if not args.no_copy:
            for split, lst in (("train", train_imgs), ("val", val_imgs)):
                dimg = os.path.join(region_out, split, "images")
                dlab = os.path.join(region_out, split, "labels")
                os.makedirs(dimg, exist_ok=True)
                os.makedirs(dlab, exist_ok=True)
                stem_map = {os.path.splitext(v)[0].lower(): v for v in imgs.values()}
                for im in lst:
                    stem = os.path.splitext(im)[0].lower()
                    shutil.copy2(os.path.join(rdir, im), os.path.join(dimg, im))
                    jf = os.path.splitext(im)[0] + ".json"
                    src_j = os.path.join(rdir, jf)
                    if os.path.exists(src_j):
                        shutil.copy2(src_j, os.path.join(dlab, jf))
        print(f"[DONE] {sname}: imgs {n_tot} -> train {n_tr} / val {n_val} "
              f"| ls inst {ls_tot} -> train {ls_tr} / val {ls_val} "
              f"| inst/img train {r_tr:.2f} val {r_val:.2f}"
              + (f" | orphans {len(orphans)}" if orphans else ""))

    # ---------- 汇总 ----------
    csv_path = os.path.join(args.out, "split_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    T = {k: sum(r[k] for r in summary_rows) for k in
         ["n_images", "ls_total", "train_imgs", "train_ls", "val_imgs", "val_ls"]}
    print("\n========== SPLIT SUMMARY ==========")
    print(f"orphan jsons moved : {total_orphans}  -> {args.backup}")
    print(f"bad shapes removed : {total_bad_shapes}")
    print(f"images  total/train/val : {T['n_images']} / {T['train_imgs']} / {T['val_imgs']} "
          f"({100*T['val_imgs']/T['n_images']:.1f}%)")
    print(f"ls inst total/train/val : {T['ls_total']} / {T['train_ls']} / {T['val_ls']} "
          f"({100*T['val_ls']/T['ls_total']:.1f}%)")
    print(f"inst/img: train {T['train_ls']/T['train_imgs']:.2f} | "
          f"val {T['val_ls']/T['val_imgs']:.2f}")
    print(f"[SAVED] {csv_path}")


if __name__ == "__main__":
    main()
