"""端到端切片管线：GSD 统一 → 1024 切片 → 目录/命名规范 → YOLO txt。

把「原始 labelme 标注（原图 + json）」变成 benchmark 直接可读的瓦片数据集：

```
{src}/{region}/{split}/{images,labels}          # 输入：原图 + labelme json
   │
   ├─ 1. smooth   多边形样条平滑，顶点数统一到 --num-points（默认 100）
   ├─ 2. scale    按 GSD 缩放：factor = 源GSD / --target-gsd，LANCZOS 重采样，
   │              标注坐标同步缩放（factor ≤ 1 时不缩放，直接复制原图）
   ├─ 3. tile     1024×1024 切片，overlap 256；边界瓦片回退对齐、重复位置去重；
   │              多边形按瓦片裁剪（shapely），面积 < --min-area 的碎片丢弃
   ├─ 4. rename   <源图名>_<tileidx:05d> → <区域短名>_<源图编号>_<tileidx:05d>
   └─ 5. yolo     labels/*.json → 同名 *.txt（见 preprocess/labelme2yolo.py）
   ↓
{out}/{region}/{split}/{images,labels}          # 输出：与 configs/dataset.yaml 的布局完全一致
```

为什么必须有 ``scale`` 这一步：本 benchmark 把所有区域统一到同一 GSD
（主基准 0.15 m/px），而原始无人机影像分布在 0.05~0.5 m/px。不统一 GSD，
"输入分辨率 1024" 在不同区域代表的实际地面范围能差 10 倍，跨区域不可比。

用法::

    # 全量（GSD 从 {src} 的文件夹名后缀 -UAV-<gsd>m- 自动解析）
    python data_pipeline/tiling/build_tiles.py \
        --src ./data/labelme_trainval --out ./data/DPT_region --target-gsd 0.15

    # 单区域
    python data_pipeline/tiling/build_tiles.py --src ./data/labelme_trainval \
        --out ./data/DPT_region --region Wenma

    # 文件夹名没有 GSD 信息时显式指定
    python data_pipeline/tiling/build_tiles.py --src ... --out ... \
        --gsd-map "Wenma=0.2,Lixian_shiziping_317_orin=0.2"

注意：
  · 每个 (region, split) 会**先删掉** ``{out}/{region}/{split}`` 再重建，请确认 out 路径；
  · 中间产物（_work / _scaled / _tiled）默认用完即删，``--keep-temp`` 保留以便排查；
  · 依赖 shapely / scipy / Pillow（见 requirements.txt）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.dirname(HERE)                     # data_pipeline/
SMOOTH_PY = os.path.join(PIPELINE, "preprocess", "smooth_labelme.py")
YOLO_PY = os.path.join(PIPELINE, "preprocess", "labelme2yolo.py")
TILE_PY = os.path.join(HERE, "tile_labelme.py")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
GSD_PAT = re.compile(r"-UAV-(\d+(?:\.\d+)?)m-", re.IGNORECASE)
TILE_IDX_PAT = re.compile(r"(.+)_(\d{5})$")


# ---------------------------------------------------------------- 参数
def parse_args():
    p = argparse.ArgumentParser(
        description="GSD 统一 + 1024 切片 + YOLO 转换 端到端管线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--src", required=True,
                   help="输入根：{src}/{region}/{split}/{images,labels}")
    p.add_argument("--out", required=True,
                   help="输出根：{out}/{region}/{split}/{images,labels}")
    p.add_argument("--region", default=None,
                   help="只处理指定区域，逗号分隔；默认全量")
    p.add_argument("--splits", default="train,val", help="要处理的 split")

    p.add_argument("--target-gsd", type=float, default=0.15,
                   help="目标地面分辨率（m/px）")
    p.add_argument("--gsd-root", default=None,
                   help="从哪个目录的文件夹名解析源 GSD（默认等于 --src）")
    p.add_argument("--gsd-map", default=None,
                   help='显式指定源 GSD，如 "Wenma=0.2,Aerzhai=0.1"，优先级高于自动解析')

    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=256)
    p.add_argument("--min-area", type=float, default=50.0,
                   help="瓦片内多边形最小面积（px）；小于此值的目标碎片丢弃")
    p.add_argument("--num-points", type=int, default=100,
                   help="smooth 后每个多边形的顶点数")
    p.add_argument("--smoothness", type=float, default=0.0,
                   help="样条平滑度；0 = 精确重采样（默认，不改变形状）")

    p.add_argument("--keep-damage", action="store_true",
                   help="保留 damage 类（默认丢弃，与论文口径一致）")
    p.add_argument("--skip-yolo", action="store_true",
                   help="只切片，不生成 YOLO txt")
    p.add_argument("--keep-temp", action="store_true",
                   help="保留 _work/_scaled/_tiled 中间目录")
    return p.parse_args()


# ---------------------------------------------------------------- 工具
def run(cmd):
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"子进程失败（exit {proc.returncode}）: {cmd}")


def short_name(region):
    """去掉 -UAV-x.xm-labelme-xxx 后缀，得到区域短名。"""
    idx = region.find("-UAV-")
    return region[:idx] if idx > 0 else region


def resolve_gsd(region, args):
    """源 GSD 解析（按可信度依次尝试）：

    1. ``--gsd-map`` 显式指定（最高优先）
    2. ``{src}/{region}/meta.json`` —— ``split_trainval.py`` 落盘的源 GSD。
       必需这个来源：划分后目录已被改成区域短名，文件夹名里的 ``-UAV-<gsd>m-``
       已经丢了，光看 --src 是猜不出来的
    3. ``--gsd-root`` / ``--src`` 下的文件夹名后缀 ``-UAV-<gsd>m-``
    4. 都没有 → 报错并给出显式指定方式
    """
    if args.gsd_map:
        table = {}
        for item in args.gsd_map.split(","):
            if not item.strip():
                continue
            k, v = item.split("=")
            table[k.strip()] = float(v)
        for key in (region, short_name(region)):
            if key in table:
                return table[key]

    meta_path = os.path.join(args.src, region, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fp:
            gsd = json.load(fp).get("gsd")
        if gsd:
            print(f"  [gsd] {region}: {gsd} m/px（来自 meta.json）", flush=True)
            return float(gsd)

    root = args.gsd_root or args.src
    for d in os.listdir(root):
        idx = d.find("-UAV-")
        name = d[:idx] if idx > 0 else d
        if name == region:
            m = GSD_PAT.search(d)
            if m:
                return float(m.group(1))
    raise SystemExit(
        f"[build_tiles] 无法确定 {region} 的源 GSD。三种办法：\n"
        f"  1) 让 --src 下的区域目录名保留 GSD 后缀（Xxx-UAV-0.15m-...）；\n"
        f"  2) --gsd-root 指向保留全名的原始目录（如 ./data/labelme_new）；\n"
        f"  3) 直接指定：--gsd-map \"{short_name(region)}=0.15\"")


# ---------------------------------------------------------------- 三步
def smooth_dir(labels_dir, args):
    run([sys.executable, SMOOTH_PY, "--input", labels_dir,
         "--smoothness", str(args.smoothness),
         "--num-points", str(args.num_points)])


def scale_dir(json_dir, img_dir, out_dir, factor):
    """按 factor 缩放图像 + 标注。factor ≤ 1 视为不需要放大，直接复制原图。"""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for jf in sorted(f for f in os.listdir(json_dir) if f.lower().endswith(".json")):
        jpath = os.path.join(json_dir, jf)
        with open(jpath, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        img_name = data.get("imagePath")
        img_src = os.path.join(img_dir, os.path.basename(img_name)) if img_name else None
        if not img_src or not os.path.exists(img_src):
            print(f"  [WARN] {jf}: 找不到图像（imagePath={img_name}），跳过")
            continue
        out_img = os.path.join(out_dir, os.path.basename(img_name))
        if factor <= 1.0001:
            shutil.copy2(img_src, out_img)
            tw, th = data.get("imageWidth"), data.get("imageHeight")
        else:
            with Image.open(img_src) as img:
                ow, oh = img.size
                tw, th = max(1, round(ow * factor)), max(1, round(oh * factor))
                img.resize((tw, th), Image.LANCZOS).save(out_img)
            for shape in data.get("shapes", []):
                shape["points"] = [[round(x * factor, 1), round(y * factor, 1)]
                                   for x, y in shape["points"]]
        data["imageWidth"], data["imageHeight"] = tw, th
        data["imageData"] = None
        with open(os.path.join(out_dir, jf), "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
        n += 1
    print(f"  scaled {n} pairs (factor={factor:.4f})", flush=True)


def tile_dir(in_dir, out_dir, args):
    run([sys.executable, TILE_PY, "--input", in_dir, "--output", out_dir,
         "--tile-size", str(args.tile_size), "--overlap", str(args.overlap),
         "--min-area", str(int(args.min_area))])


# ---------------------------------------------------------------- 命名与落盘
def build_src_index(region, src, splits):
    """源图按字母序编号（train+val 合并），保证瓦片名前缀稳定、可跨进程复现。"""
    stems = set()
    for split in splits:
        txt = os.path.join(src, region, f"{split}.txt")
        if os.path.isfile(txt):
            with open(txt, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        stems.add(os.path.splitext(os.path.basename(line))[0])
            continue
        img_dir = os.path.join(src, region, split, "images")
        if os.path.isdir(img_dir):
            for f in os.listdir(img_dir):
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    stems.add(os.path.splitext(f)[0])
    ordered = sorted(stems)
    return {stem: i for i, stem in enumerate(ordered)}


def rename_and_split(tiled_dir, final_dir, src_idx, region_short):
    dimg = os.path.join(final_dir, "images")
    dlab = os.path.join(final_dir, "labels")
    os.makedirs(dimg, exist_ok=True)
    os.makedirs(dlab, exist_ok=True)
    ni = nl = skipped = 0
    for f in sorted(os.listdir(tiled_dir)):
        stem, ext = os.path.splitext(f)
        src = os.path.join(tiled_dir, f)
        m = TILE_IDX_PAT.match(stem)
        if not m:
            skipped += 1
            continue
        src_stem, tile_idx = m.group(1), m.group(2)
        if src_stem not in src_idx:
            print(f"  [WARN] 源图 {src_stem} 不在清单中，跳过 {f}")
            skipped += 1
            continue
        new_name = f"{region_short}_{src_idx[src_stem]}_{tile_idx}{ext}"
        if ext.lower() in IMG_EXTS:
            shutil.move(src, os.path.join(dimg, new_name)); ni += 1
        elif ext.lower() == ".json":
            shutil.move(src, os.path.join(dlab, new_name)); nl += 1
        else:
            skipped += 1
    print(f"  rename+split -> images {ni} / labels {nl} / skipped {skipped}", flush=True)
    if ni != nl:
        print(f"  [WARN] images 与 labels 数量不一致（{ni} vs {nl}）！"
              "开训前务必核对，否则会有图缺标注或被静默丢弃", flush=True)


# ---------------------------------------------------------------- 主流程
def process_region(region, args, splits):
    gsd = resolve_gsd(region, args)
    factor = gsd / args.target_gsd
    sname = short_name(region)
    print(f"\n===== {region}: {gsd} m/px -> {args.target_gsd} m/px "
          f"(scale={factor:.4f}) =====", flush=True)
    src_idx = build_src_index(region, args.src, splits)
    print(f"  源图 {len(src_idx)} 张（train+val 合并字母序编号）", flush=True)

    for split in splits:
        lab_in = os.path.join(args.src, region, split, "labels")
        img_in = os.path.join(args.src, region, split, "images")
        if not os.path.isdir(lab_in):
            print(f"  [SKIP] {region}/{split}/labels 不存在")
            continue
        base = os.path.join(args.out, "_work", f"{sname}_{split}")
        work, scaled, tiled = base, base + "_scaled", base + "_tiled"
        final = os.path.join(args.out, sname, split)
        for d in (work, scaled, tiled):
            if os.path.exists(d):
                shutil.rmtree(d)
        if os.path.exists(final):
            shutil.rmtree(final)

        # 1) 工作目录：labels 复制（smooth 会原地改写），images 直接引用原目录
        os.makedirs(os.path.join(work, "labels"), exist_ok=True)
        for f in os.listdir(lab_in):
            shutil.copy2(os.path.join(lab_in, f), os.path.join(work, "labels", f))

        # 2) 平滑
        print(f"[{split}] step1 smooth", flush=True)
        smooth_dir(os.path.join(work, "labels"), args)

        # 3) GSD 缩放
        print(f"[{split}] step2 scale", flush=True)
        scale_dir(os.path.join(work, "labels"), img_in, scaled, factor)

        # 4) 切片
        print(f"[{split}] step3 tile", flush=True)
        tile_dir(scaled, tiled, args)

        # 5) 重命名 + 拆分 images/labels
        rename_and_split(tiled, final, src_idx, sname)

        # 6) 清理中间产物
        if not args.keep_temp:
            for d in (work, scaled, tiled):
                if os.path.exists(d):
                    shutil.rmtree(d)

    # 7) YOLO txt：整个 region 的 train+val 一次性转换（沿用 labelme2yolo.py 的
    #    类别口径，避免两处定义漂移；放在两个 split 都切完之后扫一遍即可）
    if not args.skip_yolo:
        print(f"[{sname}] step4 to yolo txt", flush=True)
        cmd = [sys.executable, YOLO_PY, "--root", args.out, "--region", sname]
        if args.keep_damage:
            cmd.append("--keep-damage")
        run(cmd)

    print(f"===== {region} DONE =====", flush=True)


def main():
    args = parse_args()
    args.src = os.path.abspath(args.src)
    args.out = os.path.abspath(args.out)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not os.path.isdir(args.src):
        raise SystemExit(f"[build_tiles] --src 不存在: {args.src}")

    regions = sorted(d for d in os.listdir(args.src)
                     if os.path.isdir(os.path.join(args.src, d))
                     and not d.startswith("_"))
    if args.region:
        want = [s.strip() for s in args.region.split(",") if s.strip()]
        unknown = [w for w in want if w not in regions]
        if unknown:
            raise SystemExit(f"[build_tiles] 未知区域: {unknown}\n可选: {regions}")
        regions = want

    print(f"regions={regions}")
    print(f"target_gsd={args.target_gsd}  tile={args.tile_size}  "
          f"overlap={args.overlap}  min_area={args.min_area}")
    for r in regions:
        process_region(r, args, splits)

    work_root = os.path.join(args.out, "_work")
    if os.path.isdir(work_root) and not os.listdir(work_root):
        os.rmdir(work_root)
    print("\nALL DONE.")


if __name__ == "__main__":
    main()
