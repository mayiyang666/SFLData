# -*- coding: utf-8 -*-
"""
Labelme JSON 语义标注可视化(论文插图用)。

类别配色(固定):
  landslide -> 红色 (255, 0, 0)
  road      -> 绿色 (0, 255, 0)
  背景      -> 黑色 (0, 0, 0)

两种输出模式(--mode):
  mask    : 纯语义掩膜图(黑底 + 红/绿区域), 与原图同尺寸、逐像素对齐
  overlay : 掩膜以 alpha 半透明叠加到原图上(便于论文展示原图+标注对照)

用法示例:
  # 批量处理文件夹内所有 labelme json; 图像从各 json 的 imgPath 字段自动解析
  # (找不到时自动在同目录按同名查找), 输出 <json名>_<mode>.png 到 --out 目录
  python data_pipeline/preprocess/visualize_labelme.py --dir path/to/labelme_jsons \
      --out results/paper_fig --mode overlay --alpha 0.55

保存为无损 PNG, dpi 与原图像素 1:1, 无坐标轴/边框, 可直接裁剪进论文。
"""
import os
import json
import argparse

from PIL import Image, ImageDraw

CLASS_COLORS = {
    "landslide": (255, 0, 0),
    "road": (0, 255, 0),
}
# 重叠时后绘制的覆盖先绘制的: 背景 < road < landslide
DRAW_ORDER = ["road", "landslide"]


def resolve_image_path(json_path: str, data: dict, image_arg: str | None) -> str:
    """确定原图路径: --image 参数 > json imgPath > 同目录同名图像。"""
    if image_arg:
        return image_arg
    cand = data.get("imgPath") or data.get("imagePath")
    if cand:
        p = cand if os.path.isabs(cand) else os.path.join(os.path.dirname(json_path), cand)
        if os.path.isfile(p):
            return p
    stem = os.path.splitext(json_path)[0]
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        for c in (stem + ext, stem + ext.upper()):
            if os.path.isfile(c):
                return c
    raise FileNotFoundError(f"找不到与 {json_path} 配对的图像, 请用 --image 显式指定")


def render_mask(data: dict, w: int, h: int) -> Image.Image:
    """渲染纯语义掩膜: 黑底, road 绿, landslide 红(重叠时红覆盖绿)。"""
    mask = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    n_used, n_skip = 0, 0
    for cls in DRAW_ORDER:
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "polygon" or shape.get("label") != cls:
                continue
            pts = [(float(x), float(y)) for x, y in shape["points"]]
            if len(pts) < 3:
                n_skip += 1
                continue
            draw.polygon(pts, fill=CLASS_COLORS[cls])
            n_used += 1
    print(f"[INFO] 多边形: 绘制 {n_used} 个 (跳过无效 {n_skip} 个)")
    other = [s.get("label") for s in data.get("shapes", [])
             if s.get("shape_type") == "polygon" and s.get("label") not in CLASS_COLORS]
    if other:
        print(f"[WARN] 存在未配色类别(已忽略): {sorted(set(other))}")
    return mask


def main():
    ap = argparse.ArgumentParser(description="Labelme JSON 标注可视化(红=滑坡, 绿=道路, 黑=背景)")
    ap.add_argument("--dir", required=True, help="labelme json 文件夹路径")
    ap.add_argument("--image", default=None, help="原图路径(默认从 json 的 imgPath 解析)")
    ap.add_argument("--mode", choices=["mask", "overlay"], default="mask",
                    help="mask=纯掩膜图; overlay=半透明叠加到原图")
    ap.add_argument("--alpha", type=float, default=0.55, help="overlay 模式掩膜透明度")
    ap.add_argument("--out", default=None, help="输出路径(默认 json 同目录 <名>_<mode>.png)")
    args = ap.parse_args()

    for c_json in os.listdir(args.dir):
        if c_json.endswith(".json"):
            c_json = os.path.join(args.dir, c_json)
            with open(c_json, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            img_path = resolve_image_path(c_json, data, args.image)
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            jw, jh = data.get("imageWidth"), data.get("imageHeight")
            if jw and jh and (jw, jh) != (w, h):
                print(f"[WARN] json 记录尺寸 ({jw}x{jh}) 与图像实际尺寸 ({w}x{h}) 不一致, 以图像为准")

            mask = render_mask(data, w, h)

            if args.mode == "mask":
                out_img = mask
            else:
                # 前景区域: 原图与掩膜按 alpha 混合; 背景区域保留原图
                blended = Image.blend(img, mask, args.alpha)
                out_img = Image.composite(blended, img, mask.convert("L"))

            out_path = os.path.join(args.out, os.path.basename(c_json).replace(".json", f"_{args.mode}.png"))
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            out_img.save(out_path)
            print(f"[OK] {args.mode} -> {out_path} ({w}x{h})")


if __name__ == "__main__":
    main()
