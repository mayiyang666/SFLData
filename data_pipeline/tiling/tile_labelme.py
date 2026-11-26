"""把大幅影像切成固定尺寸瓦片，并把 labelme 多边形按瓦片裁剪。

流程中的位置：GSD 已统一之后的切分步骤（上游 ``build_tiles.py`` 调用）。

关键行为（改参数前务必看懂）：
  · **步长 = tile_size − overlap**（默认 1024 − 256 = 768），即相邻瓦片重叠 25%；
  · **边界瓦片回退对齐**：最后一行/列若越界，起点回退到 ``边长 − tile_size``，
    保证每张瓦片都是完整的 tile_size×tile_size（不留黑边）；回退产生的重复位置
    用 ``seen_coords`` 去重，不会重复出图；
  · **多边形裁剪**：与瓦片矩形求交（shapely），MultiPolygon 拆开逐块处理，
    交出的每块面积必须 ≥ ``--min-area``，否则丢弃 —— 这是小目标存活的唯一闸门，
    调小它 = 保留更多碎块（本项目主基准用 50 px）。
  · **小于 tile_size 的图**：不切，整图直出（tile_idx=0）。

输出命名：``{源图stem}_{tile序号:05d}.{ext}`` + 同名 ``.json``，
下游 ``build_tiles.py`` 会再重命名成 ``{区域短名}_{源图编号}_{tile序号:05d}``。

用法::

    python data_pipeline/tiling/tile_labelme.py \
        --input <scaled目录> --output <输出目录> \
        --tile-size 1024 --overlap 256 --min-area 50
"""
import os
import json
import argparse
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

import numpy as np
from shapely.geometry import Polygon, box
import shapely


def parse_args():
    parser = argparse.ArgumentParser(description="Tile large images and LabelMe JSON annotations")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing images and JSON files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for tiled images and JSONs")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size (default: 1024)")
    parser.add_argument("--overlap", type=int, default=256, help="Overlap between tiles (default: 256)")
    parser.add_argument("--min-area", type=int, default=100, help="Minimum area threshold for polygons (default: 100)")
    return parser.parse_args()


def clip_polygon(poly, tile_poly, min_area):
    """Clip polygon to tile and filter by area"""
    if not poly.intersects(tile_poly):
        return []

    intersection = poly.intersection(tile_poly)
    if intersection.is_empty:
        return []

    polygons = []
    if intersection.geom_type == "Polygon":
        polygons = [intersection]
    elif intersection.geom_type == "MultiPolygon":
        polygons = list(intersection.geoms)
    elif intersection.geom_type == "GeometryCollection":
        polygons = [g for g in intersection.geoms if g.geom_type in ["Polygon", "MultiPolygon"]]

    valid_polys = []
    for p in polygons:
        if p.geom_type == "MultiPolygon":
            valid_polys.extend(list(p.geoms))
        elif p.area >= min_area:
            valid_polys.append(p)

    # post processing polygon, 1. make valid, 2. remove small polygons
    valid_polys = [shapely.make_valid(p) for p in valid_polys if p.area >= min_area]
    if not valid_polys:
        return []

    return valid_polys


def process_tile(tile_idx, img, json_data, xmin, ymin, tile_size, min_area):
    """Process a single tile and its annotations"""
    # Calculate tile boundaries
    xmax = min(xmin + tile_size, img.width)
    ymax = min(ymin + tile_size, img.height)

    # Skip degenerate tiles
    if xmax <= xmin or ymax <= ymin:
        return None, None

    # Crop image tile
    tile_img = img.crop((xmin, ymin, xmax, ymax))
    tile_poly = box(xmin, ymin, xmax, ymax)

    # Process annotations
    tile_shapes = []
    for shape in json_data.get("shapes", []):
        if shape["shape_type"] != "polygon":
            continue

        orig_poly = Polygon(shape["points"])
        orig_poly = shapely.make_valid(orig_poly)
        clipped_polys = clip_polygon(orig_poly, tile_poly, min_area)

        for poly in clipped_polys:
            # Adjust coordinates to tile space
            coords = np.array(poly.exterior.coords)
            adjusted_coords = (coords - [xmin, ymin]).tolist()

            tile_shapes.append(
                {
                    "label": shape["label"],
                    "points": adjusted_coords,
                    "group_id": shape.get("group_id", None),
                    "shape_type": "polygon",
                    "flags": shape.get("flags", {}),
                }
            )

    # Create JSON data for tile
    tile_json = {
        "version": "5.1.1",
        "flags": json_data.get("flags", {}),
        "shapes": tile_shapes,
        "imagePath": f"tile_{tile_idx:05d}{os.path.splitext(json_data['imagePath'])[1]}",
        "imageData": None,
        "imageHeight": tile_img.height,
        "imageWidth": tile_img.width,
    }

    return tile_img, tile_json


def main():
    args = parse_args()
    tile_size = args.tile_size
    overlap = args.overlap
    step = tile_size - overlap
    os.makedirs(args.output, exist_ok=True)

    # Process each image/JSON pair
    for fname in os.listdir(args.input):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            continue

        base = os.path.splitext(fname)[0]
        json_path = os.path.join(args.input, f"{base}.json")

        if not os.path.exists(json_path):
            continue

        print(f"Processing: {fname}")

        # Load image
        with Image.open(os.path.join(args.input, fname)) as img:
            # Load JSON
            with open(json_path) as f:
                json_data = json.load(f)

            # Handle images smaller than tile size
            if img.width <= tile_size and img.height <= tile_size:
                print(f"  Image smaller than tile size, copying directly")
                tile_img, tile_json = process_tile(0, img, json_data, 0, 0, tile_size, args.min_area)

                if tile_img is not None:
                    img_out = os.path.join(args.output, f"{base}{os.path.splitext(fname)[1]}")
                    json_out = os.path.join(args.output, f"{base}.json")

                    tile_img.save(img_out)
                    tile_json["imagePath"] = os.path.basename(img_out)
                    with open(json_out, "w") as f:
                        json.dump(tile_json, f, indent=2)
                continue

            # Tile larger images with border adjustment
            tile_count = 0
            seen_coords = set()  # Track processed tile positions

            for y in range(0, img.height, step):
                for x in range(0, img.width, step):
                    xmin = x
                    ymin = y

                    # Adjust coordinates for border tiles
                    if xmin + tile_size > img.width:
                        xmin = max(0, img.width - tile_size)
                    if ymin + tile_size > img.height:
                        ymin = max(0, img.height - tile_size)

                    # Skip duplicate tiles caused by border adjustment
                    if (xmin, ymin) in seen_coords:
                        continue
                    seen_coords.add((xmin, ymin))

                    # Process the tile
                    tile_img, tile_json = process_tile(tile_count, img, json_data, xmin, ymin, tile_size, args.min_area)

                    if tile_img is None:
                        continue

                    # Save outputs
                    tile_prefix = f"{base}_{tile_count:05d}"
                    img_out = os.path.join(args.output, f"{tile_prefix}{os.path.splitext(fname)[1]}")
                    json_out = os.path.join(args.output, f"{tile_prefix}.json")

                    tile_img.save(img_out)
                    tile_json["imagePath"] = os.path.basename(img_out)
                    with open(json_out, "w") as f:
                        json.dump(tile_json, f, indent=2)

                    tile_count += 1


if __name__ == "__main__":
    main()
