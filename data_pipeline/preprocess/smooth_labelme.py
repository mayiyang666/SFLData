"""多边形样条平滑 + 顶点数归一（labelme → 统一点数）。

流程中的位置：标注清理后的第一道几何处理，在 GSD 缩放与切片之前。
作用：
  · 把人工标注的稀疏多边形重采样成固定 ``--num-points``（默认 100）个顶点，
    使后续缩放/切片时多边形边界密度一致；
  · ``--smoothness 0`` + ``degree 1`` 时是**精确重采样**（不改变形状，只均匀布点），
    本项目全部实验都用这个设置 —— 形状改了就不是原始标注了；
  · 顺带用 shapely 修自交多边形（make_valid），并对 MultiPolygon 取最大块。

用法::

    python data_pipeline/preprocess/smooth_labelme.py --input <json目录> --smoothness 0

注意：``--output`` 不传时**原地改写** JSON（id 与文件名保持不变），
所以正式流程里先复制到工作目录再跑（``build_tiles.py`` 就是这么做的）。
"""
import os
import json
import argparse
import numpy as np
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon
from shapely.validation import make_valid


def smooth_polygon(points, num_points=100, smoothness=0.0, degree=1):
    """
    Smooth polygon using spline interpolation
    :param points: List of (x,y) points
    :param num_points: Number of points in output polygon
    :param smoothness: Smoothness parameter (0.0 = exact fit, higher = smoother)
    :param degree: Degree of spline interpolation (default 1 for linear)
    :return: Smoothed polygon points
    """
    if len(points) < 10:
        return points

    # Convert to numpy array and close polygon
    points = np.array(points)

    # Force close polygon
    points = np.vstack([points, points[0]])

    # Remove duplicates
    unique_mask = np.concatenate([[True], ~np.all(np.isclose(points[1:], points[:-1], atol=1e-3), axis=1)])
    points = points[unique_mask]

    # Separate coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Fit spline to points
    try:
        degree = 1
        used_degree = degree if len(points) > degree + 1 else len(points) - 1
        tck, u = splprep([x, y], u=None, k=used_degree, s=smoothness, per=1)
    except Exception as e:
        print(f"Spline fitting error: {e}")
        return points.tolist()[:-1]

    # Evaluate spline
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck, der=0)

    # Return polygon points without duplicate closure point
    return np.column_stack([x_new, y_new]).tolist()[:-1]


def process_json(json_path, output_path, num_points, smoothness, tolerance, degree):
    """
    Process a single LabelMe JSON file
    :param json_path: Input JSON file path
    :param output_path: Output JSON file path
    :param num_points: Number of points for spline interpolation
    :param smoothness: Smoothness parameter
    :param tolerance: Polygon simplification tolerance
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Count processed polygons
    processed = 0
    skipped = 0

    # Process each shape
    try:

        for shape in data["shapes"]:
            if shape["shape_type"] != "polygon":
                continue

            original_points = shape["points"]

            # Skip polygons with less than 3 points
            if len(original_points) < 3:
                skipped += 1
                continue

            # Smooth polygon with spline interpolation
            smoothed_points = smooth_polygon(original_points, num_points, smoothness, degree)

            # Convert to Shapely polygon
            poly = Polygon(smoothed_points)

            # Validate geometry
            if not poly.is_valid:
                poly = make_valid(poly)

                # Handle multi-polygons
                if poly.geom_type == "MultiPolygon":
                    # Take the largest polygon
                    poly = max(poly.geoms, key=lambda p: p.area)

            # Simplify polygon
            if tolerance > 0:
                poly = poly.simplify(tolerance, preserve_topology=True)

            # Get final polygon points (ignore holes)
            if not poly.is_empty:
                if poly.geom_type == "Polygon":
                    shape["points"] = list(poly.exterior.coords)[:-1]
                    processed += 1
                elif poly.geom_type == "MultiPolygon":
                    # Take the largest polygon
                    largest_poly = max(poly.geoms, key=lambda p: p.area)
                    shape["points"] = list(largest_poly.exterior.coords)[:-1]
                    processed += 1

        # keep the precision of the points to only 2 decimal place
        for shape in data["shapes"]:
            shape["points"] = [[round(coord, 2) for coord in point] for point in shape["points"]]

        # Save processed JSON
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return processed, skipped
    except Exception as e:
        return processed, skipped


def main():
    parser = argparse.ArgumentParser(description="Smooth and simplify LabelMe polygon annotations")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file or directory")
    parser.add_argument("--output", type=str, help="Output directory (default: overwrite input)")
    parser.add_argument(
        "--num-points", type=int, default=100, help="Number of points for smoothed polygon (default: 100)"
    )
    parser.add_argument(
        "--smoothness",
        type=float,
        default=0.5,
        help="Spline smoothness (0.0=exact fit, higher=smoother) (default: 0.5)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1.0, help="Simplification tolerance in pixels (default: 1.0)"
    )
    parser.add_argument("--degree", type=int, default=1, help="Degree of spline interpolation (default: 1)")
    args = parser.parse_args()

    # Create output directory if needed
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Process single file
    if os.path.isfile(args.input) and args.input.endswith(".json"):
        output_path = os.path.join(args.output, os.path.basename(args.input)) if args.output else args.input
        process_json(args.input, output_path, args.num_points, args.smoothness,
                     args.tolerance, args.degree)

    # Process directory
    elif os.path.isdir(args.input):
        for filename in os.listdir(args.input):
            if not filename.endswith(".json"):
                continue

            input_path = os.path.join(args.input, filename)
            output_path = os.path.join(args.output, filename) if args.output else input_path

            processed, skipped = process_json(input_path, output_path, args.num_points, args.smoothness, args.tolerance, args.degree)

            print(f"Processed {filename}: {processed} polygons smoothed/simplified, {skipped} skipped")

    else:
        print("Error: Input must be a JSON file or directory containing JSON files")


if __name__ == "__main__":
    main()
