"""Scoring routines used for the SFLData main and Shiziping evaluations."""

import io
import json
from math import isclose
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from benchmark.config import load_protocol_cfg
from benchmark.data.seg_dataset import SegDataset
from benchmark.eval.coco_eval import evaluate_coco
from benchmark.eval.seg_eval import ConfusionAccumulator

COLORS = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.uint8)
CLASS_NAMES = ["background", "landslide", "road"]


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_instances(gt, rows, *, by_region=False):
    # COCOeval handles matching and computes AP. The matched predictions at IoU=.5
    # give the continuous confidence sweep for precision and recall.
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    if rows:
        predicted = coco.loadRes([dict(row) for row in rows])
    else:
        predicted = COCO()
        predicted.dataset = dict(gt, annotations=[])
        predicted.createIndex()

    def evaluate(ids):
        ev = COCOeval(coco, predicted, "segm")
        ev.params.imgIds = sorted(ids)
        ev.params.maxDets = [100, 300, 1000]
        ev.params.areaRng = [[0, 1e10]]
        ev.params.areaRngLbl = ["all"]
        ev.evaluate()
        ev.accumulate()
        per_class = {}
        for index, category in enumerate(ev.params.catIds):
            name = coco.cats[category]["name"]
            # Preserve both AP limits. The paper uses AP50 at 1000 and AP50:95 at 100.
            scores, matches = [], []
            n_gt = 0
            start = index * len(ids)
            for image in ev.evalImgs[start:start + len(ids)]:
                if image is None:
                    continue
                n_gt += int((image["gtIgnore"] == 0).sum())
                keep = ~image["dtIgnore"][0, :100]
                scores.extend(np.asarray(image["dtScores"][:100])[keep])
                matches.extend(image["dtMatches"][0, :100][keep] > 0)
            p = r = f1 = 0.0
            threshold = None
            if scores and n_gt:
                scores = np.asarray(scores)
                order = np.argsort(-scores, kind="stable")
                ordered = scores[order]
                endpoints = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(order) - 1]
                tp = np.cumsum(np.asarray(matches)[order])[endpoints]
                ps, rs = tp / (endpoints + 1), tp / n_gt
                fs = 2 * ps * rs / np.maximum(ps + rs, 1e-12)
                i = int(fs.argmax())
                p, r, f1, threshold = map(float, (ps[i], rs[i], fs[i], ordered[endpoints[i]]))
            precision = ev.eval["precision"][:, :, index, 0, :]

            def mean_ap(points):
                valid = points[points >= 0]
                return float(valid.mean()) if len(valid) else None

            per_class[name] = {
                "Precision": p, "Recall": r, "F1": f1, "best_f1_conf": threshold,
                "mask_AP50": mean_ap(precision[0, :, -1]),
                "mask_AP50_95": mean_ap(precision[:, :, -1]),
                "mask_AP50_maxDets100": mean_ap(precision[0, :, 0]),
                "mask_AP50_95_maxDets100": mean_ap(precision[:, :, 0]),
                "support_instances": n_gt,
            }
        return per_class

    report = {"per_class": evaluate(set(coco.imgs))}
    if by_region:
        regions = defaultdict(set)
        for im in gt["images"]:
            regions[im.get("region", "unknown")].add(im["id"])
        report["by_region"] = {region: evaluate(ids) for region, ids in sorted(regions.items())}
    return report


def decode_saved_rgb(path):
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"))
    labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
    labels[rgb[..., 0] == 255] = 1
    labels[rgb[..., 1] == 255] = 2
    if not np.array_equal(rgb, COLORS[labels]):
        raise ValueError(f"Invalid prediction palette in {path}")
    return torch.from_numpy(labels)


def score_instance_files(gt_path, prediction_path, by_region=False, verify_ap=False):
    """Apply the final class-wise mask protocol to saved COCO detections."""
    gt = read_json(gt_path)
    rows = read_json(prediction_path)
    image_ids = {image["id"] for image in gt["images"]}
    if any(row["image_id"] not in image_ids for row in rows):
        raise ValueError("Instance predictions include unknown image IDs")
    categories = {item["id"] for item in gt["categories"]}
    rows = [row for row in rows if row["category_id"] in categories]
    with redirect_stdout(io.StringIO()):
        report = evaluate_instances(deepcopy(gt), rows, by_region=by_region)

    # Direct calls to the original evaluator were part of the final paper audit.
    scopes = [(None, report["per_class"]), *report.get("by_region", {}).items()]
    for region, classes in scopes:
        images = gt["images"] if region is None else [
            image for image in gt["images"] if image.get("region", "unknown") == region]
        ids = {image["id"] for image in images}
        for category in gt["categories"]:
            metrics = classes[category["name"]]
            ap50 = metrics["mask_AP50"]
            ap = metrics["mask_AP50_95_maxDets100"]
            if verify_ap and metrics["support_instances"]:
                selected = [row for row in rows if row["image_id"] in ids
                            and row["category_id"] == category["id"]]
                if selected:
                    annotations = [ann for ann in gt["annotations"] if ann["image_id"] in ids
                                   and ann["category_id"] == category["id"]]
                    subset = dict(gt, images=images, annotations=annotations, categories=[category])
                    with TemporaryDirectory(prefix="sfl-coco-") as directory:
                        truth = Path(directory) / "gt.json"
                        detections = Path(directory) / "detections.json"
                        truth.write_text(json.dumps(subset), encoding="utf-8")
                        detections.write_text(json.dumps(selected), encoding="utf-8")
                        with redirect_stdout(io.StringIO()):
                            native = evaluate_coco(str(truth), str(detections),
                                                   load_protocol_cfg().evaluation, "segm")
                    if not (isclose(native["mAP_50"], ap50, abs_tol=1e-12, rel_tol=0)
                            and isclose(native["mAP"], ap, abs_tol=1e-12, rel_tol=0)):
                        raise ValueError(f"COCO AP cross-check failed for {region}/{category['name']}")
                    ap50, ap = native["mAP_50"], native["mAP"]
            metrics["mask_AP50_original_entry"] = ap50
            metrics["mask_AP50_95_original_entry"] = ap
            if not metrics["support_instances"]:
                for field in ("Precision", "Recall", "F1"):
                    metrics[field] = None
    return report


def score_rgb_masks(gt_path, rgb_dir, num_classes, by_region=False, chunks=None):
    """Rebuild confusion matrices from the saved lossless class maps."""
    if num_classes not in (2, 3):
        raise ValueError("Expected two or three semantic model classes")
    dataset = SegDataset(str(gt_path), 1024, train=False)
    images = dataset.coco["images"]
    if not images:
        raise ValueError("Empty validation set")
    rgb_dir = Path(rgb_dir)
    paths = []
    for image in images:
        tile = image.get("tile_id") or Path(image["file_name"].replace("\\", "/")).stem
        paths.append(Path(image.get("region", "")) / (tile + ".png"))
    if len(set(paths)) != len(images) or {p.relative_to(rgb_dir) for p in rgb_dir.rglob("*.png")} != set(paths):
        raise ValueError(f"RGB inventory differs from ground truth in {rgb_dir}")

    def per_image(pair):
        image, relative = pair
        prediction = decode_saved_rgb(rgb_dir / relative).numpy()
        h, w = image["height"], image["width"]
        target = dataset._render_mask(image["id"], h, w)
        if prediction.shape != (h, w) or prediction.max() >= num_classes or target.max() >= num_classes:
            raise ValueError(f"Invalid mask dimensions or class ID for {relative}")
        return image.get("region", "unknown"), np.bincount(
            (target * num_classes + prediction).ravel(), minlength=num_classes**2)

    matrices = defaultdict(lambda: np.zeros(num_classes**2, dtype=np.int64))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for region, matrix in pool.map(per_image, zip(images, paths)):
            matrices[region] += matrix
    combined = np.sum(list(matrices.values()), axis=0)
    if chunks is not None and chunks.is_dir():
        recorded = sum((np.array(read_json(p), dtype=np.int64) for p in chunks.glob("*.json")),
                       np.zeros_like(combined))
        if not np.array_equal(recorded, combined):
            raise ValueError("RGB-derived confusion matrix differs from the saved chunks")

    def compute(matrix):
        acc = ConfusionAccumulator(num_classes)
        acc.cm = torch.from_numpy(matrix).double()
        return acc.compute(CLASS_NAMES[:num_classes])

    result = {"overall": compute(combined), "confusion_matrix": combined.reshape(num_classes, num_classes).tolist()}
    if by_region:
        result["by_region"] = {region: compute(matrix) for region, matrix in matrices.items()}
    return result
