"""Recalculate the published class metrics from prediction files, in CSV units."""
import csv
import os
from math import isclose
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from benchmark.eval.paper_metrics import read_json, score_instance_files, score_rgb_masks

MAIN_SEMANTIC = ("unet", "fcn_resnet50", "deeplabv3plus", "mffenet")
MAIN_INSTANCE = ("yolo26l_seg", "mask_rcnn", "solov2", "fastinst")
SOURCES = {"deeplabv3plus": ("dpt-nolixian", "cas", "gdcld", "ls4s"),
           "fastinst": ("dpt-nolixian", "lrsttc", "bijie", "rlzx")}
FIELDS = ("Precision", "Recall", "IoU", "F1", "mask_AP50_original_entry", "mask_AP50_95_original_entry")


def jobs(scope):
    if scope == "main":
        for model in MAIN_SEMANTIC + MAIN_INSTANCE:
            yield f"main/{model}", "FlashLandslide", model, model in MAIN_SEMANTIC, 3
    else:
        for model, sources in SOURCES.items():
            for source in sources:
                group = "nolixian" if source == "dpt-nolixian" else "per_dataset"
                yield f"heldout/{group}/{source}/{model}", source, model, model == "deeplabv3plus", 3 if group == "nolixian" else 2


def score_scope(scope, root, selected_job=None):
    main = scope == "main"
    gt_path = root / "cache" / scope / "coco_val.json"
    gt = read_json(gt_path)
    images = gt["images"]
    counts = Counter(image["region"] for image in images)
    expected_categories = {1: "landslide", 2: "road"} if main else {1: "landslide"}
    expected_support = {1: 3676, 2: 3672} if main else {1: 3421}
    if (len(images) != (7274 if main else 8883) or len(counts) != (18 if main else 1)
            or len({im["id"] for im in images}) != len(images)
            or {c["id"]: c["name"] for c in gt["categories"]} != expected_categories
            or Counter(a["category_id"] for a in gt["annotations"]) != expected_support):
        raise ValueError(f"Unexpected {scope} ground truth")
    for job, source, model, semantic, num_classes in jobs(scope):
        if selected_job and job != selected_job:
            continue
        folder = root / "results" / job
        # The relative job ID links a CSV row to its saved run after the workspace is moved.
        meta = read_json(folder / "metrics.json")
        if not meta["full_validation"] or meta["images"] != len(images):
            raise ValueError(f"Incomplete run: {job}")
        print(f"Evaluating {job} ({len(images)} images)", flush=True)
        if semantic:
            report = score_rgb_masks(gt_path, folder / "rgb", num_classes, main, folder / "chunks")
            overall = report["overall"]["per_class"]
        else:
            report = score_instance_files(gt_path, folder / "coco_predictions.json", main, verify_ap=True)
            overall = report["per_class"]
        scopes = {"overall": overall}
        for name, region in report.get("by_region", {}).items():
            scopes[name] = region["per_class"] if semantic else region
        for name, classes in scopes.items():
            for category in expected_categories.values():
                values = classes[category]
                recorded = (meta["per_class"] if name == "overall" else
                            meta.get("by_region", {}).get(name, {})).get(category)
                if recorded and (semantic or values["support_instances"]):
                    for field in FIELDS:
                        source_field = {"mask_AP50_original_entry": "mask_AP50",
                                        "mask_AP50_95_original_entry": "mask_AP50_95_maxDets100"}.get(field, field)
                        if field in values and not isclose(values[field], recorded[source_field],
                                                           rel_tol=0, abs_tol=1e-12):
                            raise ValueError(f"Saved report disagrees with predictions: {job}/{name}/{category}/{field}")
                support = int(values["support_px" if semantic else "support_instances"])
                row = dict(job=job, training_source=source, model=model, epoch=meta["checkpoint_epoch"],
                           region=name, images=len(images) if name == "overall" else counts[name],
                           task="semantic" if semantic else "instance", category=category, support=support,
                           support_unit="pixels" if semantic else "instances")
                for field in FIELDS:
                    value = values.get(field)
                    row[field + "_percent"] = None if value is None else 100 * value
                row["best_f1_conf"] = values.get("best_f1_conf")
                yield row


def write_paper_table(scope, root, output, selected_job=None):
    torch.set_num_threads(2)
    parts = ("main", "heldout") if scope == "all" else (scope,)
    if selected_job:
        if selected_job not in {job[0] for part in parts for job in jobs(part)}:
            raise ValueError(f"Unknown job for {scope}: {selected_job}")
        parts = (selected_job.split("/")[0],)
    rows = []
    for part in parts:
        rows.extend(score_scope(part, Path(root), selected_job))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with NamedTemporaryFile(mode="w", newline="", encoding="utf-8-sig", delete=False,
                                dir=path.parent, prefix=".class-metrics-") as handle:
            temp = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    print(f"Wrote {len(rows)} class rows to {path}", flush=True)
