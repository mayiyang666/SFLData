"""统一实例分割评估：所有模型输出统一为 COCO results JSON 后在此评估。

协议（来自 configs/protocol.yaml）:
- backend: pycocotools
- 指标: mAP / mAP@50 / mAP@75 / mAP_s / mAP_m / mAP_l / AR@1000
- 按实例面积分桶 (small/medium/large) 与按区域 (region) 细分报告

模型侧约定输出格式（results.json，COCO detection/segmentation results）:
    [{"image_id": int, "category_id": int, "score": float,
      "segmentation": {"counts": ..., "size": [h, w]} 或 [[x1,y1,...]],
      "bbox": [x, y, w, h]}, ...]
image_id 必须与 coco_{split}.json 中的 id 一致。
"""
from __future__ import annotations

import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

METRIC_NAMES = ["mAP", "mAP_50", "mAP_75", "mAP_small", "mAP_medium",
                "mAP_large", "AR_100", "AR_300", "AR_1000"]


def _empty_report() -> dict:
    """空预测兜底：训练早期模型可能零检出，pycocotools 无法处理空
    results，此时返回全 NaN 而非让训练崩溃。"""
    return {name: float("nan") for name in METRIC_NAMES}


def _is_empty_results(results_json: str) -> bool:
    with open(results_json, "r", encoding="utf-8") as f:
        return len(json.load(f)) == 0


def load_gt(gt_json: str) -> COCO:
    return COCO(gt_json)


def _summarize(coco_eval: COCOeval) -> dict:
    s = coco_eval.stats
    return {name: float(v) for name, v in zip(METRIC_NAMES, s)}


def evaluate_coco(gt_json: str, results_json: str, eval_cfg,
                  iou_type: str = "segm") -> dict:
    """主评估入口。返回总体指标 dict。"""
    if _is_empty_results(results_json):
        return _empty_report()
    coco_gt = load_gt(gt_json)
    coco_dt = coco_gt.loadRes(results_json)

    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    ev.params.maxDets = list(eval_cfg.max_dets)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return _summarize(ev)


def evaluate_by_area(gt_json: str, results_json: str, eval_cfg,
                     iou_type: str = "segm") -> dict:
    """按协议中 area_ranges 分桶评估。

    pycocotools.summarize 依赖完整的 all/small/medium/large 四档区间，
    因此整体覆盖后取 stats[3]/stats[4]/stats[5]（对应 s/m/l 的 AP）。
    """
    ranges = eval_cfg.area_ranges
    if _is_empty_results(results_json):
        return {"small": float("nan"), "medium": float("nan"),
                "large": float("nan")}
    coco_gt = load_gt(gt_json)
    coco_dt = coco_gt.loadRes(results_json)
    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    ev.params.maxDets = list(eval_cfg.max_dets)
    ev.params.areaRng = [[0, 1e10],
                         list(ranges.small), list(ranges.medium), list(ranges.large)]
    ev.params.areaRngLbl = ["all", "small", "medium", "large"]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"small": float(ev.stats[3]),
            "medium": float(ev.stats[4]),
            "large": float(ev.stats[5])}


def evaluate_by_region(gt_json: str, results_json: str, eval_cfg,
                       iou_type: str = "segm") -> dict:
    """按区域拆分评估（论文中可报告各子数据集表现）。"""
    with open(gt_json, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(results_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    regions = sorted({img.get("region", "unknown") for img in gt["images"]})
    res_by_img = {}
    for r in results:
        res_by_img.setdefault(r["image_id"], []).append(r)

    out = {}
    for region in regions:
        imgs = [im for im in gt["images"] if im.get("region") == region]
        ids = {im["id"] for im in imgs}
        sub_gt = dict(gt, images=imgs,
                      annotations=[a for a in gt["annotations"] if a["image_id"] in ids])
        sub_res = [r for iid in ids for r in res_by_img.get(iid, [])]
        if not sub_res:
            out[region] = {"mAP": float("nan")}
            continue
        tmp_gt = gt_json + f".tmp_{region}.json"
        tmp_res = results_json + f".tmp_{region}.json"
        with open(tmp_gt, "w") as f:
            json.dump(sub_gt, f)
        with open(tmp_res, "w") as f:
            json.dump(sub_res, f)
        out[region] = evaluate_coco(tmp_gt, tmp_res, eval_cfg, iou_type)
        os.remove(tmp_gt)
        os.remove(tmp_res)
    return out


def dump_report(report: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
