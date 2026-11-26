"""共享实例分割推理：检测模型 -> COCO results JSON。

供各实例分割模型 train.py 的 inference_to_coco 与 tools/evaluate.py
权重评估模式共用，保证输出格式全局唯一：

    [{"image_id": int, "category_id": int, "score": float,
      "segmentation": {"counts": ..., "size": [h, w]},   # RLE
      "bbox": [x, y, w, h]}, ...]

约定：
- 模型输出为 torchvision 检测格式 (boxes/labels/scores/masks)，
  masks 为 [N,1,H,W] 概率图，按 0.5 阈值二值化后转 RLE；
- score < conf_threshold 的预测直接丢弃（协议统一 0.001，
  阈值交由 pycocotools 的 PR 曲线体现，不在此截断）；
- 评估一律 fp32（与语义口径一致，避免 AMP 数值风险）。
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from pycocotools import mask as mask_util


def infer_to_coco(model, val_loader, out_json: str, device: torch.device,
                  conf_threshold: float = 0.001) -> list[dict]:
    """对 val_loader 全量推理并写出 COCO results JSON。返回 results 列表。"""
    model.eval()
    results: list[dict] = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = [im.to(device, non_blocking=True) for im in images]
            with torch.amp.autocast("cuda", enabled=False):
                outputs = model(images)
            for out, tgt in zip(outputs, targets):
                image_id = int(tgt["image_id"])
                scores = out["scores"].detach().cpu()
                labels = out["labels"].detach().cpu()
                boxes = out["boxes"].detach().cpu()
                masks = out["masks"].detach().cpu()  # [N,1,H,W] 概率
                keep = scores >= conf_threshold
                for i in keep.nonzero(as_tuple=False).flatten().tolist():
                    m = (masks[i, 0] >= 0.5).to(torch.uint8).numpy()
                    rle = mask_util.encode(np.asfortranarray(m))
                    rle["counts"] = rle["counts"].decode("ascii")
                    x1, y1, x2, y2 = boxes[i].tolist()
                    results.append({
                        "image_id": image_id,
                        "category_id": int(labels[i]),
                        "score": round(float(scores[i]), 6),
                        "segmentation": rle,
                        "bbox": [round(x1, 2), round(y1, 2),
                                 round(x2 - x1, 2), round(y2 - y1, 2)],
                    })
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)
    return results
