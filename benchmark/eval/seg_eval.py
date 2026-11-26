"""共享语义分割评估：混淆矩阵累计与 mIoU / OA / F1 / Precision / Recall。

与 protocol.evaluation.report_metrics.semantic 对应。
所有语义分割模型（U-Net / FCN / DeepLabV3+ / SegFormer）统一走本模块。
指标口径：
- per-class: IoU / Precision / Recall / F1
- 全局: mIoU（各类 IoU 均值）、OA（Overall Accuracy，即像素级准确率）、
  mean_Precision / mean_Recall / mean_F1（各类均值）
"""
from __future__ import annotations

import torch


class ConfusionAccumulator:
    """逐 batch 累计混淆矩阵（向量化 bincount 实现，支持多类）。"""

    def __init__(self, num_classes: int, ignore_index: int = -1):
        self.K = int(num_classes)
        self.ignore_index = ignore_index
        self.cm = torch.zeros(self.K * self.K, dtype=torch.float64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred/target: long/uint8 [B,H,W] 或 [H,W]（内部统一转 long）。"""
        p = pred.reshape(-1).cpu().long()
        t = target.reshape(-1).cpu().long()
        if self.ignore_index >= 0:
            keep = t != self.ignore_index
            p, t = p[keep], t[keep]
        idx = t * self.K + p
        self.cm += torch.bincount(idx, minlength=self.K * self.K).double()

    def matrix(self) -> torch.Tensor:
        return self.cm.reshape(self.K, self.K)

    def compute(self, class_names: list[str] | None = None) -> dict:
        """返回每类 IoU/P/R/F1 与全局均值指标（mIoU / OA / mean_P/R/F1）。"""
        cm = self.matrix()
        tp = cm.diagonal()
        fp = cm.sum(dim=0) - tp          # 按列：被预测为该类但 GT 不是
        fn = cm.sum(dim=1) - tp          # 按行：GT 为该类但预测错
        eps = 1e-12
        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        pix_acc = tp.sum() / (cm.sum() + eps)

        names = class_names or [f"class_{i}" for i in range(self.K)]
        per_class = {}
        for i, name in enumerate(names):
            per_class[name] = {
                "IoU": float(iou[i]), "F1": float(f1[i]),
                "Precision": float(precision[i]), "Recall": float(recall[i]),
                "support_px": float(cm[i].sum()),
            }
        return {
            "mIoU": float(iou.mean()),
            "OA": float(pix_acc),
            "mean_Precision": float(precision.mean()),
            "mean_Recall": float(recall.mean()),
            "mean_F1": float(f1.mean()),
            "pixel_accuracy": float(pix_acc),  # OA 别名，保留向后兼容
            "per_class": per_class,
        }
