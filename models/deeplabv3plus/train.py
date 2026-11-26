#!/usr/bin/env python
"""DeepLabV3+ (ResNet50) benchmark 训练入口（语义分割范式）。

启动（从仓库根目录运行）:
    bash tools/train.sh models/deeplabv3plus/train.py 4
单卡调试:
    python models/deeplabv3plus/train.py
覆盖超参示例:
    bash tools/train.sh models/deeplabv3plus/train.py 4 model.lr=1e-3 model.batch_size=1

评估口径: 语义分割（mIoU / OA / per-class IoU / F1 / Precision / Recall），
主评类别 landslide，统一由 benchmark.eval.seg_eval 计算。
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 保证可导入同目录 model.py

from benchmark.data.loader_utils import shm_safe_worker_init
from benchmark.data.seg_dataset import SegDataset, img_to_device
from benchmark.eval.coco_eval import dump_report
from benchmark.eval.seg_eval import ConfusionAccumulator
from benchmark.runner.base_trainer import BaseTrainer
from torch.utils.data import DataLoader, DistributedSampler

# 评估范式：semantic=语义分割（seg_eval）/ instance=实例分割（COCO mAP）
# tools/evaluate.py 权重模式据此选择评估口径
PARADIGM = "semantic"


def build_model_from_cfg(cfg) -> nn.Module:
    """按配置构建 DeepLabV3+-ResNet50（训练入口与 tools/evaluate.py 共用）。"""
    from model import DeepLabV3PlusResNet50
    arch = cfg.model.arch
    return DeepLabV3PlusResNet50(num_classes=int(cfg.model.num_classes),
                                 pretrained=bool(cfg.model.pretrained),
                                 dropout=float(arch.get("dropout", 0.1)))


class DeepLabV3PlusTrainer(BaseTrainer):
    MODEL_NAME = "deeplabv3plus"

    # ---------------- 模型 ----------------
    def build_model(self) -> nn.Module:
        return build_model_from_cfg(self.cfg)

    # ---------------- 数据 ----------------
    def build_dataloaders(self) -> tuple:
        cache = self.cfg.paths.cache_dir
        input_size = int(self.cfg.protocol.fixed.input_size)
        aug = self.cfg.protocol.fixed.augmentation
        nw = int(self.cfg.protocol.fixed.num_workers)

        train_ds = SegDataset(os.path.join(cache, "coco_train.json"),
                              input_size, train=True, aug_cfg=aug)
        val_ds = SegDataset(os.path.join(cache, "coco_val.json"),
                            input_size, train=False, aug_cfg=None)

        train_sampler = DistributedSampler(train_ds, shuffle=True) if self.is_dist else None
        train_loader = DataLoader(
            train_ds, batch_size=int(self.cfg.model.batch_size),
            shuffle=(train_sampler is None), sampler=train_sampler,
            num_workers=nw, pin_memory=bool(self.cfg.protocol.fixed.pin_memory),
            drop_last=True, worker_init_fn=shm_safe_worker_init)
        val_loader = DataLoader(
            val_ds, batch_size=int(self.cfg.model.val_batch_size),
            shuffle=False, num_workers=nw,
            pin_memory=bool(self.cfg.protocol.fixed.pin_memory),
            worker_init_fn=shm_safe_worker_init)
        return train_loader, val_loader

    # ---------------- 优化器（DeepLab 惯例 SGD；属模型固有项） ----------------
    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        lr = float(self.cfg.model.lr)
        wd = float(self.cfg.model.get("weight_decay", 1e-4))
        if self.cfg.model.optimizer == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr,
                                   momentum=float(self.cfg.model.get("momentum", 0.9)),
                                   weight_decay=wd)
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # ---------------- 损失 / 训练步 ----------------
    def _build_loss(self):
        loss_cfg = self.cfg.model.loss
        weights = loss_cfg.get("class_weights", None)
        w = torch.tensor(list(weights), dtype=torch.float32).to(self.device) \
            if weights is not None else None
        if loss_cfg.type == "ce":
            return nn.CrossEntropyLoss(weight=w)
        raise ValueError(f"未知 loss 类型: {loss_cfg.type}")

    def train_step(self, batch, model, optimizer, scaler) -> dict:
        imgs, masks = batch
        imgs = img_to_device(imgs, self.device)
        masks = masks.to(self.device, non_blocking=True).long()
        logits = model(imgs)
        return {"loss": self.loss_fn(logits, masks)}

    # ---------------- 评估（语义口径，覆盖基类的 COCO 评估） ----------------
    def inference_to_coco(self, model, val_loader, out_json: str) -> None:
        """语义模型不出 COCO results；此处按统一接口输出语义评估报告 JSON。"""
        m = model.module if self.is_dist else model
        m.eval()
        acc = ConfusionAccumulator(int(self.cfg.model.num_classes))
        for imgs, masks in val_loader:
            imgs = img_to_device(imgs, self.device)
            # 评估一律 fp32：深层激活可能溢出 fp16 上限，导致 logits 变 NaN、
            # argmax 全背景
            with torch.amp.autocast("cuda", enabled=False):
                logits = m(imgs)
            acc.update(logits.argmax(1).cpu(), masks)
        class_names = ["background"] + list(self.cfg.dataset.classes)
        report = acc.compute(class_names)
        report["epoch"] = self.epoch + 1
        report["eval_class"] = self.cfg.model.eval_class
        dump_report(report, out_json)

    def _evaluate(self, model, val_loader) -> None:
        out = os.path.join(self.work_dir, f"eval_report_epoch{self.epoch + 1}.json")
        self.logger.info(f"epoch {self.epoch + 1}: semantic eval ...")
        with torch.no_grad():
            self.inference_to_coco(model, val_loader, out)
        import json
        with open(out, "r", encoding="utf-8") as f:
            rep = json.load(f)
        main_cls = self.cfg.model.eval_class
        pc = rep["per_class"].get(main_cls, {})
        self.logger.info(
            f"epoch {self.epoch + 1}: mIoU={rep['mIoU']:.4f} OA={rep['OA']:.4f} "
            f"{main_cls}: IoU={pc.get('IoU', float('nan')):.4f} "
            f"F1={pc.get('F1', float('nan')):.4f} "
            f"P={pc.get('Precision', float('nan')):.4f} "
            f"R={pc.get('Recall', float('nan')):.4f} -> {out}")
        self.metrics.log(epoch=self.epoch + 1, mIoU=rep["mIoU"], OA=rep["OA"],
                         landslide_IoU=pc.get("IoU"), landslide_F1=pc.get("F1"),
                         landslide_Precision=pc.get("Precision"),
                         landslide_Recall=pc.get("Recall"))

    # ---------------- 主流程 ----------------
    def run(self) -> None:
        self.loss_fn = self._build_loss()
        super().run()


def main():
    DeepLabV3PlusTrainer(overrides=sys.argv[1:]).run()


if __name__ == "__main__":
    main()
