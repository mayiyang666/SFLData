#!/usr/bin/env python
"""SOLOv2 (R50-FPN) benchmark 训练入口（实例分割范式）。

启动（从仓库根目录运行）:
    bash tools/train.sh models/solov2/train.py 4
单卡调试:
    python models/solov2/train.py
覆盖超参示例:
    bash tools/train.sh models/solov2/train.py 4 model.lr=5e-3 model.batch_size=1

评估口径: 实例分割 COCO mAP（segm），主评 landslide，
周期评估在训练进程内直接调用 benchmark.eval.coco_eval 完成，
同时保留 results_epoch{N}.json 供 tools/evaluate.py --results 复评。
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 保证可导入同目录 model.py

from benchmark.data.instance_dataset import InstanceDataset, collate_fn
# /dev/shm 泄漏防护：worker 内周期性 gc.collect()
from benchmark.data.loader_utils import shm_safe_worker_init
from benchmark.eval.coco_eval import dump_report, evaluate_by_area, evaluate_coco
from benchmark.eval.instance_infer import infer_to_coco
from benchmark.runner.base_trainer import BaseTrainer
from torch.utils.data import DataLoader, DistributedSampler

# 评估范式：semantic=语义分割（seg_eval）/ instance=实例分割（COCO mAP）
PARADIGM = "instance"


def build_model_from_cfg(cfg) -> nn.Module:
    """按配置构建 SOLOv2（训练入口与 tools/evaluate.py 共用）。"""
    from model import build_solov2
    arch = cfg.model.arch
    return build_solov2(
        num_classes=int(cfg.model.num_classes),
        input_size=int(cfg.protocol.fixed.input_size),
        pretrained_backbone=bool(cfg.model.pretrained),
        arch={
            "feat_channels": int(arch.get("feat_channels", 512)),
            "stacked_convs": int(arch.get("stacked_convs", 4)),
            "strides": [int(s) for s in arch.get("strides", [8, 8, 16, 32, 32])],
            "num_grids": [int(n) for n in arch.get("num_grids", [40, 36, 24, 16, 12])],
            "scale_ranges": [[int(r[0]), int(r[1])] for r in arch.get(
                "scale_ranges", [[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]])],
            "pos_scale": float(arch.get("pos_scale", 0.2)),
            "loss_cls_weight": float(arch.get("loss_cls_weight", 1.0)),
            "loss_mask_weight": float(arch.get("loss_mask_weight", 3.0)),
            "mask_feature_head": dict(arch.get("mask_feature_head", {})),
            "test_cfg": dict(arch.get("test_cfg", {})),
        },
    )


class SOLOv2Trainer(BaseTrainer):
    MODEL_NAME = "solov2"

    # ---------------- 模型 ----------------
    def build_model(self) -> nn.Module:
        return build_model_from_cfg(self.cfg)

    # ---------------- 数据 ----------------
    def build_dataloaders(self) -> tuple:
        cache = self.cfg.paths.cache_dir
        input_size = int(self.cfg.protocol.fixed.input_size)
        aug = self.cfg.protocol.fixed.augmentation
        nw = int(self.cfg.protocol.fixed.num_workers)

        train_ds = InstanceDataset(os.path.join(cache, "coco_train.json"),
                                   input_size, train=True, aug_cfg=aug)
        val_ds = InstanceDataset(os.path.join(cache, "coco_val.json"),
                                 input_size, train=False, aug_cfg=None)

        train_sampler = DistributedSampler(train_ds, shuffle=True) if self.is_dist else None
        train_loader = DataLoader(
            train_ds, batch_size=int(self.cfg.model.batch_size),
            shuffle=(train_sampler is None), sampler=train_sampler,
            num_workers=nw, pin_memory=bool(self.cfg.protocol.fixed.pin_memory),
            collate_fn=collate_fn, drop_last=True,
            worker_init_fn=shm_safe_worker_init)
        val_loader = DataLoader(
            val_ds, batch_size=int(self.cfg.model.val_batch_size),
            shuffle=False, num_workers=nw,
            pin_memory=bool(self.cfg.protocol.fixed.pin_memory),
            collate_fn=collate_fn, worker_init_fn=shm_safe_worker_init)
        return train_loader, val_loader

    # ---------------- 优化器（SOLOv2 惯例 SGD；属模型固有项） ----------------
    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(model.parameters(), lr=float(self.cfg.model.lr),
                               momentum=float(self.cfg.model.get("momentum", 0.9)),
                               weight_decay=float(self.cfg.model.get("weight_decay", 1e-4)))

    # ---------------- 训练步 ----------------
    def train_step(self, batch, model, optimizer, scaler) -> dict:
        images, targets = batch
        images = [im.to(self.device, non_blocking=True) for im in images]
        targets = [{k: v.to(self.device, non_blocking=True)
                    for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)   # {"loss_cls", "loss_mask"}
        total = sum(loss_dict.values())
        out = {"loss": total}
        for k, v in loss_dict.items():
            out[k] = v.detach()
        return out

    # ---------------- 评估（实例口径，覆盖基类） ----------------
    def inference_to_coco(self, model, val_loader, out_json: str) -> None:
        m = model.module if self.is_dist else model
        conf = float(self.cfg.protocol.evaluation.conf_threshold)
        infer_to_coco(m, val_loader, out_json, self.device, conf)

    def _evaluate(self, model, val_loader) -> None:
        results_json = os.path.join(self.work_dir, f"results_epoch{self.epoch + 1}.json")
        self.logger.info(f"epoch {self.epoch + 1}: instance inference -> {results_json}")
        with torch.no_grad():
            self.inference_to_coco(model, val_loader, results_json)

        gt_json = os.path.join(self.cfg.paths.cache_dir, "coco_val.json")
        eval_cfg = self.cfg.protocol.evaluation
        overall = evaluate_coco(gt_json, results_json, eval_cfg, "segm")
        by_area = evaluate_by_area(gt_json, results_json, eval_cfg, "segm")
        report = {"epoch": self.epoch + 1, "model": self.MODEL_NAME,
                  "iou_type": "segm", "overall": overall, "by_area": by_area,
                  "results": results_json}
        report_path = os.path.join(self.work_dir,
                                   f"eval_report_epoch{self.epoch + 1}.json")
        dump_report(report, report_path)
        self.logger.info(
            f"epoch {self.epoch + 1}: mAP={overall['mAP']:.4f} "
            f"mAP_50={overall['mAP_50']:.4f} mAP_75={overall['mAP_75']:.4f} "
            f"s/m/l={by_area['small']:.4f}/{by_area['medium']:.4f}/{by_area['large']:.4f} "
            f"AR_100={overall['AR_100']:.4f} -> {report_path}")
        self.metrics.log(epoch=self.epoch + 1,
                         mAP=overall["mAP"], mAP_50=overall["mAP_50"],
                         mAP_75=overall["mAP_75"],
                         mAP_small=by_area["small"], mAP_medium=by_area["medium"],
                         mAP_large=by_area["large"], AR_100=overall["AR_100"])


def main():
    SOLOv2Trainer(overrides=sys.argv[1:]).run()


if __name__ == "__main__":
    main()
