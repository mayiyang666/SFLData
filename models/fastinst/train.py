#!/usr/bin/env python
"""FastInst (R50 + PPM-FPN) benchmark 训练入口（实例分割范式）。

启动（从仓库根目录运行）:
    bash tools/train.sh models/fastinst/train.py 4
单卡调试:
    python models/fastinst/train.py
覆盖超参示例:
    bash tools/train.sh models/fastinst/train.py 4 model.lr=5e-5 model.batch_size=1

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
    """按配置构建 FastInst（训练入口与 tools/evaluate.py 共用）。"""
    from model import build_fastinst
    arch = cfg.model.arch
    return build_fastinst(
        num_classes=int(cfg.model.num_classes),
        input_size=int(cfg.protocol.fixed.input_size),
        pretrained_backbone=bool(cfg.model.pretrained),
        arch={
            "hidden_dim": int(arch.get("hidden_dim", 256)),
            "num_object_queries": int(arch.get("num_object_queries", 100)),
            "num_aux_queries": int(arch.get("num_aux_queries", 8)),
            "nheads": int(arch.get("nheads", 8)),
            "dim_feedforward": int(arch.get("dim_feedforward", 1024)),
            "dec_layers": int(arch.get("dec_layers", 3)),
            "pre_norm": bool(arch.get("pre_norm", False)),
            "convs_dim": int(arch.get("convs_dim", 256)),
            "mask_dim": int(arch.get("mask_dim", 256)),
            "train_num_points": int(arch.get("train_num_points", 12544)),
            "oversample_ratio": float(arch.get("oversample_ratio", 3.0)),
            "importance_sample_ratio": float(arch.get("importance_sample_ratio", 0.75)),
            "class_weight": float(arch.get("class_weight", 2.0)),
            "mask_weight": float(arch.get("mask_weight", 5.0)),
            "dice_weight": float(arch.get("dice_weight", 5.0)),
            "cost_location": float(arch.get("cost_location", 1000.0)),
            "proposal_weight": float(arch.get("proposal_weight", 20.0)),
            "no_object_weight": float(arch.get("no_object_weight", 0.1)),
            "test_topk_per_image": int(arch.get("test_topk_per_image", 100)),
        },
    )


class FastInstTrainer(BaseTrainer):
    MODEL_NAME = "fastinst"

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

    # ---------------- 优化器（DETR 族惯例 AdamW + backbone lr×0.1） ----------------
    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        lr = float(self.cfg.model.lr)
        wd = float(self.cfg.model.get("weight_decay", 0.05))
        mult = float(self.cfg.model.get("backbone_lr_mult", 0.1))
        backbone_params = [p for n, p in model.named_parameters()
                           if n.startswith(("stem.", "layer1.", "layer2.",
                                            "layer3.", "layer4."))]
        other_params = [p for n, p in model.named_parameters()
                        if not n.startswith(("stem.", "layer1.", "layer2.",
                                             "layer3.", "layer4."))]
        return torch.optim.AdamW(
            [{"params": backbone_params, "lr": lr * mult},
             {"params": other_params, "lr": lr}],
            lr=lr, weight_decay=wd)

    # ---------------- 训练步 ----------------
    def train_step(self, batch, model, optimizer, scaler) -> dict:
        images, targets = batch
        images = [im.to(self.device, non_blocking=True) for im in images]
        targets = [{k: v.to(self.device, non_blocking=True)
                    for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)   # 已按官方 weight_dict 加权
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
    FastInstTrainer(overrides=sys.argv[1:]).run()


if __name__ == "__main__":
    main()
