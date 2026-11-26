#!/usr/bin/env python
"""MFFENet (ResNet101) benchmark 训练入口（语义分割范式）。

论文: Xu et al., Landslides 2022（MFFENet and ADANet）。模型固有训练设定
按官方代码口径: SGD momentum 0.9、weight_decay 5e-4、骨干 1x lr /
ASPP+解码器 10x lr（官方 train.py）、poly 学习率 power=0.9、
boundary-aware loss（edge_multiplier=1.5, remain_rate=0.5，官方 BALoss）。

启动（从仓库根目录运行）:
    bash tools/train.sh models/mffenet/train.py 4
单卡调试:
    python models/mffenet/train.py
覆盖超参示例:
    bash tools/train.sh models/mffenet/train.py 4 model.lr=5e-4 model.batch_size=1

评估口径: 语义分割（mIoU / OA / per-class IoU / F1 / Precision / Recall），
主评类别 landslide，统一由 benchmark.eval.seg_eval 计算。
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class PolyLR(torch.optim.lr_scheduler.LambdaLR):
    """多项式学习率衰减（论文口径）: lr * (1 - it/max_it)^power。"""

    def __init__(self, optimizer, total_iters: int, power: float = 0.9):
        lam = lambda it: (1.0 - min(it, total_iters) / max(1, total_iters)) ** power
        super().__init__(optimizer, lr_lambda=lam)


class BoundaryAwareLoss(nn.Module):
    """BAL（官方 utils/loss.py BALoss 忠实移植，论文公式 (12)-(17)）。

    边界提取: 2x2 锐化滤波 [[1,-0.5],[-0.5,0]] conv 后取 >=0 得 0/1
    边缘图 E_g；35x35 高斯（sigma=3.0，官方公式）平滑得 G_g 并插值回
    标签尺寸；权重 W = clamp(G_g * edge_multiplier + remain_rate, 0, 4)，
    BAL = sum(W * CE) / (B*H*W)。
    """

    def __init__(self, edge_multiplier: float = 1.5, remain_rate: float = 0.5,
                 gauss_size: int = 35, gauss_sigma: float = 3.0,
                 class_weights: torch.Tensor | None = None,
                 ignore_index: int = 255):
        super().__init__()
        self.edge_multiplier = float(edge_multiplier)
        self.remain_rate = float(remain_rate)
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights, persistent=False)
        # 2x2 空间锐化滤波（官方 edge_filter）
        edge = torch.tensor([[1.0, -0.5], [-0.5, 0.0]]).view(1, 1, 2, 2)
        self.register_buffer("edge_filter", edge, persistent=False)
        # 高斯平滑核（官方 gaussian_kernel 公式: 网格 [-s:s+1],
        # s=int(gauss_size/2)，exp(-(x^2+y^2)/(s*sigma)) 后归一化）
        s = int(gauss_size / 2)
        coords = torch.arange(-s, s + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(coords, coords, indexing="ij")
        g = torch.exp(-(xx ** 2 / (s * gauss_sigma)
                        + yy ** 2 / (s * gauss_sigma)))
        g = g / g.sum()
        self.register_buffer("gauss", g.view(1, 1, 2 * s + 1, 2 * s + 1),
                             persistent=False)

    def _boundary_weight(self, labels: torch.Tensor) -> torch.Tensor:
        """labels: (B,H,W) long -> 逐像素权重 (B,H,W)。

        AMP 下 train_step 处于 autocast 内，F.conv2d 会把输入自动转为 fp16
        而与 fp32 滤波核 buffer 冲突；边界权重计算始终在 fp32 下进行
        （数值稳定，与官方行为一致）。
        """
        with torch.amp.autocast(device_type=labels.device.type, enabled=False):
            t = labels.unsqueeze(1).float()
            # 滤波核 buffer 对齐到标签的设备与 fp32（防止 loss 未随模型上卡）
            edge = self.edge_filter.to(device=t.device, dtype=torch.float32)
            gauss = self.gauss.to(device=t.device, dtype=torch.float32)
            fe = F.conv2d(t, edge, stride=1, padding=0)
            eg = torch.ge(fe, 0.0).float()                      # E_g: 0/1 边缘图
            g = F.conv2d(eg, gauss, stride=1, padding=0)
            g = F.interpolate(g, size=t.shape[2:], mode="bilinear",
                              align_corners=True)
            w = g * self.edge_multiplier + self.remain_rate
            return torch.clamp(w, min=0.0, max=4.0).squeeze(1)  # 公式 (16) 截断

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        w = self._boundary_weight(labels)
        ce = F.cross_entropy(logits, labels, weight=self.class_weights,
                             ignore_index=self.ignore_index, reduction="none")
        return (w.to(ce.dtype) * ce).sum() / (labels.shape[0] * labels.shape[1] * labels.shape[2])


def build_model_from_cfg(cfg) -> nn.Module:
    """按配置构建 MFFENet（训练入口与 tools/evaluate.py 共用）。"""
    from model import MFFENet
    arch = cfg.model.arch
    return MFFENet(num_classes=int(cfg.model.num_classes),
                   backbone=str(cfg.model.get("backbone", "resnet101")),
                   pretrained=bool(cfg.model.pretrained),
                   aspp_rates=tuple(int(r) for r in arch.get(
                       "aspp_rates", (6, 12, 18))),
                   aspp_dropout=float(arch.get("aspp_dropout", 0.5)))


class MFFENetTrainer(BaseTrainer):
    MODEL_NAME = "mffenet"

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

    # ---------------- 优化器（官方: SGD，骨干 1x lr，ASPP+解码器 10x） ----------------
    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        lr = float(self.cfg.model.lr)
        wd = float(self.cfg.model.get("weight_decay", 5e-4))
        mult = float(self.cfg.model.arch.get("head_lr_mult", 10.0))
        if self.cfg.model.optimizer == "sgd":
            backbone_params, head_params = [], []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith(("conv1", "bn1", "layer")):
                    backbone_params.append(p)
                else:
                    head_params.append(p)
            return torch.optim.SGD(
                [{"params": backbone_params, "lr": lr},
                 {"params": head_params, "lr": lr * mult}],
                momentum=float(self.cfg.model.get("momentum", 0.9)),
                weight_decay=wd,
                nesterov=bool(self.cfg.model.get("nesterov", False)))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # ---------------- 调度器（论文: poly, power=0.9；属模型固有项） ----------------
    def build_scheduler(self, optimizer, steps_per_epoch: int):
        sched = self.cfg.model.get("scheduler", {})
        if sched.get("type", "poly") == "poly":
            total = int(self.cfg.protocol.fixed.epochs) * steps_per_epoch
            return PolyLR(optimizer, total, power=float(sched.get("power", 0.9)))
        return super().build_scheduler(optimizer, steps_per_epoch)

    # ---------------- 损失 / 训练步 ----------------
    def _build_loss(self):
        loss_cfg = self.cfg.model.loss
        weights = loss_cfg.get("class_weights", None)
        w = torch.tensor(list(weights), dtype=torch.float32).to(self.device) \
            if weights is not None else None
        if loss_cfg.type == "bal":
            return BoundaryAwareLoss(
                edge_multiplier=float(loss_cfg.get("edge_multiplier", 1.5)),
                remain_rate=float(loss_cfg.get("remain_rate", 0.5)),
                gauss_size=int(loss_cfg.get("gauss_size", 35)),
                gauss_sigma=float(loss_cfg.get("gauss_sigma", 3.0)),
                class_weights=w)
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
        self.loss_fn = self._build_loss().to(self.device)  # loss 不随 DDP 包装，需自行上卡
        super().run()


def main():
    MFFENetTrainer(overrides=sys.argv[1:]).run()


if __name__ == "__main__":
    main()
