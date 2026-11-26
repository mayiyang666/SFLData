"""BaseTrainer：统一训练循环骨架（DDP + AMP + 统一评估）。

模型接入步骤:
    1. 继承 BaseTrainer，实现 build_model / build_dataloader / train_step / inference_to_coco
    2. configs/models/{name}.yaml 中声明模型固有超参
    3. tools/train.sh 启动，评估走 tools/evaluate.py（统一口径）

注意：增广、种子、epoch 预算、评估协议等受控变量在基类中强制读取
protocol.yaml，子类不得覆盖。
"""
from __future__ import annotations

import abc
import glob
import json
import os
import re
import time
from datetime import datetime

import torch
from tqdm import tqdm

from ..config import build_full_cfg, model_work_dir
from ..utils.dist_seed import (cleanup_distributed, is_main_process, set_seed,
                               setup_distributed)
from ..utils.logger import (MetricLogger, get_logger, load_checkpoint,
                            save_checkpoint)


class BaseTrainer(abc.ABC):
    MODEL_NAME: str = "base"
    # 训练中「验证」与「权重快照」的**默认**间隔，二者互相独立，
    # 均可由 configs/protocol.yaml 的 evaluation.interval / ckpt_interval 覆盖。
    #   评估：默认每轮都做（EVAL_INTERVAL=1），能拿到完整指标曲线与最佳 epoch，
    #        代价是每轮多跑一遍 val；
    #   快照：默认每 10 轮（CKPT_INTERVAL=10），用于后期过拟合/提前收敛时回退选优
    #        —— 150 轮每轮都存一份权重很容易撑爆磁盘。
    EVAL_INTERVAL: int = 1
    CKPT_INTERVAL: int = 10

    def __init__(self, overrides: list[str] | None = None):
        self.cfg = build_full_cfg(self.MODEL_NAME, overrides)
        self.rank, self.world_size, self.is_dist = setup_distributed()
        set_seed(self.cfg.protocol.fixed.seed, self.rank)
        self.work_dir = model_work_dir(self.cfg)
        self.logger = get_logger(self.MODEL_NAME, self.cfg.paths.log_dir, self.rank)
        self.metrics = MetricLogger(os.path.join(self.work_dir, "metrics.jsonl")) \
            if is_main_process() else None
        self.device = torch.device(f"cuda:{self.rank % max(1, torch.cuda.device_count())}"
                                   if torch.cuda.is_available() else "cpu")
        # 评估 / 快照节奏（protocol.evaluation，属受控协议，模型侧不可覆盖）
        _ev = self.cfg.protocol.evaluation
        self.eval_interval = max(1, int(_ev.get("interval", self.EVAL_INTERVAL)))
        self.ckpt_interval = max(1, int(_ev.get("ckpt_interval", self.CKPT_INTERVAL)))
        self.keep_results = str(_ev.get("keep_results", "all")).lower()
        if self.keep_results not in {"all", "last"}:
            raise ValueError(
                f"protocol.evaluation.keep_results 只支持 all | last，"
                f"当前为 {self.keep_results!r}")
        self.epoch = 0

    # ---------------- 子类必须实现 ----------------
    @abc.abstractmethod
    def build_model(self) -> torch.nn.Module:
        """构建并返回模型（无需自行包 DDP，基类负责）。"""

    @abc.abstractmethod
    def build_dataloaders(self) -> tuple:
        """返回 (train_loader, val_loader)。训练集必须使用
        benchmark.data.transforms.TrainAugmentation；验证集不加增广。"""

    @abc.abstractmethod
    def train_step(self, batch, model, optimizer, scaler) -> dict:
        """单步训练，返回 {'loss': tensor, ...}。前向/loss 在此实现，
        backward/clip/step 由基类统一执行。"""

    @abc.abstractmethod
    def inference_to_coco(self, model, val_loader, out_json: str) -> None:
        """对验证集推理并输出 COCO results JSON（统一评估输入）。"""

    # ---------------- 可覆盖钩子 ----------------
    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """默认 AdamW；子类可按模型惯例覆盖（属模型固有项）。"""
        lr = float(self.cfg.model.lr)
        wd = float(self.cfg.model.get("weight_decay", 1e-4))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def build_scheduler(self, optimizer, steps_per_epoch: int):
        """默认 cosine 退火；子类可覆盖。"""
        total = self.cfg.protocol.fixed.epochs * steps_per_epoch
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)

    # ---------------- 工具 ----------------
    @staticmethod
    def _fmt_hms(sec: float) -> str:
        """秒 -> Xh XXm XXs 可读格式。"""
        sec = int(max(0, sec))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h{m:02d}m{s:02d}s"

    # ---------------- 断点续训 ----------------
    def _maybe_resume(self, model, optimizer, scheduler, scaler,
                      steps_per_epoch: int) -> int:
        """若 work_dir 存在 last.pth 则恢复状态并从断点继续。

        恢复模型权重、优化器、学习率调度器与 AMP scaler（旧快照无
        scheduler/scaler 字段时按已训轮数重放调度器）。以命令行覆盖
        resume=false 可强制从头训练。返回起始 epoch（0 表示全新训练）。
        """
        if not bool(self.cfg.get("resume", True)):
            return 0
        ckpt_path = os.path.join(self.work_dir, "last.pth")
        if not os.path.isfile(ckpt_path):
            return 0
        state = load_checkpoint(ckpt_path)
        if state.get("model_name", self.MODEL_NAME) != self.MODEL_NAME:
            self.logger.warning(f"last.pth 属于 {state.get('model_name')}，与当前 "
                                f"{self.MODEL_NAME} 不匹配，从头训练")
            return 0
        start_epoch = int(state.get("epoch", 0))
        epochs = int(self.cfg.protocol.fixed.epochs)
        if start_epoch >= epochs:
            self.logger.info(f"last.pth 已完成 {start_epoch}/{epochs} 轮，无需续训")
            return start_epoch
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        else:  # 旧快照兼容：按 step 级调度重放到断点
            # optimizer state_dict 不含 _step_count；若为 0，调度器首次
            # step 会触发 skip-first 告警导致余弦曲线偏移一位
            if getattr(optimizer, "_step_count", 0) == 0:
                optimizer._step_count = start_epoch * steps_per_epoch
            # CosineAnnealingLR.get_lr 为乘性递推，重放起点必须先
            # 归位到 base_lr（快照里的 lr 是保存时刻值），否则曲线偏移
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = base_lr
            for _ in range(start_epoch * steps_per_epoch):
                scheduler.step()
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        self.logger.info(f"resume: 从 {ckpt_path} 恢复，第 {start_epoch + 1} 轮继续")
        return start_epoch

    # ---------------- 统一训练循环 ----------------
    def run(self) -> None:
        model = self.build_model().to(self.device)
        train_loader, val_loader = self.build_dataloaders()
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer, max(1, len(train_loader)))
        use_amp = bool(self.cfg.protocol.fixed.amp)
        # bfloat16 的指数范围可容纳较大的激活值，训练时无需缩放损失。
        amp_dtype = torch.bfloat16
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        clip_norm = float(self.cfg.protocol.fixed.grad_clip_norm)
        start_epoch = self._maybe_resume(model, optimizer, scheduler, scaler,
                                         max(1, len(train_loader)))

        if self.is_dist:
            import torch.nn as nn
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.rank % torch.cuda.device_count()])

        epochs = int(self.cfg.protocol.fixed.epochs)
        t_start_all = time.time()
        ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"[{self.MODEL_NAME}] start: epochs={epochs} "
                         f"(from epoch {start_epoch + 1}), "
                         f"world={self.world_size}, amp={use_amp}(bf16), "
                         f"eval_every={self.eval_interval}ep, "
                         f"ckpt_every={self.ckpt_interval}ep, "
                         f"keep_results={self.keep_results}, "
                         f"backend={os.environ.get('BENCH_DIST_BACKEND', 'nccl')}, "
                         f"at {ts_start}")
        if self.eval_interval == 1:
            self.logger.info(
                f"[{self.MODEL_NAME}] 每轮验证已开启：每轮额外跑一遍 val 推理，"
                "训练总时长会增加（实例分割尤其明显）；"
                "只想定期验证可改 configs/protocol.yaml 的 evaluation.interval")
        for self.epoch in range(start_epoch, epochs):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(self.epoch)
            t0 = time.time()
            model.train()
            running = 0.0
            n_ok = 0
            n_bad_streak = 0
            n_iter = max(1, len(train_loader))
            pbar = tqdm(train_loader,
                        desc=f"[{self.MODEL_NAME}] epoch {self.epoch + 1}/{epochs}",
                        disable=not is_main_process(), dynamic_ncols=True)
            for it, batch in enumerate(pbar):
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype,
                                        enabled=use_amp):
                    out = self.train_step(batch, model, optimizer, scaler)
                loss = out["loss"]
                # NaN 防护：非有限 loss 跳过本步更新；各秩 all-reduce 对齐，
                # 保证 DDP 要么全体跳过要么全体更新，避免死锁或半更新，
                # 也防止 NaN 权重写入 last.pth 后续训空烧
                with torch.no_grad():
                    bad = (~torch.isfinite(loss)).to(self.device)
                    if self.is_dist:
                        torch.distributed.all_reduce(
                            bad, op=torch.distributed.ReduceOp.MAX)
                if bool(bad.item()):
                    n_bad_streak += 1
                    if n_bad_streak <= 3 or n_bad_streak % 200 == 0:
                        alloc = torch.cuda.memory_allocated() / 2**30
                        reserved = torch.cuda.memory_reserved() / 2**30
                        self.logger.warning(
                            f"epoch {self.epoch + 1} it {it}: 非有限 loss，"
                            f"跳过该步更新（连续 {n_bad_streak} 次）"
                            f" GPU显存 alloc={alloc:.1f}G reserved={reserved:.1f}G")
                    if n_bad_streak >= 50:
                        raise RuntimeError(
                            "连续 50 次非有限 loss，训练已不稳定，自动中止"
                            "（可回退最近的干净 ckpt_epoch*.pth 重训）")
                    scheduler.step()
                    continue
                n_bad_streak = 0
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running += float(loss.detach())
                n_ok += 1
                if is_main_process():
                    done = (self.epoch - start_epoch) + (it + 1) / n_iter
                    elapsed = time.time() - t_start_all
                    eta = elapsed / max(done, 1e-6) * (
                        epochs - self.epoch - (it + 1) / n_iter)
                    pbar.set_postfix(loss=f"{running / (it + 1):.4f}",
                                     lr=f"{scheduler.get_last_lr()[0]:.2e}",
                                     mem=f"{torch.cuda.memory_allocated() / 2**30:.1f}G",
                                     eta=self._fmt_hms(eta))
            dt = time.time() - t0

            if is_main_process():
                avg = running / max(n_ok, 1)
                trained = self.epoch + 1 - start_epoch
                remain = self._fmt_hms(
                    (time.time() - t_start_all) / max(trained, 1)
                    * (epochs - self.epoch - 1))
                self.logger.info(f"epoch {self.epoch + 1}/{epochs} "
                                 f"loss={avg:.4f} time={dt:.1f}s remain~{remain}")
                self.metrics.log(epoch=self.epoch + 1, train_loss=avg,
                                 epoch_time_s=round(dt, 1))
                model_sd = (model.module if self.is_dist else model).state_dict()
                weights_ok = all(torch.isfinite(v).all()
                                 for v in model_sd.values()
                                 if v.is_floating_point())
                if not weights_ok:
                    self.logger.error(
                        f"epoch {self.epoch + 1}: 权重含 NaN/Inf，跳过 "
                        "last.pth 与周期快照保存（保留上一干净断点）")
                else:
                    state = {"model": model_sd,
                             "optimizer": optimizer.state_dict(),
                             "scheduler": scheduler.state_dict(),
                             "scaler": scaler.state_dict(),
                             "epoch": self.epoch + 1,
                             "model_name": self.MODEL_NAME}
                    save_checkpoint(state, os.path.join(self.work_dir, "last.pth"))
                    # 周期性权重快照（每 ckpt_interval 轮 + 末轮，与评估节奏解耦）：
                    # 仅存模型权重，便于按评估报告回退选优，避免只留 last.pth
                    if ((self.epoch + 1) % self.ckpt_interval == 0
                            or self.epoch + 1 == epochs):
                        ckpt = os.path.join(self.work_dir,
                                            f"ckpt_epoch{self.epoch + 1}.pth")
                        save_checkpoint(
                            {"model": state["model"], "epoch": self.epoch + 1,
                             "model_name": self.MODEL_NAME}, ckpt)
                        self.logger.info(f"epoch {self.epoch + 1}: checkpoint -> {ckpt}")

            # 统一评估（默认每轮 + 末轮；由 protocol.evaluation.interval 控制）
            if is_main_process() and ((self.epoch + 1) % self.eval_interval == 0
                                      or self.epoch + 1 == epochs):
                self._evaluate(model, val_loader)
                self._prune_stale_results()

        total = time.time() - t_start_all
        if is_main_process():
            summary = (f"[{self.MODEL_NAME}] FINISHED: {epochs} epochs, "
                       f"start={ts_start}, end={datetime.now():%Y-%m-%d %H:%M:%S}, "
                       f"total={self._fmt_hms(total)}, "
                       f"avg={self._fmt_hms(total / max(1, epochs))}/epoch")
            self.logger.info(summary)
            self.metrics.log(summary=True, start=ts_start,
                             end=datetime.now().isoformat(timespec="seconds"),
                             total_time_s=round(total, 1), epochs=epochs)
        cleanup_distributed()

    def _prune_stale_results(self) -> None:
        """按 protocol.evaluation.keep_results 清理历史 COCO results。

        - `all`（默认）：全部保留，不做任何删除
        - `last`：只保留最近一份（末轮那份），其余删除

        为什么需要：实例分割的 `results_epoch{N}.json` 是 val 全量预测（含 RLE
        mask），单份可达上百 MB 到 GB 级；改成每轮评估后，150 轮全留极易占满磁盘，
        而真正需要复评的通常只有表现最好的那几轮 + 末轮。
        """
        if self.keep_results != "last":
            return
        pattern = re.compile(r"epoch(\d+)\.json$")
        paths = []
        for p in glob.glob(os.path.join(self.work_dir, "results_epoch*.json")):
            m = pattern.search(p)
            if m:
                paths.append((int(m.group(1)), p))
        if len(paths) <= 1:
            return
        paths.sort()
        for _, p in paths[:-1]:
            try:
                os.remove(p)
            except OSError as exc:                       # noqa: BLE001
                self.logger.warning(f"清理旧 results 失败: {p} ({exc})")

    def _evaluate(self, model, val_loader) -> None:
        """推理 -> COCO results -> 统一评估脚本（避免训练进程内评估阻塞）。"""
        results_json = os.path.join(self.work_dir, f"results_epoch{self.epoch + 1}.json")
        self.logger.info(f"epoch {self.epoch + 1}: inference -> {results_json}")
        model.eval()
        with torch.no_grad():
            self.inference_to_coco(model, val_loader, results_json)
        self.logger.info("run tools/evaluate.py for unified metrics")
