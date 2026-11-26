"""统一日志与检查点工具。"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import torch


def get_logger(name: str, log_dir: str | None = None, rank: int = 0) -> logging.Logger:
    """仅 rank0 输出到控制台与文件，其余 rank 静默。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    fmt = logging.Formatter("[%(asctime)s][%(name)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir is not None and rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(
            log_dir, f"train_{datetime.now():%Y%m%d_%H%M%S}.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class MetricLogger:
    """逐 epoch 记录标量指标并追加写入 jsonl（供论文绘图/汇总）。"""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, **kwargs) -> None:
        kwargs["timestamp"] = datetime.now().isoformat(timespec="seconds")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kwargs, ensure_ascii=False) + "\n")


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
