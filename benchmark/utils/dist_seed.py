"""种子固定 + PyTorch DDP 初始化的共享实现。"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed: int, rank: int = 0) -> None:
    """全局固定随机种子（按 rank 偏移保证各卡数据增广独立）。"""
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def setup_distributed(backend: str | None = None) -> tuple[int, int, bool]:
    """由 torchrun 环境变量初始化 DDP。返回 (rank, world_size, is_dist)。

    后端优先级：显式参数 > 环境变量 BENCH_DIST_BACKEND > 默认 nccl。
    tools/train.sh 默认选择 gloo，也可用 BENCH_DIST_BACKEND 显式选择后端。

    未设置 RANK 环境变量时按单卡模式运行，便于调试。
    """
    backend = backend or os.environ.get("BENCH_DIST_BACKEND", "nccl")
    if "RANK" not in os.environ:
        return 0, 1, False
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return rank, world, True


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)
