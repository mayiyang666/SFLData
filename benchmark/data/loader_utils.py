"""DataLoader worker 工具：共享内存（/dev/shm）泄漏防护。

多进程加载时，定期回收循环引用可缩短已用批次张量占用共享内存的时间。

用法:
    DataLoader(..., worker_init_fn=shm_safe_worker_init)
"""
from __future__ import annotations

import gc
import threading
import time

__all__ = ["shm_safe_worker_init"]


def _gc_loop(interval: float) -> None:
    while True:
        time.sleep(interval)
        gc.collect()


def shm_safe_worker_init(worker_id: int, interval: float = 2.0) -> None:
    """DataLoader worker_init_fn：为每个 worker 启动后台守护线程，
    每 `interval` 秒执行一次 gc.collect()，回收循环引用的批次张量，
    防止 /dev/shm 随迭代线性增长。线程为 daemon，随 worker 退出。"""
    threading.Thread(target=_gc_loop, args=(interval,), daemon=True).start()
