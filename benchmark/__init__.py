"""FlashLandslide Benchmark 共享包。

目录约定：
- benchmark/config.py   : 统一配置加载（dataset/protocol/model）
- benchmark/data/       : 共享数据读取与增广（所有模型必须复用）
- benchmark/utils/      : 种子 / 分布式 / 日志
- benchmark/eval/       : 统一评估协议（pycocotools）
- benchmark/runner/     : 训练器抽象基类骨架
"""

__version__ = "0.1.0"
