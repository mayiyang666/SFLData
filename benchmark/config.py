"""统一配置加载：dataset.yaml + protocol.yaml + 模型 config 合并。

约定：模型 config 只允许包含 model_specific 字段与模型结构超参，
数据 / 增广 / 评估协议一律以 dataset.yaml 与 protocol.yaml 为准。

数据集配置切换（跨数据集 / 消融实验必备）:
    默认加载 configs/dataset.yaml。
    设置环境变量 BENCH_DATASET_CFG 可指向其它数据集配置（相对仓库根或绝对路径）:
        BENCH_DATASET_CFG=configs/dataset_no_lixian.yaml python tools/build_coco.py
        BENCH_DATASET_CFG=configs/dataset_compare_ls4s.yaml bash tools/train.sh ...
    cache / work / log 目录随各配置独立，互不覆盖。

路径解析:
    dataset.root 与 paths.* 若写成相对路径（如 ./data/DPT_region），一律相对**仓库根**
    解析，因此可以在任意工作目录下启动训练。也可用 BENCH_DATA_ROOT 环境变量
    覆盖数据根目录，便于换机器时不动配置文件。
"""
from __future__ import annotations

import os

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

DATASET_CFG_ENV = "BENCH_DATASET_CFG"
DATA_ROOT_ENV = "BENCH_DATA_ROOT"


def dataset_cfg_path() -> str:
    """返回本次生效的数据集配置文件路径（支持 BENCH_DATASET_CFG 环境变量）。"""
    env = os.environ.get(DATASET_CFG_ENV)
    if not env:
        return os.path.join(CONFIG_DIR, "dataset.yaml")
    path = env if os.path.isabs(env) else os.path.join(PROJECT_ROOT, env)
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{DATASET_CFG_ENV}={env!r} 指向的数据集配置不存在: {path}"
        )
    return path


def _absolutize(value: str) -> str:
    """把以 . 开头的相对路径按仓库根解析；其它原样返回。"""
    if isinstance(value, str) and value.startswith("."):
        return os.path.normpath(os.path.join(PROJECT_ROOT, value))
    return value


def _resolve_paths(cfg: DictConfig) -> DictConfig:
    """就地解析 root / paths 下的相对路径，并应用 BENCH_DATA_ROOT 覆盖。"""
    if "dataset" in cfg and "root" in cfg.dataset:
        cfg.dataset.root = _absolutize(str(cfg.dataset.root))
    if "paths" in cfg:
        for key in cfg.paths.keys():
            cfg.paths[key] = _absolutize(str(cfg.paths[key]))
    override_root = os.environ.get(DATA_ROOT_ENV)
    if override_root and "dataset" in cfg and "root" in cfg.dataset:
        cfg.dataset.root = os.path.abspath(override_root)
    return cfg


def load_dataset_cfg() -> DictConfig:
    """加载生效的数据集配置（含 dataset / paths / benchmark 三段）。"""
    return _resolve_paths(OmegaConf.load(dataset_cfg_path()))


def load_protocol_cfg() -> DictConfig:
    return OmegaConf.load(os.path.join(CONFIG_DIR, "protocol.yaml"))


def load_model_cfg(model_name: str) -> DictConfig:
    """加载 configs/models/{model_name}.yaml。"""
    path = os.path.join(CONFIG_DIR, "models", f"{model_name}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"缺少模型配置: {path}\n"
            f"请先在 configs/models/ 下创建 {model_name}.yaml。"
        )
    return OmegaConf.load(path)


def build_full_cfg(model_name: str, overrides: list[str] | None = None) -> DictConfig:
    """合并 dataset + protocol + model 配置，并应用命令行 overrides。

    overrides 形如 ["model.lr=1e-4", "model.batch_size=4"]。
    """
    dataset_cfg = load_dataset_cfg()
    cfg = OmegaConf.merge(
        {"dataset": dataset_cfg.dataset,
         "paths": dataset_cfg.paths,
         "benchmark": dataset_cfg.benchmark,
         "protocol": load_protocol_cfg()},
        {"model": load_model_cfg(model_name)},
    )
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    # 校验：模型侧不得覆盖受控字段
    protected = {"dataset", "protocol"}
    for ov in overrides or []:
        key = ov.split("=")[0]
        if any(key == p or key.startswith(p + ".") for p in protected):
            raise ValueError(f"受控配置不允许被模型/命令行覆盖: {ov}")
    return _resolve_paths(cfg)


def model_work_dir(cfg: DictConfig) -> str:
    name = cfg.model.get("name", "unnamed")
    return os.path.join(cfg.paths.work_dir, name)
