#!/usr/bin/env python
"""Train YOLO26l-seg using the shared benchmark protocol and model configuration."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.config import build_full_cfg
from protocol import (build_protocol_seg_trainer, prepare_yolo_dataset)

# 评估范式：instance=实例分割（COCO mAP，经 predict.py 出 results 后统一评估）
PARADIGM = "instance"

# ultralytics 认的优化器名（含本分支特有的 MuSGD）；另一侧是小写别名
_OPT_CANON = {
    "sgd": "SGD", "adam": "Adam", "adamw": "AdamW", "nadam": "NAdam",
    "radam": "RAdam", "rmsprop": "RMSProp", "muon": "MuSGD", "musgd": "MuSGD",
    "auto": "auto",
}


def resolve_optimizer(raw: str, allow_auto: bool) -> str:
    """Normalise the optimizer name and require explicit opt-in to automatic selection."""
    key = str(raw).strip().lower()
    if key not in _OPT_CANON:
        raise SystemExit(
            f"[yolo26l_seg] 不认识的 optimizer={raw!r}；可选："
            f"{sorted(set(_OPT_CANON.values()))}")
    canon = _OPT_CANON[key]
    if canon == "auto" and not allow_auto:
        raise SystemExit(
            "Choose an explicit optimizer, or set model.allow_auto_optimizer=True.")
    return canon


def build_model_from_cfg(cfg):
    """按配置构建 YOLO26l-seg 模型（ultralytics YOLO 封装，非 nn.Module）。"""
    from ultralytics import YOLO
    return YOLO(str(cfg.model.get("weights", "yolo26l-seg.pt")))


def main():
    cfg = build_full_cfg("yolo26l_seg", overrides=sys.argv[1:])

    # ---- 数据视图（symlink 树 + damage 行剔除 + dataset.yaml）----
    info = prepare_yolo_dataset(cfg)
    print(f"[yolo26l_seg] 数据视图: train={info['train']} val={info['val']} "
          f"过滤 damage 标签 {info['filtered_labels']} 个文件 / "
          f"{info['filtered_lines']} 行 -> {info['data_yaml']}")

    # ---- 协议受控项（来自 protocol.yaml，禁止模型侧覆盖）----
    fixed = cfg.protocol.fixed
    aug = fixed.augmentation
    hsv_s = float(aug.photometric.saturation)
    hsv_v = float(aug.photometric.brightness)   # ultralytics HSV 无 contrast 项，
                                                # 亮度取 brightness 幅度近似
    trainer_cls = build_protocol_seg_trainer(
        p_hflip=float(aug.random_horizontal_flip),
        p_vflip=float(aug.random_vertical_flip),
        p_rot90=float(aug.random_rotate90),
        hsv_s=hsv_s, hsv_v=hsv_v,
        grad_clip_norm=float(fixed.grad_clip_norm),
        ddp_backend=str(cfg.model.get("ddp_backend", "gloo")))

    # ---- 模型固有项（configs/models/yolo26l_seg.yaml）----
    run_name = str(cfg.model.get("name", "yolo26l_seg"))
    optimizer = resolve_optimizer(
        cfg.model.get("optimizer", "auto"),
        allow_auto=bool(cfg.model.get("allow_auto_optimizer", False)))
    ddp_backend = str(cfg.model.get("ddp_backend", "gloo")).strip().lower()
    if ddp_backend not in {"gloo", "nccl"}:
        raise SystemExit(
            f"[yolo26l_seg] 不认识的 ddp_backend={ddp_backend!r}；可选 gloo | nccl。\n"
            "  默认 gloo：本 benchmark 实验环境的 NCCL 通信域初始化会段错误\n"
            "  （连 world=1 单卡都 SIGSEGV，挂在 ncclCommInitRankConfig）。\n"
            "  换到 NCCL 正常的机器后可改 model.ddp_backend=nccl。")

    train_args = dict(
        data=info["data_yaml"],
        trainer=trainer_cls,
        # 受控项注入
        seed=int(fixed.seed),
        imgsz=int(fixed.input_size),
        epochs=int(fixed.epochs),
        amp=bool(fixed.amp),
        workers=int(fixed.num_workers),
        # 增广对齐：协议外增广全部显式关闭（自定义链虽不读取，
        # 显式置零保证 args.json 记录与 val 回退路径一致）
        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        hsv_h=0.0, hsv_s=hsv_s, hsv_v=hsv_v,
        fliplr=float(aug.random_horizontal_flip),
        flipud=float(aug.random_vertical_flip),
        bgr=0.0, erasing=0.0, close_mosaic=0,
        # 固定训练预算：禁用早停
        patience=int(cfg.model.get("patience", 1000)),
        # 产物落 benchmark work_dirs（run_name 决定目录，重训不覆盖旧 run）
        project=str(cfg.paths.work_dir),
        name=run_name,
        exist_ok=True,
        save_period=int(cfg.model.get("save_period", 10)),
        save=True,
        # 模型固有项
        batch=int(cfg.model.batch_size),
        device=str(cfg.model.get("device", "0,1,2")),
        optimizer=optimizer,
    )
    if cfg.model.get("lr", None) is not None:
        train_args["lr0"] = float(cfg.model.lr)
    if cfg.model.get("momentum", None) is not None:
        train_args["momentum"] = float(cfg.model.momentum)
    if cfg.model.get("weight_decay", None) is not None:
        train_args["weight_decay"] = float(cfg.model.weight_decay)
    if bool(cfg.model.get("resume", False)):
        train_args["resume"] = True
    # 仅供冒烟/调试：把训练集按比例截断以缩短一次 epoch。
    # 默认不设 → ultralytics 用 1.0（全量），正式训练不受影响。
    # 用法：model.debug_fraction=0.02
    if cfg.model.get("debug_fraction", None) is not None:
        frac = float(cfg.model.debug_fraction)
        if not 0.0 < frac <= 1.0:
            raise SystemExit("[yolo26l_seg] debug_fraction 必须在 (0, 1] 内，"
                             "实得 %r" % frac)
        train_args["fraction"] = frac
        print("[yolo26l_seg] !! 调试模式：debug_fraction=%.4f（仅用 %.1f%% 训练集，"
              "产出的指标无意义）" % (frac, frac * 100))

    model = build_model_from_cfg(cfg)
    print(f"[yolo26l_seg] run名={run_name} 受控项: seed={train_args['seed']} "
          f"imgsz={train_args['imgsz']} epochs={train_args['epochs']} "
          f"amp={train_args['amp']} workers={train_args['workers']} "
          f"grad_clip={float(fixed.grad_clip_norm)} "
          f"batch(全局)={train_args['batch']} device={train_args['device']}")
    print(f"[yolo26l_seg] 优化器={train_args['optimizer']} "
          f"lr0={train_args.get('lr0', 'auto(未指定)')} "
          f"momentum={train_args.get('momentum', '默认')} "
          f"weight_decay={train_args.get('weight_decay', '默认')}")
    print(f"[yolo26l_seg] DDP 后端={ddp_backend}"
          + ("（gloo，速度略慢于 NCCL）"
             if ddp_backend == "gloo" and len(str(train_args["device"]).split(",")) > 1
             else ""))
    print(f"[yolo26l_seg] 产物目录: {os.path.join(str(cfg.paths.work_dir), run_name)}")
    print("[yolo26l_seg] 启动后请核对日志里 'optimizer:' 那一行与上面一致 —— "
          "若出现 MuSGD 说明走错了配方，立即停止。")
    model.train(**train_args)


if __name__ == "__main__":
    main()
