#!/usr/bin/env python
"""Score a checkpoint, saved predictions, or the manuscript benchmark tables.

Run from the repository root with one input mode:

    python tools/evaluate.py --model-name unet --ckpt checkpoint.pth
    python tools/evaluate.py --results predictions.json --ground-truth coco_val.json --by-region
    python tools/evaluate.py --rgb rgb/ --ground-truth coco_val.json --num-classes 3 --report semantic.json
    python tools/evaluate.py --paper all --validation-root validation/ --report class_metrics.csv

Checkpoint, COCO and RGB modes produce JSON reports. COCO mode defaults to mask
evaluation and also accepts --iou-type bbox. RGB mode reads lossless class masks.
Paper mode reads saved predictions, ground-truth caches and run metadata for
main, heldout or all experiments and writes a class-specific CSV. Use cache/<scope>/coco_val.json and results/<job>/ under the validation root.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from benchmark.config import build_full_cfg, load_dataset_cfg, load_protocol_cfg
from benchmark.data.coco_convert import default_cache_path
from benchmark.data.loader_utils import shm_safe_worker_init
from benchmark.data.seg_dataset import SegDataset, img_to_device
from benchmark.eval.coco_eval import (dump_report, evaluate_by_area,
                                      evaluate_by_region, evaluate_coco)
from benchmark.eval.seg_eval import ConfusionAccumulator
from benchmark.eval.paper_metrics import score_instance_files, score_rgb_masks
from benchmark.eval.paper_tables import write_paper_table
from torch.utils.data import DataLoader


def load_train_module(model_name: str):
    """加载 models/{model_name}/train.py（须暴露 PARADIGM / build_model_from_cfg）。"""
    train_py = os.path.join(PROJECT_ROOT, "models", model_name, "train.py")
    if not os.path.isfile(train_py):
        raise FileNotFoundError(f"缺少模型入口: {train_py}")
    spec = importlib.util.spec_from_file_location(
        f"bench_train_{model_name}", train_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_model_from_cfg"):
        raise AttributeError(
            f"{train_py} 未暴露 build_model_from_cfg(cfg)，无法走权重评估模式")
    return mod


def load_weights(model: torch.nn.Module, ckpt: str) -> int:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model", state) if isinstance(state, dict) else state
    # 兼容偶发的 DDP 前缀
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    epoch = state.get("epoch", -1) if isinstance(state, dict) else -1
    if missing:
        print(f"[evaluate][WARN] 权重缺失 {len(missing)} 项: {missing[:5]} ...")
    if unexpected:
        print(f"[evaluate][WARN] 权重多余 {len(unexpected)} 项: {unexpected[:5]} ...")
    return int(epoch)


def eval_semantic(cfg, model, split: str, device: torch.device,
                  batch_size: int | None = None) -> dict:
    """语义口径：val 全量推理 -> 混淆矩阵 -> mIoU/OA/P/R/F1。"""
    cache = cfg.paths.cache_dir
    input_size = int(cfg.protocol.fixed.input_size)
    nw = int(cfg.protocol.fixed.num_workers)
    bs = int(batch_size or cfg.model.val_batch_size)
    val_ds = SegDataset(os.path.join(cache, f"coco_{split}.json"),
                        input_size, train=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True,
                            worker_init_fn=shm_safe_worker_init)

    acc = ConfusionAccumulator(int(cfg.model.num_classes))
    model.eval()
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = img_to_device(imgs, device)
            # 评估一律 fp32：深层激活可能溢出 fp16 上限，导致 logits 变 NaN、
            # argmax 全背景
            with torch.amp.autocast("cuda", enabled=False):
                logits = model(imgs)
            acc.update(logits.argmax(1).cpu(), masks)
    class_names = ["background"] + list(cfg.dataset.classes)
    if int(cfg.model.num_classes) == 3 and class_names == ["background", "landslide"]:
        class_names.append("road")  # Three-output checkpoint on landslide-only ground truth.
    return acc.compute(class_names)


def eval_instance(cfg, model, split: str, device: torch.device,
                  batch_size: int | None, out_json: str, by_region: bool = False) -> dict:
    """实例口径：val 全量推理 -> COCO results -> mAP / by_area 报告。"""
    from benchmark.data.instance_dataset import InstanceDataset, collate_fn
    from benchmark.eval.instance_infer import infer_to_coco

    cache = cfg.paths.cache_dir
    input_size = int(cfg.protocol.fixed.input_size)
    nw = int(cfg.protocol.fixed.num_workers)
    bs = int(batch_size or cfg.model.val_batch_size)
    val_ds = InstanceDataset(os.path.join(cache, f"coco_{split}.json"),
                             input_size, train=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True,
                            collate_fn=collate_fn,
                            worker_init_fn=shm_safe_worker_init)
    conf = float(cfg.protocol.evaluation.conf_threshold)
    infer_to_coco(model, val_loader, out_json, device, conf)

    gt_json = os.path.join(cache, f"coco_{split}.json")
    eval_cfg = cfg.protocol.evaluation
    report = {"overall": evaluate_coco(gt_json, out_json, eval_cfg, "segm"),
              "by_area": evaluate_by_area(gt_json, out_json, eval_cfg, "segm"),
              "results": out_json}
    classes = score_instance_files(gt_json, out_json, by_region)
    report["per_class"] = classes["per_class"]
    if by_region:
        report["class_by_region"] = classes["by_region"]
    return report


def run_ckpt_mode(args) -> None:
    cfg = build_full_cfg(args.model_name, overrides=args.overrides or None)
    mod = load_train_module(args.model_name)
    paradigm = getattr(mod, "PARADIGM", "semantic")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mod.build_model_from_cfg(cfg).to(device)
    epoch = load_weights(model, args.ckpt)
    print(f"[evaluate] model={args.model_name} paradigm={paradigm} "
          f"ckpt={args.ckpt} (epoch={epoch}) device={device}")

    if paradigm == "semantic":
        report = eval_semantic(cfg, model, args.split, device, args.batch_size)
        report["model"] = args.model_name
        report["ckpt"] = args.ckpt
        report["ckpt_epoch"] = epoch
        main_cls = cfg.model.get("eval_class", None)
        print("\n== semantic overall ==")
        print(f"  mIoU={report['mIoU']:.4f} OA={report['OA']:.4f} "
              f"mean_P={report['mean_Precision']:.4f} "
              f"mean_R={report['mean_Recall']:.4f} mean_F1={report['mean_F1']:.4f}")
        print("== per-class ==")
        for name, v in report["per_class"].items():
            mark = "  <- eval_class" if name == main_cls else ""
            print(f"  {name}: IoU={v['IoU']:.4f} P={v['Precision']:.4f} "
                  f"R={v['Recall']:.4f} F1={v['F1']:.4f}{mark}")
        ckpt_stem = os.path.splitext(os.path.basename(args.ckpt))[0]
        out = args.report or os.path.join(
            cfg.paths.work_dir, args.model_name, f"eval_report_{ckpt_stem}.json")
        dump_report(report, out)
        print(f"\n[evaluate] report -> {out}")
    else:
        # instance 范式：推理 COCO results -> 统一 mAP 评估
        ckpt_stem = os.path.splitext(os.path.basename(args.ckpt))[0]
        work = os.path.join(cfg.paths.work_dir, args.model_name)
        results_json = os.path.join(work, f"results_{ckpt_stem}.json")
        report = eval_instance(cfg, model, args.split, device,
                               args.batch_size, results_json, args.by_region)
        report["model"] = args.model_name
        report["ckpt"] = args.ckpt
        report["ckpt_epoch"] = epoch
        report["iou_type"] = "segm"
        ov, ba = report["overall"], report["by_area"]
        print("\n== instance overall (segm) ==")
        print(f"  mAP={ov['mAP']:.4f} mAP_50={ov['mAP_50']:.4f} "
              f"mAP_75={ov['mAP_75']:.4f} AR_100={ov['AR_100']:.4f}")
        print(f"== by_area == small={ba['small']:.4f} "
              f"medium={ba['medium']:.4f} large={ba['large']:.4f}")
        out = args.report or os.path.join(
            work, f"eval_report_{ckpt_stem}.json")
        dump_report(report, out)
        print(f"\n[evaluate] report -> {out}")


def run_results_mode(args) -> None:
    ds_cfg = load_dataset_cfg()
    # 允许命令行覆盖 paths.cache_dir 等（本机评估时指向本地缓存）
    if args.overrides:
        from omegaconf import OmegaConf
        ds_cfg = OmegaConf.merge(ds_cfg, OmegaConf.from_dotlist(args.overrides))
    proto = load_protocol_cfg()
    eval_cfg = proto.evaluation
    gt_json = args.ground_truth or default_cache_path(ds_cfg, args.split)
    if not os.path.isfile(gt_json):
        raise FileNotFoundError(f"缺少 GT: {gt_json}，请先运行 tools/build_coco.py")

    print(f"[evaluate] GT={gt_json}\n[evaluate] results={args.results} iou={args.iou_type}")
    report = {"results": args.results, "iou_type": args.iou_type}

    report["overall"] = evaluate_coco(gt_json, args.results, eval_cfg, args.iou_type)
    print("\n== overall ==", report["overall"])

    report["by_area"] = evaluate_by_area(gt_json, args.results, eval_cfg, args.iou_type)
    print("== by_area ==", report["by_area"])

    if args.by_region:
        report["by_region"] = evaluate_by_region(gt_json, args.results,
                                                 eval_cfg, args.iou_type)
        print("== by_region ==")
        for r, m in report["by_region"].items():
            print(f"  {r}: mAP={m.get('mAP', float('nan')):.3f}")

    if args.iou_type == "segm":
        classes = score_instance_files(gt_json, args.results, args.by_region)
        report["per_class"] = classes["per_class"]
        if args.by_region:
            report["class_by_region"] = classes["by_region"]

    out = args.report or os.path.join(os.path.dirname(args.results),
                                      f"eval_report_{args.iou_type}.json")
    dump_report(report, os.path.abspath(out))
    print(f"[evaluate] report -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Score checkpoints, saved predictions and SFLData manuscript tables.")
    ap.add_argument("--paper", choices=["main", "heldout", "all"], help="Recalculate the manuscript CSV")
    ap.add_argument("--paper-job", help="Limit output to one job, for example main/deeplabv3plus")
    ap.add_argument("--validation-root", help="Directory containing cache/ and results/ for --paper")
    ap.add_argument("--rgb", help="Directory of saved semantic RGB class masks")
    ap.add_argument("--ground-truth", help="COCO annotation JSON for saved RGB or instance predictions")
    ap.add_argument("--num-classes", type=int, choices=[2, 3],
                    help="Semantic model output classes, including background")
    # 模式 A：模型 + 权重
    ap.add_argument("--model-name", default=None,
                    help="模型名（对应 configs/models/{name}.yaml 与 models/{name}/）")
    ap.add_argument("--ckpt", default=None,
                    help="权重路径，如 checkpoint.pth")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="评估 batch（默认取模型配置 val_batch_size）")
    # 模式 B：现成 COCO results
    ap.add_argument("--results", default=None, help="模型输出的 COCO results JSON")
    # 通用
    ap.add_argument("--split", default="val")
    ap.add_argument("--iou-type", default="segm", choices=["segm", "bbox"])
    ap.add_argument("--by-region", action="store_true")
    ap.add_argument("--report", default=None, help="报告输出路径（默认按模式自动命名）")
    ap.add_argument("overrides", nargs="*",
                    help="可选配置覆盖（如 paths.cache_dir=...），本机评估时使用")
    args = ap.parse_args()

    if args.validation_root and not args.paper:
        ap.error("--validation-root requires --paper")
    if args.ground_truth and not (args.rgb or args.results):
        ap.error("--ground-truth requires --rgb or --results")
    if args.num_classes and not args.rgb:
        ap.error("--num-classes requires --rgb")
    if args.report:
        args.report = os.path.abspath(args.report)
    if args.paper_job and not args.paper:
        ap.error("--paper-job requires --paper")
    if args.paper:
        if not args.validation_root or not args.report:
            ap.error("--paper requires --validation-root and --report (output CSV)")
        if args.rgb or args.results or args.ckpt or args.iou_type != "segm":
            ap.error("--paper reads the saved runs under --validation-root")
        write_paper_table(args.paper, args.validation_root, args.report, args.paper_job)
    elif args.rgb:
        if not args.ground_truth or not args.num_classes or not args.report:
            ap.error("--rgb requires --ground-truth, --num-classes and --report")
        if args.ckpt or args.results or args.iou_type != "segm":
            ap.error("--rgb accepts semantic predictions only")
        torch.set_num_threads(2)
        report = score_rgb_masks(args.ground_truth, args.rgb, args.num_classes, args.by_region)
        dump_report(report, args.report)
        print(f"Wrote semantic scores to {args.report}")
    elif args.ckpt:
        if not args.model_name:
            ap.error("--ckpt 模式必须同时指定 --model-name")
        run_ckpt_mode(args)
    elif args.results:
        run_results_mode(args)
    else:
        ap.error("Specify --ckpt (with --model-name), --results, --rgb or --paper.")


if __name__ == "__main__":
    main()
