#!/usr/bin/env bash
# ============================================================
# 统一多卡训练启动模板（所有模型共用同一启动方式）
#
# 用法:
#   bash tools/train.sh <模型入口脚本> [GPU数] [附加参数...]
# 例:
#   bash tools/train.sh models/unet/train.py 4 model.lr=1e-4
# 指定卡号:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash tools/train.sh models/unet/train.py 4
# 换数据集 / 消融配置:
#   BENCH_DATASET_CFG=configs/dataset_no_lixian.yaml bash tools/train.sh models/fastinst/train.py 7
#
# 环境说明:
#   · 若设置了 BENCH_CONDA_ENV，脚本会先激活同名 conda 环境（不设则用当前环境）
#   · 若 conda 不在默认位置，先 export CONDA_BASE=/path/to/miniconda3
# ============================================================
set -e

cd "$(dirname "$0")/.."

ENTRY=${1:?"缺少模型入口脚本，如 models/unet/train.py"}
NGPU=${2:-4}
shift 2 2>/dev/null || shift $#
export TRAIN_GPUS="$NGPU"

# ---------------- 可选：激活 conda 环境 ----------------
if [ -n "${BENCH_CONDA_ENV:-}" ]; then
    CONDA_BASE=${CONDA_BASE:-"$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"}
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$BENCH_CONDA_ENV"
        echo "[train.sh] conda env = $BENCH_CONDA_ENV"
    else
        echo "[train.sh][ERROR] 找不到 $CONDA_BASE/etc/profile.d/conda.sh；"
        echo "                请设置 CONDA_BASE 或直接手工 activate 后去掉 BENCH_CONDA_ENV"
        exit 1
    fi
fi

# 未指定时使用默认卡；若外部已设置 CUDA_VISIBLE_DEVICES 则尊重外部设置
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# The launcher defaults to Gloo. Set BENCH_DIST_BACKEND=nccl for NCCL deployments.
export BENCH_DIST_BACKEND=${BENCH_DIST_BACKEND:-gloo}

# ---------------- CUDA 预检（失败立即终止，避免 torchrun 起一堆进程才报错） ----------------
echo "[train.sh] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  backend=$BENCH_DIST_BACKEND"
python - <<'EOF'
import os, torch
print(f"[preflight] torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
      f"device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    print("[preflight][ERROR] 无可用 CUDA 设备！")
    print(f"[preflight][ERROR] 当前 CUDA_VISIBLE_DEVICES={cvd!r}")
    print("[preflight][HINT] 若该值不是本机卡号，请先执行: unset CUDA_VISIBLE_DEVICES")
    print("[preflight][HINT] 或显式指定空闲卡: CUDA_VISIBLE_DEVICES=0,1 bash tools/train.sh ...")
    raise SystemExit(1)
n_need = int(os.environ.get("TRAIN_GPUS", "1"))
if torch.cuda.device_count() < n_need:
    print(f"[preflight][WARN] 可见卡数 {torch.cuda.device_count()} < 请求 {n_need}，"
          f"将按可见卡数运行")
EOF

torchrun --nproc_per_node="$NGPU" --master_port=${BENCH_MASTER_PORT:-29501} \
    "$ENTRY" "$@"
