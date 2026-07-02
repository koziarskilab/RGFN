#!/bin/bash
#SBATCH --job-name=fr_drd2
#SBATCH --time=24:00:00                       # reaction-env single-shot ~5002 iters (~14-16h, like the sEH-proxy run); DRD2 oracle is fast
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Fixed-reward (single-shot) RGFN on the DRD2 surrogate reward generator — the RGFN paper's
# third proxy benchmark (`[koziarski2024rgfn]` §4.1). Trains the GFN ONCE against the
# Therapeutics Data Commons DRD2 oracle (tdc.Oracle("DRD2"); DRD2Proxy), no active-learning
# loop and NO oracle-refit. The DRD2 oracle is a fast sklearn+ECFP model (called every step,
# in-process) — no docking, so no OpenCL gate / QV2-GPU / boost libs.
# See configs/glue/fixed_reward_drd2.gin.
#
# Submit with:  sbatch experiments/fixed_reward/drd2/submit_fixed_drd2.sh

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

export WANDB_PROJECT=rgfn
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export WANDB_CONFIG_DIR=$SCRATCH/.config/wandb
export WANDB_DATA_DIR=$SCRATCH/.cache/wandb
export WANDB_DIR=$SCRATCH/wandb
export WANDB_MODE=offline
export HF_HOME=$SCRATCH/.cache/huggingface
export TORCH_HOME=$SCRATCH/.cache/torch
export PIP_CACHE_DIR=$SCRATCH/.cache/pip

FR_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" \
        "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$FR_ROOT_DIR"

module load cuda/11.8.0                 # dgl graphbolt CUDA-11.8 runtime
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn

export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

# DRD2Proxy uses tdc.Oracle("DRD2"), which reads the cached model at ./oracle/drd2_current.pkl
# (pre-downloaded on the login node; $HOME is read-only on compute nodes so it must NOT need
# to re-download). Fail early with a clear message if it is missing.
if [ ! -f oracle/drd2_current.pkl ]; then
    echo "FATAL: oracle/drd2_current.pkl missing. Pre-download on the login node:"
    echo "       conda run -n rgfn python -c \"from tdc import Oracle; Oracle(name='DRD2')\""
    exit 44
fi

python scripts/fixed_reward.py \
        --cfg configs/glue/fixed_reward_drd2.gin \
        --seed 42 \
        --root-dir "$FR_ROOT_DIR"
