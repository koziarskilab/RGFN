#!/bin/bash
#SBATCH --job-name=fr_scent_drd2
#SBATCH --time=24:00:00                       # 5000-iter single-shot GFN run: ~10s/iter (like RGFN's 14.6h proxy run) -> ~14h (job 69566 timed out at 8h)
#SBATCH --partition=compute                   # 1-GPU job -> regular partition
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# SCENT (cost-aware, synthesizable) FIXED-REWARD run on sEH — the cost-aware peer in the
# matched four-way fixed-reward comparison. Trains SCENT's cost-guided reaction-GFN ONCE
# against its FROZEN pretrained @TDCProxy(DRD2) (the SAME Bengio-2021 DRD2 oracle the
# RGFN entrant uses), no active-learning loop and NO oracle. No docking -> no OpenCL gate /
# QV2-GPU / boost libs. Runs in the `scent` env (SCENT's package is named `rgfn`, namespace
# clash); candidate emission (with routes -> has_route=1) shells to
# scripts/ingest_candidates.py under the rgfn env (needs CUDA-11.8).
#
# Submit with:  sbatch experiments/fixed_reward/scent_seh/submit_fixed_scent_seh.sh

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

# CUDA-11.8 serves BOTH envs (scent torch/dgl cu118; rgfn glue/dgl cu118 for ingest).
module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

conda activate scent
python validation/generators/scent/run_scent_fixed.py \
        --cfg validation/configs/scent_drd2_fixed.gin \
        --seed 42 \
        --root-dir "$FR_ROOT_DIR"
