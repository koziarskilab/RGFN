#!/bin/bash
#SBATCH --job-name=fr_rxnflow_stdlib_seh
#SBATCH --time=24:00:00                       # 5000-step single-shot GFN run: ~12s/step -> ~17h
#SBATCH --partition=compute                   # 1-GPU job -> regular partition
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# RxnFlow FIXED-REWARD on sEH — SHARED-LIBRARY variant (Logs/020). Identical to
# submit_fixed_rxnflow_seh.sh EXCEPT RxnFlow uses the shared standard set glue_standard_v1
# baked into data/models/rxnflow_env_stdlib (412 blocks / 112 templates from SCENT's SMALL
# library) instead of its native 10k ZINCFrag / 109 Enamine-REAL env. This is RxnFlow's
# entry in the SAME-LIBRARY four-way sEH benchmark. Trains ONCE against the FROZEN
# Bengio-2021 sEH proxy (same reward as RGFN), routes -> has_route=1. NO loop, NO oracle.
# Candidate emission shells to scripts/ingest_candidates.py under the rgfn env (CUDA-11.8).
# Submit with:  sbatch experiments/fixed_reward/stdlib_seh/submit_fixed_rxnflow_stdlib_seh.sh

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

# CUDA-11.8 serves BOTH envs (rxnflow torch cu121 is self-contained; rgfn glue/dgl cu118 for ingest).
module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

conda activate rxnflow
python validation/generators/rxnflow/run_rxnflow_fixed.py \
        --cfg validation/configs/rxnflow_seh_fixed_stdlib.yaml \
        --seed 42 \
        --root-dir "$FR_ROOT_DIR"
