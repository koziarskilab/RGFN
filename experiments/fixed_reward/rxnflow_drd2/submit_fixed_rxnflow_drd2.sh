#!/bin/bash
#SBATCH --job-name=fr_rxnflow_drd2
#SBATCH --time=24:00:00                       # 5000-step single-shot GFN run: ~12s/step at this scale -> ~17h (job 69565 timed out at 8h/step2300)
#SBATCH --partition=compute                   # 1-GPU job -> regular partition
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# RxnFlow (synthesizable) FIXED-REWARD run on sEH — the synthesizable peer in the matched
# four-way fixed-reward comparison. Trains RxnFlow's reaction-template + building-block
# synthesis GFN ONCE against the FROZEN pretrained Bengio-2021 DRD2 oracle (the SAME reward
# generator as the RGFN entrant), no active-learning loop and NO oracle. No docking ->
# no OpenCL gate / QV2-GPU / boost libs. Candidate emission (with synthesis routes ->
# has_route=1) shells to scripts/ingest_candidates.py under the rgfn env (needs CUDA-11.8).
#
# Submit with:  sbatch experiments/fixed_reward/rxnflow_seh/submit_fixed_rxnflow_seh.sh

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
        --cfg validation/configs/rxnflow_drd2_fixed.yaml \
        --seed 42 \
        --root-dir "$FR_ROOT_DIR"
