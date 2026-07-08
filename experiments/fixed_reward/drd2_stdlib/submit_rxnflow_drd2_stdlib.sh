#!/bin/bash
#SBATCH --job-name=fr_rxnflow_drd2_stdlib
#SBATCH --time=24:00:00                       # 5000-step single-shot GFN run: ~12s/step -> ~17h
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# RxnFlow FIXED-REWARD on DRD2 — SHARED-LIBRARY variant. Identical to
# submit_fixed_rxnflow_stdlib_seh.sh but the DRD2 reward + config (glue_standard_v1 via
# data/models/rxnflow_env_stdlib). RxnFlow's entry in the four-way DRD2-on-SMALL benchmark.
# No docking, no loop, no oracle. Candidate emission shells to ingest_candidates.py under rgfn.
# Submit with:  sbatch experiments/fixed_reward/drd2_stdlib/submit_rxnflow_drd2_stdlib.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export WANDB_PROJECT=rgfn WANDB_MODE=offline
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb WANDB_DIR=$SCRATCH/wandb
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch PIP_CACHE_DIR=$SCRATCH/.cache/pip
FR_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$FR_ROOT_DIR"

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

conda activate rxnflow
python validation/generators/rxnflow/run_rxnflow_fixed.py \
        --cfg validation/configs/rxnflow_drd2_fixed_stdlib.yaml \
        --seed 42 --root-dir "$FR_ROOT_DIR"
echo "[fr_rxnflow_drd2_stdlib] done"
