#!/bin/bash
#SBATCH --job-name=fr_fgfn_drd2_uni
#SBATCH --time=08:00:00                       # single-shot 5000-step GFN run (no docking); AL run was 5h incl. 5 dock rounds
#SBATCH --partition=compute                   # 1-GPU job -> regular partition
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# FragGFN (non-synthesizable) FIXED-REWARD run on sEH — the foil in the matched four-way
# fixed-reward comparison. Trains Recursion's fragment-GFN ONCE against the FROZEN
# pretrained Bengio-2021 DRD2 oracle (the SAME reward generator as the RGFN entrant), no
# active-learning loop and NO oracle. So — unlike the AL baseline — there is NO docking:
# no OpenCL health gate, no QuickVina2-GPU / gnina / boost libs. The only cross-env step is
# candidate emission, which shells to scripts/ingest_candidates.py under the rgfn env
# (imports glue -> dgl -> needs the CUDA-11.8 module, loaded below).
#
# Submit with:  sbatch experiments/fixed_reward/fraggfn_seh/submit_fixed_fraggfn_seh.sh

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

# CUDA-11.8 serves BOTH envs (fraggfn torch cu118; rgfn glue/dgl cu118 for the ingest step).
module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

conda activate fraggfn
python validation/generators/fraggfn/run_fraggfn_fixed.py \
        --cfg validation/configs/fraggfn_drd2_uniform.yaml \
        --seed 42 \
        --root-dir "$FR_ROOT_DIR"
