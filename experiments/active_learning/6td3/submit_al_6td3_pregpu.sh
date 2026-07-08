#!/bin/bash
#SBATCH --job-name=al_6td3_pregpu
#SBATCH --time=08:00:00                       # ~5 h like the mini run; headroom for 3 dock rounds
#SBATCH --partition=compute                   # 1-GPU job -> the regular (non full_node) partition
#SBATCH --gpus-per-node=1
# Absolute paths on $SCRATCH: $HOME is read-only on compute nodes, so relative
# %x-%j.out (resolved against the submit cwd) fails to open and SLURM kills the
# job at startup (job 69450 died this way in 4 s). Hardcode a writable dir so the
# log location no longer depends on where sbatch was invoked.
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# "Pre-GPU-exploration" baseline (Logs/012): the SAME 3-round 6TD3 active-learning
# loop as submit_al_6td3_mini.sh (job 69445, Logs/011), re-run now that the docking
# oracle is instrumented at the sub-step level. The loop turns on
# Docking6TD3Oracle.enable_step_timing(), so this run writes
#   <run>/active_learning/docking_timings.csv  (round, step, seconds, n_molecules)
# breaking oracle_score into embed / tier2_dock / pose_select / tier1_rescore.
# Purpose: measure where the docking phase's time actually goes BEFORE we try to
# swap the Tier-2 search (step 2) for a GPU docker. Config unchanged so this is a
# like-for-like baseline against Logs/011.
#
# Submit with:  sbatch experiments/active_learning/6td3/submit_al_6td3_pregpu.sh

set -euo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

# --- Caches + run outputs: redirect everything away from the read-only $HOME ---
export WANDB_PROJECT=rgfn
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export WANDB_CONFIG_DIR=$SCRATCH/.config/wandb
export WANDB_DATA_DIR=$SCRATCH/.cache/wandb
export WANDB_DIR=$SCRATCH/wandb
export WANDB_MODE=offline
export HF_HOME=$SCRATCH/.cache/huggingface
export TORCH_HOME=$SCRATCH/.cache/torch
export PIP_CACHE_DIR=$SCRATCH/.cache/pip

AL_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" \
        "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$AL_ROOT_DIR"

module load cuda/11.8.0                 # CUDA-11.8 runtime libs (matches torch/dgl cu118; dgl graphbolt needs it)
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn

# gnina docking oracle launcher (oracle defaults to this path via $GNINA).
export GNINA=/scratch/markymoo/gnina/run_gnina.sh

export PYTHONUNBUFFERED=1
nvidia-smi

python scripts/active_learning.py \
        --cfg configs/glue/active_learning_6td3_mini.gin \
        --seed 42 \
        --root-dir "$AL_ROOT_DIR"
