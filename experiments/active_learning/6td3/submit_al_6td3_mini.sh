#!/bin/bash
#SBATCH --job-name=al_6td3_mini
#SBATCH --time=08:00:00                       # mini run is ~3-5 h; headroom for the 3 dock rounds
#SBATCH --partition=compute                   # 1-GPU job -> the regular (non full_node) partition
#SBATCH --gpus-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Multi-round 6TD3 active-learning loop (3 rounds) on a Balam COMPUTE node.
# This is the compute-node re-run staged in Logs/009: unlike the login node, a
# compute node has no per-process CPU-time rlimit, so the per-round gnina docking
# step won't be SIGXCPU-killed. See configs/glue/active_learning_6td3_mini.gin.
#
# Submit with:  sbatch experiments/active_learning/6td3/submit_al_6td3_mini.sh

set -euo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

# --- Caches + run outputs: redirect everything away from the read-only $HOME ---
# On a compute node $HOME is read-only, so run dirs / per-round dataset CSVs /
# checkpoints / wandb must land on $SCRATCH (passed via --root-dir below).
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

# scripts/active_learning.py imports glue/ first so gin can resolve our oracle/
# proxy/dataset/loop. --root-dir redirects run outputs onto $SCRATCH.
python scripts/active_learning.py \
        --cfg configs/glue/active_learning_6td3_mini.gin \
        --seed 42 \
        --root-dir "$AL_ROOT_DIR"
