#!/bin/bash
#SBATCH --job-name=fr_seh_proxy
#SBATCH --time=24:00:00                       # 5002-iter single GFN run; ~11.8s/iter measured (job 69514) -> ~16.5h + sampling
#SBATCH --partition=compute                   # 1-GPU job -> regular partition
#SBATCH --gpus-per-node=1
# Absolute $SCRATCH log paths: $HOME is read-only on compute nodes (Logs/012; a
# relative --output dies at job startup).
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Fixed-reward (single-shot) RGFN on the pretrained sEH proxy — the RGFN-paper
# reproduction (test #1) and RGFN's entry in the matched four-way sEH comparison.
# NO active-learning loop, NO oracle: the frozen Bengio-2021 sEH MPNN is the reward
# generator, called every training step. See configs/glue/fixed_reward_seh_proxy.gin.
# Submit with:  sbatch experiments/fixed_reward/seh_proxy/submit_fixed_seh_proxy.sh
#
# Unlike the docking runs this needs no QuickVina2-GPU / OpenCL gate — the reward is
# a fast neural proxy (in-process), so only the dgl CUDA runtime is required.

set -uo pipefail
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

FR_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" \
        "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$FR_ROOT_DIR"

module load cuda/11.8.0                 # dgl graphbolt CUDA-11.8 runtime
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn

export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

# Single-shot fixed-reward run. Outputs (candidate dataset + top_k + timings) land
# under $FR_ROOT_DIR/fixed_reward/seh_proxy/<timestamp>/fixed_reward/.
python scripts/fixed_reward.py \
        --cfg configs/glue/fixed_reward_seh_proxy.gin \
        --seed 42 \
        --root-dir "$FR_ROOT_DIR"
