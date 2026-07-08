#!/bin/bash
#SBATCH --job-name=fr_rgfn_stdlib_seh
#SBATCH --time=3-00:00:00                      # ~54s/iter on the larger glue_standard_v1 (112 rxns vs xlsx 66) -> ~70h for 5002 iters; needs the 3-day cap (job 69613 at 24h would time out ~step 1600). See Logs/020.
#SBATCH --partition=compute                   # 1-GPU job -> regular partition
#SBATCH --gpus-per-node=1
# Absolute $SCRATCH log paths: $HOME is read-only on compute nodes (Logs/012).
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Fixed-reward (single-shot) RGFN on the pretrained sEH proxy — SHARED-LIBRARY variant
# (Logs/020). Identical to submit_fixed_seh_proxy.sh EXCEPT RGFN builds from the canonical
# standard set glue_standard_v1 (data/libraries/glue_standard_v1 = SCENT's SMALL library:
# 418 fragments / 112 templates) instead of upstream data/chemistry.xlsx, via
# GlueReactionDataFactory. This is RGFN's entry in the SAME-LIBRARY four-way sEH benchmark
# (RGFN/SCENT/RxnFlow share glue_standard_v1; FragGFN is the native non-synth foil).
# NO active-learning loop, NO oracle. See configs/glue/fixed_reward_seh_proxy_stdlib.gin.
# Submit with:  sbatch experiments/fixed_reward/stdlib_seh/submit_fixed_rgfn_stdlib_seh.sh

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

# Outputs land under $FR_ROOT_DIR/fixed_reward/seh_proxy_stdlib/<timestamp>/fixed_reward/
# (config stem fixed_reward_seh_proxy_stdlib -> fixed_reward/seh_proxy_stdlib).
python scripts/fixed_reward.py \
        --cfg configs/glue/fixed_reward_seh_proxy_stdlib.gin \
        --seed 42 \
        --root-dir "$FR_ROOT_DIR"
