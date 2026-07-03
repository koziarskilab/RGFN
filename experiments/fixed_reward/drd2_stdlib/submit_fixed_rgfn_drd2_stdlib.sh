#!/bin/bash
#SBATCH --job-name=fr_rgfn_drd2_stdlib
#SBATCH --time=3-00:00:00                       # glue_standard_v1 (112 rxns) ~54s/iter x 5002 ~= 70h; 3-day cap (Logs/020)
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Fixed-reward (single-shot) RGFN on the DRD2 proxy — SHARED-library variant (glue_standard_v1),
# RGFN's entry in the four-way DRD2-on-SMALL comparison. Frozen DRD2 TDC oracle, beta=48, no
# docking. See configs/glue/fixed_reward_drd2_stdlib.gin. Mirrors submit_fixed_rgfn_stdlib_seh.sh.
# Submit: sbatch experiments/fixed_reward/drd2_stdlib/submit_fixed_rgfn_drd2_stdlib.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export WANDB_PROJECT=rgfn WANDB_MODE=offline
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb WANDB_DIR=$SCRATCH/wandb
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch PIP_CACHE_DIR=$SCRATCH/.cache/pip
FR_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$FR_ROOT_DIR"

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

python scripts/fixed_reward.py \
        --cfg configs/glue/fixed_reward_drd2_stdlib.gin \
        --seed 42 --root-dir "$FR_ROOT_DIR"
echo "[fr_rgfn_drd2_stdlib] done"
