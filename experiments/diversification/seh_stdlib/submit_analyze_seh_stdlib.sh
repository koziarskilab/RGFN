#!/bin/bash
#SBATCH --job-name=div_seh_stdlib
#SBATCH --time=02:00:00                        # sampling + observed-expander sweep is light; enumerative would need more
#SBATCH --partition=compute                    # 1-GPU job -> regular partition
#SBATCH --gpus-per-node=1
# Absolute $SCRATCH log paths: $HOME is read-only on compute nodes (Logs/012).
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Hub-based late-stage-diversification sweep over a TRAINED sEH-proxy GFN on the priced
# standard library (glue_standard_v1). Produces results.csv (the diversity-vs-concurrency
# /-cost Pareto raw material). See glue/analysis/README.md and this dir's README.
#
# Submit:  CKPT=<path/to/best_gfn.pt> sbatch experiments/diversification/seh_stdlib/submit_analyze_seh_stdlib.sh

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

# The trained checkpoint to analyze (override with:  CKPT=... sbatch ...).
CKPT="${CKPT:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/seh_proxy_stdlib/2026-07-02_14-18-34/train/checkpoints/best_gfn.pt}"

export WANDB_MODE=offline
export TORCH_HOME=$SCRATCH/.cache/torch
export PYTHONUNBUFFERED=1

module load cuda/11.8.0                 # dgl graphbolt CUDA-11.8 runtime
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn

echo "host=$(hostname)"; nvidia-smi -L
echo "checkpoint=$CKPT"

TS=$(date +%Y-%m-%d_%H-%M-%S)
OUT="$SCRATCH/rgfn_runs/experiments/diversification/seh_stdlib/$TS"
mkdir -p "$OUT"

python scripts/analyze_gfn.py \
        --cfg configs/glue/fixed_reward_seh_proxy_stdlib.gin \
        --checkpoint_path "$CKPT" \
        --spec experiments/diversification/seh_stdlib/spec.json \
        --out "$OUT"

echo "results -> $OUT/results.csv"
