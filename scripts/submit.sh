#!/bin/bash
#SBATCH --job-name=rgfn_test
#SBATCH --time=20:00:00                       # walltime, raise/lower as needed
#SBATCH --partition=compute                   # see partition options with `sinfo`
#SBATCH --gpus-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

# --- Caches: redirect everything away from $HOME ----------------------------
export WANDB_PROJECT=rgfn
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export WANDB_CONFIG_DIR=$SCRATCH/.config/wandb
export WANDB_DATA_DIR=$SCRATCH/.cache/wandb
export WANDB_DIR=$SCRATCH/wandb
export WANDB_MODE=offline
export HF_HOME=$SCRATCH/.cache/huggingface
export TORCH_HOME=$SCRATCH/.cache/torch
export PIP_CACHE_DIR=$SCRATCH/.cache/pip

mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" \
        "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"

module load cuda/11.8.0                 # provides nvcc + CUDA runtime libs (matches torch/dgl cu118 build)
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn

# Vina-GPU-2.1 docking oracle: built/installed under $SCRATCH because $HOME is
# read-only on Balam compute nodes and the OpenCL kernel cache must be writable.
# (quickvina_dir symlink -> $SCRATCH/vina_gpu/Vina-GPU-2.1)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$SCRATCH/vina_gpu/boost/lib"

# Optional: nice-to-haves
export PYTHONUNBUFFERED=1
nvidia-smi

# Use scripts/train.py so any glue/ components are registered with gin first.
# (Harmless for plain upstream configs like this one.)
python scripts/train.py --cfg configs/rgfn_seh_docking.gin
