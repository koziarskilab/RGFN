#!/bin/bash
#SBATCH --job-name=fr_seh_docking
#SBATCH --time=20:00:00                       # docking-in-the-loop: ~160s/iter x 400 iters ~= 18h (Patch #3 cap)
#SBATCH --partition=compute                   # 1-GPU job -> regular partition
#SBATCH --exclude=balam008                     # balam008 OpenCL wedged (Logs/013/014); QV2-GPU yields all-no_pose there
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Fixed-reward (single-shot) RGFN on sEH GPU DOCKING — the RGFN paper's second sEH
# reproduction (test #2, RGFN-only): QuickVina2-GPU docking called DIRECTLY as the reward
# every training step (the paper's "docking directly in the training loop eliminates proxy
# generalization failure"). Unlike the sEH-proxy runs this DOES dock, so it needs the
# QuickVina2-GPU boost libs + the OpenCL health gate. See
# configs/glue/fixed_reward_seh_docking.gin (includes configs/rgfn_seh_docking.gin: beta=4,
# Trainer.n_iterations=400).
#
# Submit with:  sbatch experiments/fixed_reward/seh_docking/submit_fixed_seh_docking.sh

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

module load cuda/11.8.0                 # dgl graphbolt AND QuickVina2-GPU OpenCL runtime
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn

# QuickVina2-GPU-2.1 boost runtime libs (build lives on $SCRATCH; quickvina_dir symlink
# -> $SCRATCH/vina_gpu/Vina-GPU-2.1).
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

# --- OpenCL health gate: prove QuickVina2-GPU can create a context on THIS node. ---
HC=$SCRATCH/vina_gpu/opencl_healthcheck
if [ ! -x "$HC" ]; then
    echo "FATAL: healthcheck binary $HC missing -- build it on the login node (see Logs/013)."
    exit 43
fi
HC_OUT=$(CUDA_VISIBLE_DEVICES=0 "$HC" 2>&1)
if ! grep -q "clCreateContext err=0" <<<"$HC_OUT"; then
    echo "FATAL: NVIDIA OpenCL clCreateContext FAILS on $(hostname) -- bad node."
    echo "$HC_OUT"
    echo "       Add '$(hostname)' to the #SBATCH --exclude list and resubmit."
    exit 42
fi
echo "OpenCL health OK on $(hostname)"

python scripts/fixed_reward.py \
        --cfg configs/glue/fixed_reward_seh_docking.gin \
        --seed 42 \
        --root-dir "$FR_ROOT_DIR"
