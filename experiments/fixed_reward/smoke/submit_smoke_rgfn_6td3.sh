#!/bin/bash
#SBATCH --job-name=smoke_rgfn_6td3
#SBATCH --time=01:00:00
#SBATCH --partition=compute
#SBATCH --exclude=balam008
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# SMOKE (5 iters): RGFN 6TD3 differential fixed-reward, in-env — proves OracleRewardProxy
# docks every training step without QuickVina2-GPU being starved by torch (empty_cache guard,
# Logs/014). See configs/glue/fixed_reward_6td3_smoke.gin.
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch
FR_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$WANDB_DIR" "$HF_HOME" "$TORCH_HOME" "$FR_ROOT_DIR"

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export GNINA=/scratch/markymoo/gnina/run_gnina.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

HC=$SCRATCH/vina_gpu/opencl_healthcheck
HC_OUT=$(CUDA_VISIBLE_DEVICES=0 "$HC" 2>&1)
grep -q "clCreateContext err=0" <<<"$HC_OUT" || { echo "FATAL: OpenCL dead on $(hostname)"; echo "$HC_OUT"; exit 42; }
echo "OpenCL health OK on $(hostname)"

python scripts/fixed_reward.py \
        --cfg configs/glue/fixed_reward_6td3_smoke.gin \
        --seed 42 --root-dir "$FR_ROOT_DIR"
echo "[smoke_rgfn_6td3] done"
