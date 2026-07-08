#!/bin/bash
#SBATCH --job-name=fr_rgfn_6td3
#SBATCH --time=20:00:00                        # 6TD3 differential docking-in-loop: ~10h at 400 iters (benchmark 69691); slack for larger mols
#SBATCH --partition=compute
#SBATCH --exclude=balam008                     # balam008 OpenCL wedged (Logs/013/014)
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Fixed-reward (single-shot) RGFN on the 6TD3 two-tier docking DIFFERENTIAL, built from the
# shared SMALL library (glue_standard_v1). OracleRewardProxy(Docking6TD3GpuOracle) is the
# per-step reward; empty_cache before each dock keeps QuickVina2-GPU alive during training.
# See configs/glue/fixed_reward_6td3.gin. Submit: sbatch experiments/fixed_reward/6td3/submit_fixed_6td3.sh
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
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export GNINA=/scratch/markymoo/gnina/run_gnina.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

HC=$SCRATCH/vina_gpu/opencl_healthcheck
HC_OUT=$(CUDA_VISIBLE_DEVICES=0 "$HC" 2>&1)
grep -q "clCreateContext err=0" <<<"$HC_OUT" || { echo "FATAL: OpenCL dead on $(hostname)"; echo "$HC_OUT"; exit 42; }
echo "OpenCL health OK on $(hostname)"

python scripts/fixed_reward.py \
        --cfg configs/glue/fixed_reward_6td3.gin \
        --seed 42 --root-dir "$FR_ROOT_DIR"
echo "[fr_rgfn_6td3] done"
