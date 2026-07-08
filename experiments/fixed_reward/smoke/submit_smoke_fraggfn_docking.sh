#!/bin/bash
#SBATCH --job-name=smoke_fgfn_dock
#SBATCH --time=01:00:00
#SBATCH --partition=compute
#SBATCH --exclude=balam008
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# SMOKE (3 steps): FragGFN per-step DOCKING via the cross-env bridge — proves the baseline
# DockingBridgeReward shells to `conda run -n rgfn score_batch.py` every training step while
# FragGFN's torch holds the GPU. The parent runs in the `fraggfn` env; it EXPORTS the docking
# libs (boost, GNINA) + loads the cuda module so the inherited `rgfn` subprocess can run
# QuickVina2-GPU. See validation/configs/fraggfn_6td3_docking_smoke.yaml.
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export WANDB_MODE=offline
FR_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$FR_ROOT_DIR"

# cuda module + docking libs must be set in the PARENT so the `conda run -n rgfn` docking
# subprocess (spawned by DockingBridgeReward) inherits them.
module load cuda/11.8.0
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export GNINA=/scratch/markymoo/gnina/run_gnina.sh
export PYTHONUNBUFFERED=1

source /home/markymoo/miniconda3/etc/profile.d/conda.sh
echo "host=$(hostname)"; nvidia-smi -L

HC=$SCRATCH/vina_gpu/opencl_healthcheck
HC_OUT=$(CUDA_VISIBLE_DEVICES=0 "$HC" 2>&1)
grep -q "clCreateContext err=0" <<<"$HC_OUT" || { echo "FATAL: OpenCL dead on $(hostname)"; echo "$HC_OUT"; exit 42; }
echo "OpenCL health OK on $(hostname)"

conda run --no-capture-output -n fraggfn python \
        validation/generators/fraggfn/run_fraggfn_fixed.py \
        --cfg validation/configs/fraggfn_6td3_docking_smoke.yaml \
        --root-dir "$FR_ROOT_DIR"
echo "[smoke_fgfn_dock] done"
