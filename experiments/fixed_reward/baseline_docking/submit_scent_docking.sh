#!/bin/bash
#SBATCH --job-name=fr_scent_dock
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --exclude=balam008
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# SCENT per-step DOCKING fixed-reward (cost-aware synthesizable peer, SHARED SMALL library).
# Config as $1 (a gin file):
#   sbatch experiments/fixed_reward/baseline_docking/submit_scent_docking.sh validation/configs/scent_6td3_fixed.gin
#   sbatch experiments/fixed_reward/baseline_docking/submit_scent_docking.sh validation/configs/scent_clpp_fixed.gin
# Runs in the `scent` env; exports docking libs + cuda module for the per-step rgfn subprocess
# (@DockingBridgeProxy). SCENT walltime slightly higher (cost-guidance overhead per iter).
set -uo pipefail
CFG=${1:?usage: submit_scent_docking.sh <validation/configs/*.gin>}
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export WANDB_MODE=offline
FR_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$FR_ROOT_DIR"

module load cuda/11.8.0
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export GNINA=/scratch/markymoo/gnina/run_gnina.sh
export PYTHONUNBUFFERED=1
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
echo "host=$(hostname) cfg=$CFG"; nvidia-smi -L

HC=$SCRATCH/vina_gpu/opencl_healthcheck
HC_OUT=$(CUDA_VISIBLE_DEVICES=0 "$HC" 2>&1)
grep -q "clCreateContext err=0" <<<"$HC_OUT" || { echo "FATAL: OpenCL dead on $(hostname)"; echo "$HC_OUT"; exit 42; }
echo "OpenCL health OK on $(hostname)"

conda run --no-capture-output -n scent python \
        validation/generators/scent/run_scent_fixed.py \
        --cfg "$CFG" --root-dir "$FR_ROOT_DIR"
echo "[fr_scent_dock] done ($CFG)"
