#!/bin/bash
#SBATCH --job-name=fr_fgfn_dock
#SBATCH --time=20:00:00
#SBATCH --partition=compute
#SBATCH --exclude=balam008
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# FragGFN per-step DOCKING fixed-reward (the non-synthesizable foil). Pass the config as $1:
#   sbatch experiments/fixed_reward/baseline_docking/submit_fraggfn_docking.sh validation/configs/fraggfn_6td3_docking_fixed.yaml
#   sbatch experiments/fixed_reward/baseline_docking/submit_fraggfn_docking.sh validation/configs/fraggfn_clpp_docking_fixed.yaml
# Runs in the `fraggfn` env; EXPORTS the docking libs + cuda module so the per-step
# `conda run -n rgfn score_batch.py` subprocess (DockingBridgeReward) can run QuickVina2-GPU.
set -uo pipefail
CFG=${1:?usage: submit_fraggfn_docking.sh <validation/configs/*.yaml>}
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

conda run --no-capture-output -n fraggfn python \
        validation/generators/fraggfn/run_fraggfn_fixed.py \
        --cfg "$CFG" --root-dir "$FR_ROOT_DIR"
echo "[fr_fgfn_dock] done ($CFG)"
