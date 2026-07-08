#!/bin/bash
#SBATCH --job-name=al_fraggfn_6td3
#SBATCH --time=05:00:00                       # GFN training dominates; docking ~60x faster (GPU oracle). Mirrors al_6td3_gpu.
#SBATCH --partition=compute                   # 1-GPU job -> the regular (non full_node) partition
#SBATCH --exclude=balam008                     # balam008 OpenCL wedged (Logs/013); QV2-GPU yields all-no_pose there
#SBATCH --gpus-per-node=1
# Absolute $SCRATCH log paths: $HOME is read-only on compute nodes (Logs/012).
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# FragGFN (non-synthesizable baseline) active-learning loop on 6TD3 — the head-to-head
# foil for the RGFN run (experiments/active_learning/6td3/submit_al_6td3_gpu.sh).
# IDENTICAL oracle / seed / budget; only the generator differs (Recursion's fragment
# GFlowNet via FragMolBuildingEnvContext, no synthesis route -> has_route=0).
#
# Two-env design (see validation/generators/fraggfn/README.md):
#   * the loop runs in the `fraggfn` env (py3.10 + Recursion's gflownet);
#   * each round it labels its query batch by shelling out to the SHARED oracle
#     bridge `scripts/score_batch.py` under the `rgfn` env (gnina + QuickVina2-GPU
#     + glue). One CUDA-11.8 module + boost libs cover both envs.
#
# Submit with:  sbatch experiments/active_learning/fraggfn_6td3/submit_fraggfn_6td3.sh

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

# --- Caches + run outputs: redirect everything away from the read-only $HOME ---
export WANDB_PROJECT=rgfn
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export WANDB_CONFIG_DIR=$SCRATCH/.config/wandb
export WANDB_DATA_DIR=$SCRATCH/.cache/wandb
export WANDB_DIR=$SCRATCH/wandb
export WANDB_MODE=offline
export HF_HOME=$SCRATCH/.cache/huggingface
export TORCH_HOME=$SCRATCH/.cache/torch
export PIP_CACHE_DIR=$SCRATCH/.cache/pip

AL_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" \
        "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$AL_ROOT_DIR"

# CUDA-11.8 serves BOTH envs (fraggfn torch cu118; rgfn torch/dgl cu118 + QV2-GPU OpenCL).
module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

# Shared-oracle (rgfn env) runtime deps; exported so the bridge SUBPROCESS inherits
# them even though the loop itself runs under fraggfn.
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export GNINA=/scratch/markymoo/gnina/run_gnina.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

# --- OpenCL health gate (same as the RGFN run): prove clCreateContext works here. -
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

# --- Run the loop under the fraggfn env (bridge subprocess re-enters rgfn). -------
conda activate fraggfn
python validation/generators/fraggfn/run_fraggfn_al.py \
        --cfg validation/configs/fraggfn_6td3.yaml \
        --seed-csv experiments/active_learning/6td3/seed_6td3.csv \
        --seed 42 \
        --root-dir "$AL_ROOT_DIR"
