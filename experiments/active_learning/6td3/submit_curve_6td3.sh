#!/bin/bash
#SBATCH --job-name=al_6td3_curve
#SBATCH --time=15:00:00                        # policy arm: 10 rounds x ~1h GFN train; random arm finishes far sooner
#SBATCH --partition=compute                    # 1-GPU job -> regular partition
#SBATCH --gpus-per-node=1
#SBATCH --exclude=balam008,balam009            # balam008 OpenCL wedged (Logs/013); balam009 degraded (Logs/014)
# Absolute $SCRATCH log paths ($HOME is read-only on compute nodes; Logs/012).
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# One arm/seed of the top-k-vs-oracle-calls curve (Objective 1, [bengio2021gflownet]
# Fig. 7). Parameterised by env vars passed at submit time:
#     ACQ  = policy | random                 (default policy)
#     SEED = integer                         (default 42)
#     CFG  = gin config                      (default the 10-round curve config)
# e.g.  sbatch --export=ALL,ACQ=random,SEED=1 experiments/active_learning/6td3/submit_curve_6td3.sh
# The launcher launch_curve_6td3.sh fires all 3 seeds x 2 arms.
#
# For a cheap single-seed PILOT, point CFG at the already-validated 3-round GPU
# config (Logs/014, job 69517):
#   sbatch --export=ALL,ACQ=policy,SEED=42,CFG=configs/glue/active_learning_6td3_gpu.gin \
#          --job-name=al_6td3_pilot_policy experiments/active_learning/6td3/submit_curve_6td3.sh
#
# Both arms share the config; --acquisition selects the arm. The loop writes the
# trace to <run>/active_learning/oracle_calls.csv; aggregate with
# validation.harness.acquisition_curve once the campaign completes.

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

ACQ=${ACQ:-policy}
SEED=${SEED:-42}
CFG=${CFG:-configs/glue/active_learning_6td3_gpu_curve.gin}
echo "[curve] arm=$ACQ seed=$SEED cfg=$CFG"

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

module load cuda/11.8.0                 # QuickVina2-GPU OpenCL runtime (+ torch/dgl cu118)
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn

export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export GNINA=/scratch/markymoo/gnina/run_gnina.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

# --- OpenCL health gate (both arms dock, so both need a healthy GPU node). ------
HC=$SCRATCH/vina_gpu/opencl_healthcheck
if [ ! -x "$HC" ]; then
    echo "FATAL: healthcheck binary $HC missing -- build it on the login node (see Logs/013)."
    exit 43
fi
HC_OUT=$(CUDA_VISIBLE_DEVICES=0 "$HC" 2>&1)
if ! grep -q "clCreateContext err=0" <<<"$HC_OUT"; then
    echo "FATAL: NVIDIA OpenCL clCreateContext FAILS on $(hostname) -- bad node."
    echo "$HC_OUT"
    exit 42
fi
echo "OpenCL health OK on $(hostname)"

# --- Pre-flight DOCK gate (OpenCL probe necessary but not sufficient; Logs/014). --
python experiments/active_learning/6td3/preflight_dock.py
PF=$?
if [ "$PF" -ne 0 ]; then
    echo "FATAL: pre-flight docking failed on $(hostname) (exit $PF) -- add it to --exclude and resubmit."
    exit "$PF"
fi

python scripts/active_learning.py \
        --cfg "$CFG" \
        --seed "$SEED" \
        --acquisition "$ACQ" \
        --root-dir "$AL_ROOT_DIR"
