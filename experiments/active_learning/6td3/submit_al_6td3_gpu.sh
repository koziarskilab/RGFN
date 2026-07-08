#!/bin/bash
#SBATCH --job-name=al_6td3_gpu
#SBATCH --time=05:00:00                       # GFN training dominates now (docking ~60x faster); headroom over the ~3h expected
#SBATCH --partition=compute                   # 1-GPU job -> the regular (non full_node) partition
#SBATCH --gpus-per-node=1
#SBATCH --exclude=balam008,balam009            # balam008 OpenCL wedged (Logs/013); balam009 degraded post-outage — passes OpenCL probe but QV2-GPU makes 0 poses (Logs/014 jobs 69481/69511)
# Absolute $SCRATCH log paths: $HOME is read-only on compute nodes, so a relative
# %x-%j.out fails to open and SLURM kills the job at startup (Logs/012, job 69450).
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# GPU-oracle 6TD3 active-learning loop (Logs/013 next-step): the SAME 3-round mini
# loop as the entry-012 CPU baseline (job 69451, submit_al_6td3_pregpu.sh), but with
# the GPU differential oracle (@Docking6TD3GpuOracle: QuickVina2-GPU pose search +
# gnina CNN pose-pick + two-tier --score_only). Config active_learning_6td3_gpu.gin
# holds every other knob fixed, so this is a clean diff vs entry 012.
#
# This run also produces the new per-round provenance artifacts (loop writes them
# automatically): <run>/active_learning/suggestions/{round_NNN.csv,
# routes_round_NNN.jsonl, suggestions_all.csv, batch_metrics.csv}.
#
# Submit with:  sbatch experiments/active_learning/6td3/submit_al_6td3_gpu.sh

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

module load cuda/11.8.0                 # CUDA-11.8: QuickVina2-GPU OpenCL runtime (+ torch/dgl cu118)
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn

# QuickVina2-GPU-2.1 boost runtime libs; gnina launcher (self-loads CUDA-12).
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export GNINA=/scratch/markymoo/gnina/run_gnina.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

# --- OpenCL health gate: prove clCreateContext works on THIS node before docking.
# QuickVina2-GPU uses NVIDIA OpenCL; on a wedged node every dock silently yields no
# poses (Logs/013). Prebuilt binary (build once on login node; see exp 013 header).
HC=$SCRATCH/vina_gpu/opencl_healthcheck
if [ ! -x "$HC" ]; then
    echo "FATAL: healthcheck binary $HC missing -- build it on the login node (see Logs/013)."
    exit 43
fi
HC_OUT=$(CUDA_VISIBLE_DEVICES=0 "$HC" 2>&1)
if ! grep -q "clCreateContext err=0" <<<"$HC_OUT"; then
    echo "FATAL: NVIDIA OpenCL clCreateContext FAILS on $(hostname) -- bad node."
    echo "$HC_OUT"
    echo "       Add '$(hostname)' to the #SBATCH --exclude list and resubmit;"
    echo "       report the node to balam-support@scinet.utoronto.ca."
    exit 42
fi
echo "OpenCL health OK on $(hostname)"

# --- Pre-flight DOCK gate: the OpenCL probe above is necessary but not sufficient
# (balam009 passes it yet QV2-GPU makes 0 poses — Logs/014). Dock a couple of seed
# molecules for real and bail in ~2 min if this node can't pose them, rather than
# discovering it 62 min later at round-1 docking.
python experiments/active_learning/6td3/preflight_dock.py
PF=$?
if [ "$PF" -ne 0 ]; then
    echo "FATAL: pre-flight docking failed on $(hostname) (exit $PF) -- add it to --exclude and resubmit."
    exit "$PF"
fi

python scripts/active_learning.py \
        --cfg configs/glue/active_learning_6td3_gpu.gin \
        --seed 42 \
        --root-dir "$AL_ROOT_DIR"
