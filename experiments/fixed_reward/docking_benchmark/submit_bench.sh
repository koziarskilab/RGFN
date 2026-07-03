#!/bin/bash
#SBATCH --job-name=fr_dock_bench
#SBATCH --time=02:00:00                        # ~20 min of docking; 2h gives slack for gnina variance
#SBATCH --partition=compute                    # 1-GPU job -> regular partition
#SBATCH --exclude=balam008                     # balam008 OpenCL wedged (Logs/013/014): QV2-GPU -> all no_pose
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Phase-0 docking-throughput benchmark (docs/plan fuzzy-popping-quilt): measure the cold
# cross-env per-call overhead of scripts/score_batch.py vs the warm in-env dock rate, for
# docking_seh + docking_6td3_gpu, and project full single-shot-run wall-clock. Decides whether
# per-step docking is feasible for the baselines (per-step score_batch vs persistent server) and
# the iteration budget. See experiments/fixed_reward/docking_benchmark/bench_docking_throughput.py.
#
# Submit with:  sbatch experiments/fixed_reward/docking_benchmark/submit_bench.sh

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

export PIP_CACHE_DIR=$SCRATCH/.cache/pip
OUT_DIR=$SCRATCH/rgfn_runs/docking_benchmark/${SLURM_JOB_ID:-manual}
mkdir -p "$OUT_DIR" "$PIP_CACHE_DIR"

module load cuda/11.8.0                 # dgl graphbolt AND QuickVina2-GPU OpenCL runtime
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn

# QuickVina2-GPU-2.1 boost runtime libs + gnina launcher (inherited by the cold
# `conda run -n rgfn score_batch.py` subprocesses the benchmark spawns).
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export GNINA=/scratch/markymoo/gnina/run_gnina.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname)  out=$OUT_DIR"; nvidia-smi -L

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

python experiments/fixed_reward/docking_benchmark/bench_docking_throughput.py \
        --oracles docking_seh docking_6td3_gpu \
        --smiles-csv experiments/active_learning/6td3/seed_6td3.csv \
        --n 200 --batch-sizes 8 32 100 --repeats 2 \
        --iters 400 1000 --mols-per-iter 100 \
        --out-dir "$OUT_DIR"
echo "[submit_bench] done -> $OUT_DIR"
