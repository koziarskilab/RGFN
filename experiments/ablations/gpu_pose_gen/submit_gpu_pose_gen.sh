#!/bin/bash
#SBATCH --job-name=gpu_pose_gen
#SBATCH --time=02:00:00
#SBATCH --partition=compute                   # 1-GPU job -> regular (non full_node) partition
#SBATCH --gpus-per-node=1
#SBATCH --exclude=balam008                     # balam008 OpenCL is wedged (see below)
#SBATCH --output=/scratch/markymoo/rgfn_runs/experiments/gpu_pose_gen/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/experiments/gpu_pose_gen/%x-%j.err

# Re-dock the entry-006 molecule set (160 known + 248 decoy) with the GPU
# pose-generation oracle (QuickVina2-GPU search + gnina --score_only CNN selection
# + two-tier Vina differential), then analyse discrimination vs the 006 gnina-search
# baseline (AUROC 0.946) and per-molecule agreement.
#
#   sbatch experiments/ablations/gpu_pose_gen/submit_gpu_pose_gen.sh
#
# COMPUTE-NODE OpenCL HEALTH (job 69468 post-mortem, 2026-06-29): QuickVina2-GPU
# uses NVIDIA *OpenCL*. On a bad node, clCreateContext returns CL_OUT_OF_RESOURCES
# (-5) and every dock silently yields no poses (all rows "no_pose"; empty stderr
# because QV2 output is suppressed). This is NODE-SPECIFIC, not a driver/partition
# problem: a 20-line raw-OpenCL repro (opencl_healthcheck.c) FAILS on balam008 but
# SUCCEEDS on balam009 (same A100-SXM4 / driver). CUDA is unaffected (torch/gnina
# work). So we (a) --exclude the known-bad node and (b) gate on a live OpenCL
# context check below, exiting fast with a clear message if we land on another bad
# node (add it to --exclude and resubmit; ask SciNet to reset its GPU/OpenCL).
# NOTE: --output/--error MUST be absolute $SCRATCH paths (read-only $HOME on
# compute nodes; a relative/$HOME path makes the job die at startup) -- Logs/012.

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

OUT_DIR=$SCRATCH/rgfn_runs/experiments/gpu_pose_gen
mkdir -p "$OUT_DIR"

module load cuda/11.8.0                 # CUDA-11.8: QuickVina2-GPU OpenCL runtime (+ dgl)
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn

# QuickVina2-GPU-2.1 boost runtime libs; gnina launcher (self-loads CUDA-12).
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export GNINA=/scratch/markymoo/gnina/run_gnina.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

# --- OpenCL health gate: prove clCreateContext works on THIS node before docking.
# Uses a prebuilt binary (compiling in-job is unreliable: gcc isn't on PATH under
# the cuda/11.8 module). Build once on the login node and keep it on $SCRATCH:
#   module load cuda/12.3.1
#   gcc -o $SCRATCH/vina_gpu/opencl_healthcheck \
#       experiments/ablations/gpu_pose_gen/opencl_healthcheck.c \
#       -I$CUDA_HOME/include -L$CUDA_HOME/lib64 -lOpenCL
HC=$SCRATCH/vina_gpu/opencl_healthcheck
if [ ! -x "$HC" ]; then
    echo "FATAL: healthcheck binary $HC missing -- build it on the login node (see header)."
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

RESULTS=$OUT_DIR/gpu_pose_gen_results.csv
TIMING=$OUT_DIR/gpu_pose_gen_timing.csv

# --- Step 1: dock all 408 molecules (resumable; one process is fine on a compute
#     node -- no login CPU-time rlimit here, so --max-chunks is unset).
python experiments/ablations/gpu_pose_gen/dock_gpu.py \
        --out "$RESULTS" --timing-csv "$TIMING" \
        --num-modes 9 --exhaustiveness 8000 --chunk 32

# --- Step 2: discrimination + agreement vs the 006 gnina-search baseline.
python experiments/ablations/gpu_pose_gen/analyze_gpu_pose_gen.py \
        --results "$RESULTS" --out-dir "$OUT_DIR"

echo "===== done; results in $OUT_DIR ====="
ls -la "$OUT_DIR"
