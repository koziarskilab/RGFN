#!/bin/bash
# Drive the GPU re-dock on the Balam LOGIN A100 in bounded, resumable processes.
#
# Why not the compute node: QuickVina2-GPU's cached OpenCL kernels
# (Kernel*_Opt.bin) were compiled for the login A100 (PCIE); the compute nodes are
# A100-SXM4 with a newer driver and the cached kernels fail to load there (job
# 69468: every dock returned no poses). Until the kernels are rebuilt per-device,
# QV2 runs only on the login node -- which has a ~3600s per-process CPU rlimit, so
# we call the resumable dock_gpu.py in small --max-chunks batches (fresh process =
# fresh CPU budget) until all 408 molecules are done.
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/11.8.0 2>/dev/null
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export GNINA=/scratch/markymoo/gnina/run_gnina.sh
export PYTHONUNBUFFERED=1

OUT_DIR=$SCRATCH/rgfn_runs/experiments/gpu_pose_gen
RESULTS=$OUT_DIR/gpu_pose_gen_results.csv
TIMING=$OUT_DIR/gpu_pose_gen_timing.csv

for pass in $(seq 1 20); do
    done_n=$(( $(tail -n +2 "$RESULTS" 2>/dev/null | wc -l) ))
    echo "===== pass $pass: $done_n/408 rows so far ====="
    [ "$done_n" -ge 408 ] && { echo "all done"; break; }
    python experiments/ablations/gpu_pose_gen/dock_gpu.py \
        --out "$RESULTS" --timing-csv "$TIMING" \
        --chunk 32 --max-chunks 3 2>&1 | grep -vE "Docking attempt" || true
done

echo "===== docking finished; analysing ====="
python experiments/ablations/gpu_pose_gen/analyze_gpu_pose_gen.py \
    --results "$RESULTS" --out-dir "$OUT_DIR"
echo "===== DONE ====="
ls -la "$OUT_DIR"
