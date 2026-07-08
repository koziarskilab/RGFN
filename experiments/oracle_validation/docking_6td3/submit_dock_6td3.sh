#!/bin/bash
#SBATCH --job-name=dock6td3
#SBATCH --partition=debug_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=64
#SBATCH --time=01:00:00
#SBATCH --chdir=/scratch/markymoo/rgfn_runs
#SBATCH --output=dock6td3-%j.out
#SBATCH --error=dock6td3-%j.err

# 6TD3/CR8 glue docking on a full debug node (4x A100, 64 cores). Per Balam docs: multi-GPU jobs
# use a *_full_node partition and must NOT request cpus/mem explicitly (scheduled per-GPU: 16
# cores + ~256GB each). $HOME is read-only on compute nodes, so all outputs + temp go to $SCRATCH;
# receptors/inputs are read from the repo.
set -euo pipefail

REPO=/home/markymoo/projects/RGFN_Fork/RGFN-Fork
DOCK=$REPO/experiments/oracle_validation/docking_6td3
export OUTDIR=$SCRATCH/rgfn_runs/dock_6td3_${SLURM_JOB_ID}
export WORK=$SCRATCH/dock_6td3_cluster_${SLURM_JOB_ID}
export N_GPU=4
export PROCS_PER_GPU=4
mkdir -p "$OUTDIR" "$WORK"

source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn          # RDKit only; gnina loads its own CUDA-12 via run_gnina.sh

echo "node=$(hostname)  gpus=$CUDA_VISIBLE_DEVICES  cpus=${SLURM_CPUS_ON_NODE:-?}"
nvidia-smi -L

python "$DOCK/dock_cluster.py"

echo "===== discrimination summary (fresh 6TD3 vs committed CRBN) ====="
TD3_DIR="$OUTDIR" python "$REPO/experiments/oracle_validation/compare_systems.py" || true
echo "results in $OUTDIR"
