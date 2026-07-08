#!/bin/bash
#SBATCH --job-name=dockcrbn
#SBATCH --partition=debug_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=64
#SBATCH --time=01:00:00
#SBATCH --chdir=/scratch/markymoo/rgfn_runs
#SBATCH --output=dockcrbn-%j.out
#SBATCH --error=dockcrbn-%j.err

# CRBN/5HXB warhead-anchored glue docking on a full debug node (4x A100). Computes the GSPT1
# differential to line up against the 6TD3 DDB1 differential. Balam: no explicit cpu/mem;
# outputs to $SCRATCH ($HOME read-only on compute nodes).
set -euo pipefail

REPO=/home/markymoo/projects/RGFN_Fork/RGFN-Fork
DOCK=$REPO/experiments/oracle_validation/docking_crbn
export OUTDIR=$SCRATCH/rgfn_runs/dock_crbn_${SLURM_JOB_ID}
export WORK=$SCRATCH/dock_crbn_cluster_${SLURM_JOB_ID}
export N_GPU=4
export PROCS_PER_GPU=4
export ANCHOR_PASS=B          # flexible warhead anchor
export ANCHOR_EMBED=free      # matches the validated Pass-B batch
export N_CONFS=100
mkdir -p "$OUTDIR" "$WORK"

source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn           # RDKit only; gnina loads its own CUDA-12 via run_gnina.sh

echo "node=$(hostname)  gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

python "$DOCK/dock_cluster_crbn.py"

echo "===== cross-system comparison (CRBN GSPT1 vs 6TD3 DDB1) ====="
CRBN_DIR="$OUTDIR" TD3_DIR="$REPO/experiments/oracle_validation/docking_6td3" python "$REPO/experiments/oracle_validation/compare_systems.py" || true
echo "results in $OUTDIR"
