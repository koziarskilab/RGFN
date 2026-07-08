#!/bin/bash
#SBATCH --job-name=pose_abl
#SBATCH --partition=debug_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=64
#SBATCH --time=01:00:00
#SBATCH --chdir=/scratch/markymoo/rgfn_runs
#SBATCH --output=pose_abl-%j.out
#SBATCH --error=pose_abl-%j.err

# Pose-selection ablation redock (entry 008): re-dock the entry-002 known+decoy sets on
# 6TD3/CR8 keeping ALL poses + per-pose Tier-1 scores, so CNN-vs-Vina pose selection can be
# compared offline. Same node profile as submit_dock_6td3.sh: multi-GPU -> *_full_node
# partition, do NOT request cpus/mem (scheduled per-GPU). $HOME is read-only on compute
# nodes, so outputs + temp go to $SCRATCH; receptors/inputs are read from the repo.
set -euo pipefail

REPO=/home/markymoo/projects/RGFN_Fork/RGFN-Fork
ABL=$REPO/experiments/ablations/pose_selection
export OUTDIR=$SCRATCH/rgfn_runs/pose_abl_${SLURM_JOB_ID}
export WORK=$SCRATCH/dock_6td3_allposes_${SLURM_JOB_ID}
export N_GPU=4
export PROCS_PER_GPU=4
mkdir -p "$OUTDIR" "$WORK"

source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn          # RDKit only; gnina loads its own CUDA-12 via run_gnina.sh

echo "node=$(hostname)  gpus=$CUDA_VISIBLE_DEVICES  cpus=${SLURM_CPUS_ON_NODE:-?}"
nvidia-smi -L

# Phase 1+2: redock retaining every pose -> known_allposes.csv / decoy_allposes.csv in $OUTDIR
python "$ABL/dock_allposes.py"

# Analysis: re-pick poses by CNN vs Vina and compare discrimination, reading $OUTDIR CSVs.
# Outputs (stats CSV + figure) also land in $OUTDIR (scratch) -- $HOME is read-only here.
echo "===== pose-selection ablation (CNN vs Vina) ====="
DATA_DIR="$OUTDIR" python "$ABL/analyze_pose_selection.py" || true

echo "results in $OUTDIR"
echo "To pull them back into the repo (run from a LOGIN node, \$HOME is read-only here):"
echo "    bash $ABL/collect_results.sh ${SLURM_JOB_ID}"
