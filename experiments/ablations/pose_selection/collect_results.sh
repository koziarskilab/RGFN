#!/bin/bash
# Copy the pose-selection ablation results from scratch back INTO this repo dir, so the
# committed test dataset lives next to the code that made it (entry 008).
#
# WHY THIS IS A SEPARATE STEP: the docking job (submit_pose_ablation.sh) runs on a Balam
# COMPUTE node, where $HOME (and thus this repo) is READ-ONLY -- it can only write to
# $SCRATCH. So results are written to scratch during docking, and this script pulls them
# back. Run it FROM A LOGIN NODE (where $HOME is writable), after the job finishes.
#
# Usage (from a login node):
#   bash collect_results.sh <SLURM_JOB_ID>            # -> $SCRATCH/rgfn_runs/pose_abl_<id>
#   bash collect_results.sh /full/path/to/OUTDIR      # explicit $OUTDIR
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -ne 1 ]; then
    echo "usage: bash collect_results.sh <SLURM_JOB_ID | /path/to/OUTDIR>" >&2
    exit 1
fi

# Resolve OUTDIR: a bare number is treated as a job id; anything else is a path.
if [[ "$1" =~ ^[0-9]+$ ]]; then
    OUTDIR="${SCRATCH:?SCRATCH not set}/rgfn_runs/pose_abl_$1"
else
    OUTDIR="$1"
fi

if [ ! -d "$OUTDIR" ]; then
    echo "ERROR: OUTDIR not found: $OUTDIR" >&2
    exit 1
fi

# The committed test dataset (per-pose CSVs) + the analysis outputs.
FILES=(
    known_allposes.csv
    decoy_allposes.csv
    pose_selection_stats.csv
    pose_selection_violins.png
)

echo "collecting from: $OUTDIR"
missing=0
for f in "${FILES[@]}"; do
    if [ -f "$OUTDIR/$f" ]; then
        cp -v "$OUTDIR/$f" "$HERE/$f"
    else
        echo "  WARNING: missing $f (skipped)" >&2
        missing=1
    fi
done

echo "done -> $HERE"
[ "$missing" -eq 0 ] || echo "NOTE: some files were missing; re-run analyze_pose_selection.py if needed." >&2
echo "Next: review pose_selection_stats.csv, then commit the CSVs + figure (see README.md)."
