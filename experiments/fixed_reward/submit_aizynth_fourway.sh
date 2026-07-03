#!/bin/bash
#SBATCH --job-name=fr_aizynth
#SBATCH --time=06:00:00                       # AiZynth retrosynthesis over up to 4 x top-100 candidates
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1                      # AiZynth is CPU-only, but all Balam partitions are GPU nodes
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Score a matched four-way fixed-reward benchmark for synthesizability with AiZynthFinder,
# then assemble the comparison table. Handles either reward system via the SYSTEM env var:
#   SYSTEM=seh  (default) -> sEH proxy four-way  -> validation/results/seh_fixed_reward/
#   SYSTEM=drd2           -> DRD2 oracle four-way -> validation/results/drd2_fixed_reward/
# Robust to partial completion: processes whichever generators' candidate datasets exist
# (skips the rest), so it can run after RGFN alone or after all four.
#
# Submit with:  sbatch experiments/fixed_reward/submit_aizynth_fourway.sh              # sEH
#               sbatch --export=ALL,SYSTEM=drd2 experiments/fixed_reward/submit_aizynth_fourway.sh  # DRD2

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

export PIP_CACHE_DIR=$SCRATCH/.cache/pip
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

SYSTEM=${SYSTEM:-seh}
FR_ROOT=$SCRATCH/rgfn_runs/experiments/fixed_reward
CONFIG=data/models/aizynthfinder/config.yml
# $HOME (the repo) is READ-ONLY on Balam compute nodes, so the aggregated table must be
# written to $SCRATCH here — NOT into validation/results/ (that write fails with
# PermissionError; jobs 69688/69689). Copy it into the repo from the login node afterwards:
#   cp $SCRATCH/rgfn_runs/results/${SYSTEM}_fixed_reward/comparison.* validation/results/${SYSTEM}_fixed_reward/
# (The per-dataset synthesizability_summary.json files are written under each candidate dir
# on $SCRATCH by synthesizability.py, so those are unaffected.)
RESULTS=$SCRATCH/rgfn_runs/results/${SYSTEM}_fixed_reward
mkdir -p "$RESULTS"
echo "host=$(hostname) SYSTEM=$SYSTEM  results->$RESULTS"

# generator label -> run-dir subfolder under $FR_ROOT (RGFN's sEH run dir is 'seh_proxy';
# all others follow <gen>_<system>).
if [ "$SYSTEM" = "seh" ]; then
    declare -A SUBDIR=( [rgfn]=seh_proxy [fraggfn]=fraggfn_seh [rxnflow]=rxnflow_seh [scent]=scent_seh )
else
    declare -A SUBDIR=( [rgfn]=$SYSTEM [fraggfn]=fraggfn_$SYSTEM [rxnflow]=rxnflow_$SYSTEM [scent]=scent_$SYSTEM )
fi

AGG_ARGS=()
for gen in rgfn fraggfn rxnflow scent; do
    # newest candidates dir for this generator (timestamped run dirs).
    cand=$(ls -td "$FR_ROOT/${SUBDIR[$gen]}"/*/fixed_reward/candidates 2>/dev/null | head -1)
    if [ -z "$cand" ] || [ ! -f "$cand/candidates.csv" ]; then
        echo "[aizynth] SKIP $gen — no candidates at $FR_ROOT/${SUBDIR[$gen]}/*/fixed_reward/candidates"
        continue
    fi
    echo "[aizynth] scoring $gen -> $cand"
    conda run --no-capture-output -n aizynth python validation/harness/synthesizability.py \
        --dataset "$cand" --config "$CONFIG" --nproc 16 --top-k 100 \
        || { echo "[aizynth] WARNING $gen synthesizability failed"; continue; }
    AGG_ARGS+=( --dataset "$gen=$cand" )
done

if [ ${#AGG_ARGS[@]} -eq 0 ]; then
    echo "[aizynth] no datasets scored; nothing to aggregate."
    exit 1
fi

echo "[aizynth] aggregating: ${AGG_ARGS[*]}"
conda run --no-capture-output -n rgfn python validation/harness/aggregate_synthesizability.py \
    "${AGG_ARGS[@]}" \
    --out "$RESULTS/comparison.csv" \
    --out-md "$RESULTS/comparison.md" \
    --title "Matched four-way fixed-reward ${SYSTEM} benchmark (RGFN / FragGFN / RxnFlow / SCENT)"
echo "[aizynth] done -> $RESULTS/comparison.csv"
