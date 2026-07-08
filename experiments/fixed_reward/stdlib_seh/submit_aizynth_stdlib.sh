#!/bin/bash
#SBATCH --job-name=fr_aizynth_stdlib
#SBATCH --time=06:00:00                       # AiZynth retrosynthesis over 4 x top-100 candidates
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1                      # AiZynth is CPU-only, but all Balam partitions are GPU nodes
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Synthesizability (AiZynthFinder + SA) for the SAME-LIBRARY four-way sEH benchmark (Logs/020),
# then assemble the comparison table. Mirrors submit_aizynth_fourway.sh but points at the
# shared-library (glue_standard_v1) run dirs:
#   RGFN    -> seh_proxy_stdlib   (job 69616, GlueReactionDataFactory on glue_standard_v1)
#   RxnFlow -> rxnflow_seh_stdlib (job 69614, baked env from glue_standard_v1)
#   SCENT   -> scent_seh          (job 69608, SMALL == glue_standard_v1 by construction)
#   FragGFN -> fraggfn_seh        (job 69606, native fragment vocab — the non-synth foil)
# Robust to partial completion: skips any generator whose candidates are missing.
#
# Submit with:  sbatch experiments/fixed_reward/stdlib_seh/submit_aizynth_stdlib.sh

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

export PIP_CACHE_DIR=$SCRATCH/.cache/pip
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

FR_ROOT=$SCRATCH/rgfn_runs/experiments/fixed_reward
CONFIG=data/models/aizynthfinder/config.yml
# $HOME (repo) is READ-ONLY on Balam compute nodes -> write the table to $SCRATCH, copy into
# the repo from the login node afterwards:
#   cp $SCRATCH/rgfn_runs/results/seh_fixed_reward_stdlib/comparison.* validation/results/seh_fixed_reward_stdlib/
RESULTS=$SCRATCH/rgfn_runs/results/seh_fixed_reward_stdlib
mkdir -p "$RESULTS"
echo "host=$(hostname)  results->$RESULTS"

declare -A SUBDIR=( [rgfn]=seh_proxy_stdlib [fraggfn]=fraggfn_seh [rxnflow]=rxnflow_seh_stdlib [scent]=scent_seh )

AGG_ARGS=()
for gen in rgfn fraggfn rxnflow scent; do
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
    --title "Same-library four-way fixed-reward sEH benchmark on glue_standard_v1 (RGFN / FragGFN / RxnFlow / SCENT)"
echo "[aizynth] done -> $RESULTS/comparison.csv"
