#!/bin/bash
# Launch the full top-k-vs-oracle-calls campaign: 3 seeds x 2 acquisition arms
# (Objective 1, [bengio2021gflownet] Fig. 7). Fires 6 jobs via submit_curve_6td3.sh.
#
#   bash experiments/active_learning/6td3/launch_curve_6td3.sh
#
# Override the grid with env vars, e.g.:
#   SEEDS="1 2 3 4 5" ARMS="policy random" bash .../launch_curve_6td3.sh
#
# When all jobs COMPLETE, draw the curve:
#   python -m validation.harness.acquisition_curve \
#       --runs $SCRATCH/rgfn_runs/experiments/active_learning/6td3_gpu_curve \
#       --out validation/results/6td3_acquisition_curve \
#       --metric topk_best --title "6TD3 - RGFN vs random acquisition"

set -euo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

SEEDS=${SEEDS:-"1 2 3"}
ARMS=${ARMS:-"policy random"}
SUBMIT=experiments/active_learning/6td3/submit_curve_6td3.sh

echo "[launch] seeds=[$SEEDS] arms=[$ARMS]"
for acq in $ARMS; do
    for seed in $SEEDS; do
        jid=$(sbatch --parsable --job-name="al_6td3_curve_${acq}_s${seed}" \
                     --export=ALL,ACQ="$acq",SEED="$seed" "$SUBMIT")
        echo "[launch] submitted arm=$acq seed=$seed -> job $jid"
    done
done
echo "[launch] done. Watch with: squeue -u \$USER"
