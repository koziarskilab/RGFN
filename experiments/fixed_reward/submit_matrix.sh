#!/bin/bash
# Master launcher for the four-reward x four-generator fixed-reward matrix (Logs/021).
#
#   rewards    : sEH proxy | DRD2 proxy | ClpP docking | 6TD3 differential
#   generators : RGFN | RxnFlow | SCENT  (shared SMALL library glue_standard_v1)
#              + FragGFN (own fragment library, the non-synthesizable foil)
#
# This submits the NEW / MISSING cells only (the "add only new + missing" scope): the two
# docking rewards for ALL FOUR generators (per-step GPU docking; RGFN in-env via
# OracleRewardProxy, baselines via the cross-env score_batch.py bridge, both validated by the
# benchmark 69691 + smoke tests 69692/69693), plus DRD2-on-SMALL for RGFN + RxnFlow (SCENT is
# natively SMALL; FragGFN keeps its own library). The sEH/DRD2 PROXY four-ways that were
# already launched (jobs 69606-69616) are NOT resubmitted here — uncomment the PROXY block to
# re-run everything fresh with uniform provenance.
#
# Run:  bash experiments/fixed_reward/submit_matrix.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
D=experiments/fixed_reward
sub() { echo "  + $*"; sbatch "$@"; }

echo "== RGFN docking (in-env OracleRewardProxy; SMALL library) =="
sub $D/6td3/submit_fixed_6td3.sh
sub $D/clpp/submit_fixed_clpp.sh

echo "== Baseline docking (cross-env score_batch.py bridge) =="
sub $D/baseline_docking/submit_fraggfn_docking.sh validation/configs/fraggfn_6td3_docking_fixed.yaml
sub $D/baseline_docking/submit_fraggfn_docking.sh validation/configs/fraggfn_clpp_docking_fixed.yaml
sub $D/baseline_docking/submit_rxnflow_docking.sh validation/configs/rxnflow_6td3_docking_fixed.yaml
sub $D/baseline_docking/submit_rxnflow_docking.sh validation/configs/rxnflow_clpp_docking_fixed.yaml
sub $D/baseline_docking/submit_scent_docking.sh   validation/configs/scent_6td3_fixed.gin
sub $D/baseline_docking/submit_scent_docking.sh   validation/configs/scent_clpp_fixed.gin

echo "== DRD2 on the SHARED SMALL library (completes the DRD2 four-way on one library) =="
# SCENT DRD2 is already SMALL (scent_drd2_fixed.gin); FragGFN DRD2 keeps its own library.
sub $D/drd2_stdlib/submit_fixed_rgfn_drd2_stdlib.sh
sub $D/drd2_stdlib/submit_rxnflow_drd2_stdlib.sh

# --- OPTIONAL: re-run the sEH + DRD2 PROXY four-ways fresh (uniform provenance) ---
# echo "== PROXY four-ways (uncomment to re-run fresh) =="
# sub $D/stdlib_seh/submit_fixed_rgfn_stdlib_seh.sh
# sub $D/stdlib_seh/submit_fixed_rxnflow_stdlib_seh.sh
# ... (scent_seh_fixed.gin, fraggfn_seh_fixed.yaml, + the DRD2 equivalents)

echo "== submitted. Monitor with: squeue -u markymoo =="
