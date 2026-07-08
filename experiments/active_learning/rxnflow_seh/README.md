# RxnFlow sEH active-learning run (synthesizable baseline)

The head-to-head **synthesizable peer** for the RGFN sEH run in `../seh/`, and the
sEH analogue of `../rxnflow_6td3/`. **sEH** is the classic GFlowNet docking
benchmark ([bengio2021gflownet], RGFN's `configs/rgfn_seh_docking.gin`) — run first
as a **reproduction check** before the novel 6TD3 glue task.

Same protocol / oracle / seed `D_0` / budget / β / proxy `M`; only the generator
differs:

| | RGFN (`../seh/`) | RxnFlow (here) |
|---|---|---|
| Generator | reaction DAG over building blocks (`rgfn/`) | reaction-template + building-block synthesis GFlowNet |
| Synthesizable? | ✅ `has_route=1` | ✅ `has_route=1` + `routes.jsonl` |
| Proxy `M` | `glue.proxies.LearnedGlueProxy` (atom-graph MPNN) | same MPNN family, refit each round |
| Oracle `O` | `DockingSEHOracle` (in-process) | **same** oracle (`docking_seh`), via the bridge `scripts/score_batch.py` (rgfn env), routes passed with `--routes` |
| Env | `rgfn` (py3.11) | `rxnflow` (py3.12 + bundled gflownet) |
| Config | `configs/glue/active_learning_seh.gin` | `validation/configs/rxnflow_seh.yaml` |

This is the **synthesizable-vs-synthesizable** comparison (FragGFN is the
non-synth foil). The oracle score is the raw AutoDock-Vina sEH binding energy
(QuickVina2-GPU; kcal/mol, more negative = better, `higher_is_better=false`) — a
single-target affinity, not the 6TD3 dvina differential. "Good binder" reporting
cutoff: −8.0 kcal/mol (≈ Q1 of the seed `D_0`; metric-only).

## Components

- Adapter / loop: `validation/generators/rxnflow/` (shared with the 6TD3 run).
- Oracle bridge: `scripts/score_batch.py` (rgfn env; `--oracle docking_seh`).
- Config: `validation/configs/rxnflow_seh.yaml`.
- Building blocks + templates: `data/models/rxnflow_env` (`external/setup_rxnflow.sh`).
- Seed `D_0`: **reuses** `../seh/seed_seh.csv` (250 sEH Vina labels).

## Run (Balam compute node)

```
sbatch experiments/active_learning/rxnflow_seh/submit_rxnflow_seh.sh
```

Outputs (git-ignored) under
`$SCRATCH/rgfn_runs/experiments/active_learning/rxnflow_seh/<timestamp>/`; the
`suggestions/candidates.csv` (standard format, `has_route=1` + `routes.jsonl`)
drops in next to the RGFN run's for direct comparison.

## Status

Code + env + config + submit script: implemented; `docking_seh` bridge path
validated live. **Real multi-round GPU run pending.**
