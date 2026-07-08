# SCENT sEH active-learning run (cost-aware baseline)

The **cost-aware peer** to the RGFN sEH run in `../seh/`, and the sEH analogue of
`../scent_6td3/`. **sEH** is the classic GFlowNet docking benchmark
([bengio2021gflownet], RGFN's `configs/rgfn_seh_docking.gin`) — run first as a
**reproduction check** before the novel 6TD3 glue task.

SCENT ([gainski2025scent]) is a fork of RGFN adding Recursive Cost Guidance, an
Exploitation Penalty, and a Dynamic Library. Same protocol / oracle / seed `D_0` /
budget / β / proxy `M` as the RGFN run; only the generator differs:

| | RGFN (`../seh/`) | SCENT (here) |
|---|---|---|
| Generator | reaction DAG over building blocks (`rgfn/`) | cost-guided reaction-GFN on the SMALL library (shipped prices + yields) |
| Synthesizable? | ✅ `has_route=1` | ✅ `has_route=1` + `routes.jsonl` |
| Proxy `M` | `glue.proxies.LearnedGlueProxy` | `LearnedDockingProxy` (our refit-able MPNN; replaces SCENT's `@SehMoleculeProxy`) |
| Oracle `O` | `DockingSEHOracle` (in-process) | **same** oracle (`docking_seh`), via the bridge `scripts/score_batch.py` (rgfn env) |
| Env | `rgfn` (py3.11) | `scent` (SCENT's package is itself named `rgfn` → namespace clash, needs its own env) |
| Config | `configs/glue/active_learning_seh.gin` | `validation/configs/scent_seh.gin` |

Note SCENT's config swaps SCENT's own sEH **property** proxy for OUR proxy `M`
trained on the real `docking_seh` labels, so the comparison isolates the
*generator*, not the scorer. The oracle score is the raw AutoDock-Vina sEH binding
energy (QuickVina2-GPU; kcal/mol, more negative = better). "Good binder" reporting
cutoff: −8.0 kcal/mol (≈ Q1 of the seed `D_0`; metric-only).

## Components

- Adapter / loop: `validation/generators/scent/` (shared with the 6TD3 run).
- Oracle bridge: `scripts/score_batch.py` (rgfn env; `--oracle docking_seh`).
- Config: `validation/configs/scent_seh.gin` (parsed with CWD = the SCENT clone
  `external/scent`; see the config header).
- Seed `D_0`: **reuses** `../seh/seed_seh.csv` (250 sEH Vina labels).

## Run (Balam compute node)

```
sbatch experiments/active_learning/scent_seh/submit_scent_seh.sh
```

Outputs (git-ignored) under
`$SCRATCH/rgfn_runs/experiments/active_learning/scent_seh/<timestamp>/`; the
`suggestions/candidates.csv` (standard format, `has_route=1`) drops in next to the
RGFN run's for direct comparison.

## Status

Code + env + config + submit script: implemented (gin is a surgical mirror of the
validated `scent_6td3.gin`); `docking_seh` bridge path validated live. **Real
multi-round GPU run pending.**
