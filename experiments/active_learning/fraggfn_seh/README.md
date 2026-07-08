# FragGFN sEH active-learning run (non-synthesizable baseline)

The head-to-head **foil** for the RGFN sEH run in `../seh/`, and the sEH analogue
of `../fraggfn_6td3/`. **sEH** (soluble epoxide hydrolase) is the classic GFlowNet
docking benchmark ([bengio2021gflownet], RGFN's `configs/rgfn_seh_docking.gin`) —
we run it first as a **reproduction check** that the comparative harness recovers a
well-known result before trusting it on the novel 6TD3 glue task.

Same active-learning protocol, same docking oracle, same seed `D_0`, same budget —
the **only** difference is the generator:

| | RGFN (`../seh/`) | FragGFN (here) |
|---|---|---|
| Generator | reaction DAG over building blocks (`rgfn/`) | fragment-junction GFlowNet (Recursion `gflownet`) |
| Synthesizable? | ✅ by construction (route per molecule) | ❌ `has_route=0`, no `routes.jsonl` |
| Proxy `M` | `glue.proxies.LearnedGlueProxy` (atom-graph MPNN) | `AtomMPNNProxy` (same MPNN, from `gflownet.models.bengio2021flow`) |
| Oracle `O` | `DockingSEHOracle` (in-process) | **same** oracle (`docking_seh`), via the bridge `scripts/score_batch.py` (rgfn env) |
| Env | `rgfn` (py3.11) | `fraggfn` (py3.10) |
| Config | `configs/glue/active_learning_seh.gin` | `validation/configs/fraggfn_seh.yaml` |

The oracle score is the raw **AutoDock-Vina binding energy** against sEH
(QuickVina2-GPU; kcal/mol, more negative = stronger binding = better,
`higher_is_better=false`) — a single-target affinity, **not** the 6TD3 dvina
differential. The "good binder" reporting cutoff is −8.0 kcal/mol (≈ Q1 of the
seed `D_0`; metric-only, never enters optimization).

## Why a separate conda env

Recursion's `gflownet` pins python 3.10 / torch 2.1.2 — incompatible with the
`rgfn` env. FragGFN runs in its own `fraggfn` env (`external/setup_fraggfn.sh`)
and reaches the shared docking oracle across the env boundary: each round the loop
shells out to `conda run -n rgfn python scripts/score_batch.py --oracle docking_seh …`,
which scores the batch AND writes the standard candidate-dataset shard +
`batch_metrics`. Full rationale: `validation/generators/fraggfn/README.md`.

## Components

- Adapter / loop: `validation/generators/fraggfn/` (shared with the 6TD3 run).
- Oracle bridge: `scripts/score_batch.py` (rgfn env; `--oracle docking_seh`).
- Config: `validation/configs/fraggfn_seh.yaml` (budget matched to the RGFN gin).
- Seed `D_0`: **reuses** `../seh/seed_seh.csv` (250 sEH Vina labels) — identical
  to the RGFN run.

## Run (Balam compute node)

```
sbatch experiments/active_learning/fraggfn_seh/submit_fraggfn_seh.sh
```

Outputs land (git-ignored) under
`$SCRATCH/rgfn_runs/experiments/active_learning/fraggfn_seh/<timestamp>/`; the
`suggestions/candidates.csv` (standard format, `has_route=0`) drops in next to the
RGFN run's for a direct comparison.

## Local CPU smoke (no GPU/docking)

The generator-level mock-oracle smoke (`validation/configs/fraggfn_smoke.yaml`)
validates the whole loop without gnina/GPU — it is target-agnostic (mock oracle),
so it covers this run's code path too.

## Status

Code + env + config + submit script: implemented; bridge path validated live
(`docking_seh` scores aspirin −6.3 on the H100). **Real multi-round GPU run pending.**
