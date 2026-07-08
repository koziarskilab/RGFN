# RxnFlow 6TD3 active-learning run (synthesizable baseline)

The head-to-head **synthesizable peer** for the RGFN run in `../6td3/` (where FragGFN
in `../fraggfn_6td3/` is the *non*-synthesizable foil). Same active-learning protocol
(`[bengio2021gflownet]` Alg. 1), same docking oracle, same seed `D_0`, same budget,
same proxy `M` — the **only** difference is the generator:

| | RGFN (`../6td3/`) | RxnFlow (here) |
|---|---|---|
| Generator | reaction DAG over building blocks (`rgfn/`) | reaction-template + building-block synthesis GFlowNet (RxnFlow) |
| Synthesizable? | ✅ by construction (route per molecule) | ✅ `has_route=1`, `routes.jsonl` |
| Proxy `M` | `glue.proxies.LearnedGlueProxy` (atom-graph MPNN) | `AtomMPNNProxy` (the **same** MPNN, reused from the FragGFN entrant) |
| Oracle `O` | `Docking6TD3GpuOracle` (in-process) | **same** oracle, via the bridge `scripts/score_batch.py` (rgfn env) |
| Env | `rgfn` (py3.11) | `rxnflow` (py3.12, torch cu121) — see below |
| Config | `configs/glue/active_learning_6td3_gpu.gin` | `validation/configs/rxnflow_6td3.yaml` |

This is Objective 4 / 5 evidence in `docs/RESEARCH_CONTEXT.md`, the *synthesizable-vs-
synthesizable* arm: with synthesizability held constant, does RGFN's reaction-DAG action
space match or beat RxnFlow's template+block action space on the **same oracle and
budget** — in glue score, diversity, and route quality?

## Why a separate conda env

RxnFlow pins python ≥3.12 / torch 2.5.1+cu121 — incompatible with both the `rgfn` env
(3.11 / torch 2.3) and the `fraggfn` env (3.10 / torch 2.1.2). RxnFlow therefore runs in
its own `rxnflow` env (built by `external/setup_rxnflow.sh`) and reaches the shared
docking oracle across the env boundary: each round the loop shells out to
`conda run -n rgfn python scripts/score_batch.py …`, which scores the batch AND writes
the standard candidate-dataset shard + `batch_metrics`, joining the per-round routes
(passed via `--routes`) into `routes.jsonl`. The oracle is thus the single shared
scoring standard for every entrant. Full rationale:
`validation/generators/rxnflow/README.md`.

## Components

- Adapter / loop: `validation/generators/rxnflow/` (`proxy.py`, `task.py`,
  `al_loop.py`, `run_rxnflow_al.py`).
- Oracle bridge: `scripts/score_batch.py` (rgfn env; reusable by any baseline,
  now route-aware via `--routes`).
- Config: `validation/configs/rxnflow_6td3.yaml` (budget matched to the RGFN gin).
- Seed `D_0`: **reuses** `../6td3/seed_6td3.csv` (408 validated docking labels) —
  identical to the RGFN/FragGFN runs.
- Env dir (building blocks + 71 reaction templates): prepared by
  `external/setup_rxnflow.sh` into `data/models/rxnflow_env/`.

## Run (Balam compute node)

```
bash external/setup_rxnflow.sh        # once: env + RxnFlow + prepared env dir
sbatch experiments/active_learning/rxnflow_6td3/submit_rxnflow_6td3.sh
```

Outputs land (git-ignored) under
`$SCRATCH/rgfn_runs/experiments/active_learning/rxnflow_6td3/<timestamp>/`:

```
active_learning/
  dataset_round_NNN.csv         # accumulating D after each round
  top_k.csv                     # the Top-K deliverable
  suggestions/
    shards/round_NNN.csv        # per-round labelled batch (crash-safe)
    shards/round_NNN_routes.jsonl  # per-round synthesis routes (joined on --finalize)
    round_NNN_batch.smi         # the batch fed to the oracle bridge
    round_NNN_routes.jsonl      # routes written by the loop, passed via --routes
    round_NNN_labels.csv        # bridge output (label + vina_t2/vina_t1/cnnsc/…)
    batch_metrics.csv           # per-round diversity / medchem / score metrics
    candidates.csv + manifest.json + routes.jsonl   # standard format (has_route=1), built by --finalize
train/                          # gflownet logs + model_state.pt
```

`candidates.csv` is the standard format read by the validation harness, so this run's
output drops in next to the RGFN and FragGFN runs' `suggestions/candidates.csv` for a
direct three-way comparison.

## Local CPU smoke (no GPU/docking)

A tiny mock-oracle dry run validates the whole loop without gnina/GPU — see the
verification section of `validation/generators/rxnflow/README.md`.

## Status

- Code + config + submit script: implemented; locally validated by `py_compile` only.
- **Pending Balam:** `external/setup_rxnflow.sh` (env + RxnFlow install + env-dir prep),
  the CPU mock smoke, then the real 3-round GPU docking run. The RxnFlow upstream API
  (Config schema, env-dir wiring, per-action route fields) is flagged for validation on
  the cluster — see `docs/REFACTOR_LOG.md`. Write up `Logs/016` after the run.
