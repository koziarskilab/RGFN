# FragGFN 6TD3 active-learning run (non-synthesizable baseline)

The head-to-head **foil** for the RGFN run in `../6td3/`. Same active-learning
protocol (`[bengio2021gflownet]` Alg. 1), same docking oracle, same seed `D_0`,
same budget — the **only** difference is the generator:

| | RGFN (`../6td3/`) | FragGFN (here) |
|---|---|---|
| Generator | reaction DAG over building blocks (`rgfn/`) | fragment-junction GFlowNet (Recursion `gflownet`) |
| Synthesizable? | ✅ by construction (route per molecule) | ❌ `has_route=0`, no `routes.jsonl` |
| Proxy `M` | `glue.proxies.LearnedGlueProxy` (atom-graph MPNN) | `AtomMPNNProxy` (same MPNN, from `gflownet.models.bengio2021flow`) |
| Oracle `O` | `Docking6TD3GpuOracle` (in-process) | **same** oracle, via the bridge `scripts/score_batch.py` (rgfn env) |
| Env | `rgfn` (py3.11) | `fraggfn` (py3.10) — see below |
| Config | `configs/glue/active_learning_6td3_gpu.gin` | `validation/configs/fraggfn_6td3.yaml` |

This is Objective 4 / 5 evidence in `docs/RESEARCH_CONTEXT.md`: does RGFN's
by-construction synthesizability beat a non-synthesizable generator on the **same
oracle and budget**?

## Why a separate conda env

Recursion's `gflownet` pins python 3.10 / torch 2.1.2 — incompatible with the
`rgfn` env (3.11 / torch 2.3). FragGFN therefore runs in its own `fraggfn` env
(built by `external/setup_fraggfn.sh`) and reaches the shared docking oracle
across the env boundary: each round the loop shells out to
`conda run -n rgfn python scripts/score_batch.py …`, which scores the batch AND
writes the standard candidate-dataset shard + `batch_metrics`. The oracle is thus
the single shared scoring standard for every entrant. Full rationale:
`validation/generators/fraggfn/README.md`.

## Components

- Adapter / loop: `validation/generators/fraggfn/` (`proxy.py`, `task.py`,
  `al_loop.py`, `run_fraggfn_al.py`).
- Oracle bridge: `scripts/score_batch.py` (rgfn env; reusable by any baseline).
- Config: `validation/configs/fraggfn_6td3.yaml` (budget matched to the RGFN gin).
- Seed `D_0`: **reuses** `../6td3/seed_6td3.csv` (408 validated docking labels) —
  identical to the RGFN run.

## Run (Balam compute node)

```
sbatch experiments/active_learning/fraggfn_6td3/submit_fraggfn_6td3.sh
```

Outputs land (git-ignored) under
`$SCRATCH/rgfn_runs/experiments/active_learning/fraggfn_6td3/<timestamp>/`:

```
active_learning/
  dataset_round_NNN.csv         # accumulating D after each round
  top_k.csv                     # the Top-K deliverable
  suggestions/
    shards/round_NNN.csv        # per-round labelled batch (crash-safe)
    round_NNN_batch.smi         # the batch fed to the oracle bridge
    round_NNN_labels.csv        # bridge output (label + vina_t2/vina_t1/cnnsc/…)
    batch_metrics.csv           # per-round diversity / medchem / score metrics
    candidates.csv + manifest.json   # standard format (has_route=0), built by --finalize
train/                          # gflownet logs + model_state.pt
```

`candidates.csv` is the standard format read by the validation harness, so this
run's output drops in next to the RGFN run's `suggestions/candidates.csv` for a
direct comparison.

## Local CPU smoke (no GPU/docking)

A tiny mock-oracle dry run validates the whole loop without gnina/GPU — see the
verification section of `validation/generators/fraggfn/README.md`.

## Status

- Code + env + config + submit script: implemented and validated.
- **Real 3-round GPU docking run DONE** — Balam job 69482 (balam009, 27m33s),
  completed before the 2026-06-29 ~18:40 cluster outage; all 3 rounds docked
  32/32. Results in `Logs/015`: best `dvina` −4.86, median −2.06, 54% ≤ −2.0,
  fully diverse, `has_route=0` — competitive with RGFN on glue score, so
  synthesizability is the differentiator. Output (git-ignored):
  `$SCRATCH/rgfn_runs/experiments/active_learning/fraggfn_6td3/2026-06-29_17-20-46/`.
- **Remaining:** a matched-oracle RGFN **GPU** rerun on a healthy node — the
  entry-014 RGFN GPU run (job 69481) died on a wedged OpenCL node, so the current
  head-to-head is vs entry 011 (CPU oracle).
