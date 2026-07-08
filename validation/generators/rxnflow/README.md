# validation/generators/rxnflow/

**RxnFlow** — reaction-template + building-block GFlowNet (`[seo2024rxnflow]`), our
*synthesizable* baseline and the closest published peer to RGFN. Where FragGFN is the
**non-synthesizable** foil (fragment junctions, no route), RxnFlow — like RGFN —
assembles each molecule along an explicit **synthetic pathway** (pick a building block,
apply a reaction template to a chosen reactant), so every sample carries a forward
route. Its headline trick is **action-space subsampling**, which lets it train over a
huge action space (~1.2M building blocks × 71 reaction templates). Comparing RGFN vs.
RxnFlow on the same oracle and budget is the apples-to-apples *synthesizable-vs-
synthesizable* evidence for Objectives 4–5 in `docs/RESEARCH_CONTEXT.md`.

This directory is a **thin adapter** over RxnFlow
([SeonghwanSeo/RxnFlow](https://github.com/SeonghwanSeo/RxnFlow), which bundles
Recursion's `gflownet`); the heavy upstream code is installed by
`external/setup_rxnflow.sh` (cloned under `external/`, not vendored). It is an *entrant*
in the benchmark, never part of the production pipeline.

## The two-environment design (and why)

RxnFlow pins **python ≥3.12 / torch 2.5.1+cu121**, which can't coexist with the `rgfn`
env (**python 3.11 / torch 2.3+cu118**) or the `fraggfn` env (py3.10 / torch 2.1.2). So
RxnFlow runs in its **own `rxnflow` conda env**, and reaches the shared docking oracle
**across the env boundary** — the same pattern FragGFN uses:

```
 rxnflow env (py3.12)                          rgfn env (py3.11) — the shared standard
 ───────────────────                           ──────────────────────────────────────
 run_rxnflow_al.py                             scripts/score_batch.py  (oracle bridge)
   RxnFlowActiveLearningLoop                     - Docking6TD3GpuOracle.score_detailed()
     fit M (AtomMPNNProxy) ─┐                    - writes standard candidate shard
     train RxnFlow synth-GFN│                      + batch_metrics (glue/ code)
     sample batch B + ROUTES│      subprocess     - joins ROUTES (--routes) -> routes.jsonl
     label B  ──────────────┴── conda run -n rgfn python scripts/score_batch.py …
     D ∪= (B, labels); refit M next round
```

The bridge is the **single shared scoring standard**: every entrant (FragGFN, RxnFlow,
…), from any env, labels its molecules with the same oracle by calling it. `glue` is
imported only inside the bridge (rgfn env); this package never imports `glue`/`rgfn`
(its env can't load them).

## Files

| File | Role |
|---|---|
| `proxy.py` | Re-exports FragGFN's `AtomMPNNProxy` — the in-loop reward `M`, **the same proxy as every other entrant** (Bengio-2021 atom-graph MPNN from `gflownet.models.bengio2021flow`). Reused, not copied, so `M` is identical by construction. |
| `task.py` | `RxnFlowTask` (RxnFlow `BaseTask`: reward = `M`) + `RxnFlowGlueTrainer` (`RxnFlowTrainer` with our proxy, `num_workers=0`, constant β). |
| `al_loop.py` | `RxnFlowActiveLearningLoop` — `[bengio2021gflownet]` Alg. 1, mirroring `fraggfn/al_loop.py` (incl. the all-NaN oracle-failure guard); `LabelStore` = the accumulating `D`. Adds `extract_route(...)` so each sampled molecule's synthesis route is recorded. |
| `run_rxnflow_al.py` | entry point: load YAML, build proxy + trainer + loop, run. |

Config: `validation/configs/rxnflow_6td3.yaml`. Run scaffolding (Balam submit, seed
reuse): `experiments/active_learning/rxnflow_6td3/`.

## Faithfulness to the RGFN run (apples-to-apples)

Held identical to `configs/glue/active_learning_6td3_gpu.gin`: **oracle**
(`Docking6TD3GpuOracle`, same knobs), **seed** `D_0`
(`experiments/active_learning/6td3/seed_6td3.csv`), **budget** (3 rounds × 32
query/batch, 300 GFN steps/round, top-16), **β=8**, and the **proxy** `M` (the same
atom-graph MPNN, refit on the full history each round). The β maps exactly: a constant
`TemperatureConditional` β over reward `R=exp(signed·v)` gives the TB target
`R^β = exp(signed·v·β)`, the same target RGFN trains against. The invariant holds:
oracle labels enter only by refitting `M`, never as a direct GFN reward.

Unlike FragGFN, RxnFlow is **synthesizable**: the standard dataset records `has_route=1`
+ `routes.jsonl` (the per-round routes are passed to the bridge via `--routes`). This is
the apples-to-apples match for RGFN — both synthesizable — and the differentiator vs.
the non-synthesizable foil.

> **Status: written, not yet run.** The RxnFlow upstream API (Config schema, env-dir
> wiring, per-action route fields used by `extract_route`) is taken from the RxnFlow
> docs/examples and is **flagged for Balam validation** (`docs/REFACTOR_LOG.md`) — the
> heavy stack only exists on the cluster. Locally validated by `py_compile` only.

## Install

```
bash external/setup_rxnflow.sh        # creates the rxnflow env + installs RxnFlow + prepares blocks/templates
```

This also prepares the env directory (building blocks + reaction templates) under
`data/models/rxnflow_env/`; point `rxnflow.env_dir` in the config at it.

## Run

```
# Balam compute node (real GPU docking):
sbatch experiments/active_learning/rxnflow_6td3/submit_rxnflow_6td3.sh

# Manual (from repo root):
conda run -n rxnflow python validation/generators/rxnflow/run_rxnflow_al.py \
    --cfg validation/configs/rxnflow_6td3.yaml \
    --seed-csv experiments/active_learning/6td3/seed_6td3.csv \
    --root-dir $SCRATCH/rgfn_runs/experiments
```

## Local CPU smoke (no GPU / no docking)

Validate the full loop with the cheap mock oracle (the bridge still needs the `rgfn`
env, which needs `module load cuda/11.8.0` for dgl):

```
module load cuda/11.8.0
conda run -n rxnflow python validation/generators/rxnflow/run_rxnflow_al.py \
    --cfg validation/configs/rxnflow_smoke.yaml \
    --seed-csv experiments/active_learning/6td3/seed_6td3.csv \
    --device cpu --root-dir /tmp/rxnflow_smoke
```

(`rxnflow_smoke.yaml` is `rxnflow_6td3.yaml` with `oracle.name: mock`, tiny
`n_rounds`/`n_train_steps`/`query_batch_size`, and a small building-block subset.)
