# validation/generators/fraggfn/

**FragGFN** — fragment-based GFlowNet, the *non-synthesizable* baseline from the
RGFN paper (`[koziarski2024rgfn]`). RGFN's claim is synthesizability **by
construction** (a reaction DAG over a building-block library); FragGFN is the foil
that builds molecules by joining fragments at arbitrary attachment points
(Recursion's `FragMolBuildingEnvContext`) — valid graphs with **no synthesis
route**. Comparing the two on the same oracle and budget is the headline evidence
for Objectives 4–5 in `docs/RESEARCH_CONTEXT.md`.

This directory is a **thin adapter** over Recursion's
[`gflownet`](https://github.com/recursionpharma/gflownet); the heavy upstream code
is installed by `external/setup_fraggfn.sh` (cloned under `external/`, not
vendored). It is an *entrant* in the benchmark, never part of the production
pipeline.

## The two-environment design (and why)

Recursion's `gflownet` (pinned commit `da99940`) hard-pins **python 3.10 /
torch 2.1.2 / torch-geometric 2.4.0**, which can't coexist with the `rgfn` env
(**python 3.11 / torch 2.3**). So FragGFN runs in its **own `fraggfn` conda env**,
and reaches the shared docking oracle **across the env boundary**:

```
 fraggfn env (py3.10)                         rgfn env (py3.11) — the shared standard
 ───────────────────                          ──────────────────────────────────────
 run_fraggfn_al.py                            scripts/score_batch.py  (oracle bridge)
   FragGFNActiveLearningLoop                    - Docking6TD3GpuOracle.score_detailed()
     fit M (AtomMPNNProxy) ─┐                   - writes standard candidate shard
     train fragment-GFN     │                     + batch_metrics (glue/ code)
     sample query batch B   │     subprocess
     label B  ──────────────┴── conda run -n rgfn python scripts/score_batch.py …
     D ∪= (B, labels); refit M next round
```

The bridge is the **single shared scoring standard**: every entrant (FragGFN,
SynFlowNet, VAE-BO, …), from any env, labels its molecules with the same oracle by
calling it. This is exactly "different env per benchmarked generator; one shared
oracle to compare them all." `glue` is imported only inside the bridge (rgfn env);
this package never imports `glue`/`rgfn` (its env can't load them).

## Files

| File | Role |
|---|---|
| `proxy.py` | `AtomMPNNProxy` — the in-loop reward `M`. Same Bengio-2021 atom-graph MPNN as RGFN's `glue.proxies.LearnedGlueProxy`, imported from `gflownet.models.bengio2021flow` (no `rgfn` dep). Fit each round; `reward()` maps to `exp(signed·)`. |
| `task.py` | `FragGFNTask` (GFNTask: reward = `M`) + `FragGFNTrainer` (`StandardOnlineTrainer` over `FragMolBuildingEnvContext`, `num_workers=0`, constant β). |
| `al_loop.py` | `FragGFNActiveLearningLoop` — `[bengio2021gflownet]` Alg. 1, mirroring `glue/active_learning/loop.py`; `LabelStore` = the accumulating `D`. |
| `run_fraggfn_al.py` | entry point: load YAML, build proxy + trainer + loop, run. |

Config: `validation/configs/fraggfn_6td3.yaml`. Run scaffolding (Balam submit,
seed reuse): `experiments/active_learning/fraggfn_6td3/`.

## Faithfulness to the RGFN run (apples-to-apples)

Held identical to `configs/glue/active_learning_6td3_gpu.gin`: **oracle**
(`Docking6TD3GpuOracle`, same knobs), **seed** `D_0`
(`experiments/active_learning/6td3/seed_6td3.csv`), **budget** (3 rounds × 32
query/batch, 300 GFN steps/round, top-16), **β=8**, and the **proxy** architecture
(atom-graph MPNN, refit on the full history each round). The β maps exactly: a
constant `TemperatureConditional` β over reward `R=exp(signed·v)` gives the TB
target `R^β = exp(signed·v·β)`, the same target RGFN trains against. The invariant
holds: oracle labels enter only by refitting `M`, never as a direct GFN reward.

The one intended divergence is the **point of the experiment**: FragGFN molecules
are non-synthesizable, so the standard dataset records `has_route=0` and writes no
`routes.jsonl`.

## Install

```
bash external/setup_fraggfn.sh        # creates the fraggfn env + installs gflownet
```

## Run

```
# Balam compute node (real GPU docking):
sbatch experiments/active_learning/fraggfn_6td3/submit_fraggfn_6td3.sh

# Manual (from repo root):
conda run -n fraggfn python validation/generators/fraggfn/run_fraggfn_al.py \
    --cfg validation/configs/fraggfn_6td3.yaml \
    --seed-csv experiments/active_learning/6td3/seed_6td3.csv \
    --root-dir $SCRATCH/rgfn_runs/experiments
```

## Local CPU smoke (no GPU / no docking)

Validate the full loop with the cheap mock oracle (the bridge still needs the
`rgfn` env, which needs `module load cuda/11.8.0` for dgl):

```
module load cuda/11.8.0
conda run -n fraggfn python validation/generators/fraggfn/run_fraggfn_al.py \
    --cfg validation/configs/fraggfn_smoke.yaml \
    --seed-csv experiments/active_learning/6td3/seed_6td3.csv \
    --device cpu --root-dir /tmp/fraggfn_smoke
```

(`fraggfn_smoke.yaml` is `fraggfn_6td3.yaml` with `oracle.name: mock`, tiny
`n_rounds`/`n_train_steps`/`query_batch_size`.)
