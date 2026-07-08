# validation/generators/scent/

**SCENT** — *Scalable and Cost-Efficient de Novo Template-Based Molecular
Generation* (`[gainski2025scent]`, arXiv:2506.19865), our **cost-aware** baseline.
SCENT is a **fork of RGFN from the same lab** that keeps RGFN's synthesizable
reaction-template action space and adds three mechanisms on top:

- **Recursive Cost Guidance** — auxiliary models estimate synthesis cost from
  **building-block prices** + **reaction yields** and steer the *backward* policy
  toward cheap routes;
- **Exploitation Penalty** — a visitation-count term so cost guidance doesn't
  collapse diversity;
- **Dynamic Library** — promotes high-value intermediates to building blocks,
  enabling tree-structured (not just left-leaning) synthesis routes.

Comparing it head-to-head with our RGFN entrant — same oracle, seed, budget, β and
proxy — isolates **what SCENT's cost-awareness buys** (cost / diversity / quality)
for glue generation (Objectives 4–5 in `docs/RESEARCH_CONTEXT.md`).

This directory is a **thin adapter** over SCENT; the heavy upstream code is
installed by `external/setup_scent.sh` (cloned under `external/scent`, not
vendored). It is an *entrant* in the benchmark, never part of the production
pipeline.

## The two-environment design (and why)

SCENT's python package is **literally named `rgfn`** (it *is* an RGFN fork, same
py3.11/torch2.3/dgl/gin stack). Installed into the `rgfn` env it would **shadow our
own `rgfn/`**. So SCENT runs in its **own `scent` conda env**, and reaches the
shared docking oracle **across the env boundary** — the same two-env pattern as
FragGFN/RxnFlow, just forced by a **namespace** clash rather than a **version** one:

```
 scent env (py3.11, SCENT's `rgfn`)            rgfn env (py3.11) — the shared standard
 ─────────────────────────────────            ──────────────────────────────────────
 run_scent_al.py  (CWD = external/scent)       scripts/score_batch.py  (oracle bridge)
   ScentActiveLearningLoop                        - Docking6TD3GpuOracle.score_detailed()
     fit M (LearnedDockingProxy) ─┐               - writes standard candidate shard
     train SCENT cost-guided GFN  │                 + per-round batch_metrics (glue/ code)
     sample query batch B + routes│   subprocess    + joins our --routes -> has_route=1
     label B  ────────────────────┴── conda run -n rgfn python scripts/score_batch.py …
     D ∪= (B, labels); refit M next round
```

The bridge is the **single shared scoring standard**: every entrant, from any env,
labels its molecules with the same oracle by calling it. This package never imports
`glue`/our `rgfn` (its env can't load them); `glue` is imported only inside the
bridge (rgfn env).

### Namespace hygiene — why the adapter looks unusual

Because `import rgfn` must resolve to **SCENT's installed package**, the runner
(`run_scent_al.py`) deliberately:
- **never** puts the RGFN-Fork repo root on `sys.path` (that would shadow SCENT's
  `rgfn` with our repo-local `rgfn/`). The adapter modules import each other as
  plain **siblings** (`import proxy`/`import route`), off the runner's own dir;
- **`chdir`s into `external/scent`** so SCENT's relative paths resolve exactly as
  its own `train.py` expects: gin `include 'configs/…'`, `import gin_config`, and
  the SMALL library's `data/small/*` cost/yield files. Every *our*-side path
  (config, seed CSV, run dir, the repo root the bridge runs in) is absolutised up
  front, before the chdir.

This is why the adapter is **gin-driven** (like `scripts/active_learning.py`),
unlike the FragGFN/RxnFlow adapters (which build a `gflownet` `Config`
programmatically) — SCENT is configured entirely through its own gin, so we author
a gin AL config that `include`s SCENT's `scent_base.gin` and swaps in our proxy.

## Files

| File | Role |
|---|---|
| `proxy.py` | `LearnedDockingProxy` — the in-loop reward `M`. Same Bengio-2021 atom-graph MPNN as RGFN's `glue.proxies.LearnedGlueProxy`, here subclassing **SCENT's** `CachedProxyBase` and importing **SCENT's** bundled `MPNNet`/`mol2graph` (identical symbols, its own `rgfn` fork). Fresh init, refit each round. |
| `route.py` | self-contained copy of `glue.active_learning.route` (`extract_route`/`route_to_str`) for the `scent` env. SCENT is synthesizable → routes recorded. |
| `al_loop.py` | `ScentActiveLearningLoop` — `[bengio2021gflownet]` Alg. 1, mirroring `glue/active_learning/loop.py`; `LabelStore` = the accumulating `D`; scores via the oracle bridge and passes `--routes`. |
| `run_scent_al.py` | entry point: absolutise paths, chdir into the SCENT clone, register gin classes, parse the gin config, run. |

Config: `validation/configs/scent_6td3.gin` (+ `scent_smoke.gin`). Run scaffolding
(Balam submit): `experiments/active_learning/scent_6td3/`.

## Faithfulness to the RGFN run (apples-to-apples)

Held identical to `configs/glue/active_learning_6td3_gpu.gin`: **oracle**
(`Docking6TD3GpuOracle`, same knobs), **seed** `D_0`
(`experiments/active_learning/6td3/seed_6td3.csv`), **budget** (3 rounds × 32
query/batch, 300 GFN steps/round, top-16), **β=8**, and the **proxy** architecture
(atom-graph MPNN, refit on the full history each round). SCENT's reward is its
exponential boosting `exp(signed·β)` over our proxy value — the same TB target RGFN
trains against. The invariant holds: oracle labels enter only by refitting `M`,
never as a direct GFN reward.

The **point** of the experiment is the one intended divergence: SCENT runs its
**full cost-aware method** (cost guidance + exploitation penalty + dynamic library)
on its **SMALL building-block library**, which ships the prices + yields its cost
model needs (`external/scent/data/small/{fragment_to_real_cost.json,
templates_yields.csv}`). That library differs from RGFN's `chemistry.xlsx` — an
inherent property of comparing a different generator, not a confound in the
reward/oracle/budget, which are identical. (To ablate cost-awareness in-codebase,
comment the cost-guidance / exploitation / dynamic-library `include`s in SCENT's
`configs/scent_base.gin` — the toggles SCENT itself ships.)

## Install

```
bash external/setup_scent.sh          # clones SCENT + creates the scent env + installs
```

## Run

```
# Balam compute node (real GPU docking):
sbatch experiments/active_learning/scent_6td3/submit_scent_6td3.sh

# Manual (from repo root):
conda run -n scent python validation/generators/scent/run_scent_al.py \
    --cfg validation/configs/scent_6td3.gin \
    --seed-csv experiments/active_learning/6td3/seed_6td3.csv \
    --root-dir $SCRATCH/rgfn_runs/experiments
```

## Local CPU smoke (no GPU / no docking)

Validate the full loop with the cheap mock oracle (the bridge still needs the
`rgfn` env, which needs `module load cuda/11.8.0` for dgl):

```
module load cuda/11.8.0
conda run -n scent python validation/generators/scent/run_scent_al.py \
    --cfg validation/configs/scent_smoke.gin \
    --seed-csv experiments/active_learning/6td3/seed_6td3.csv \
    --root-dir /tmp/scent_smoke
```

(`scent_smoke.gin` is `scent_6td3.gin` with `oracle_name='mock'`, tiny
`n_rounds`/`n_iterations`/`query_batch_size`.)
