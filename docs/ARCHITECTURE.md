# Architecture & repository structure

This document is the detailed reference for how the repo is organized and why.
For the short version and the rules agents must follow, see `CLAUDE.md`.

## Design principle: a clean upstream/ours boundary

This is a fork of [RGFN](https://github.com/koziarskilab/RGFN). We keep the
upstream package **pristine and mergeable** and put **all of our work in a
sibling package** (`glue/`) plus clearly-owned top-level directories. The mental
model is simple:

> **Anything under `rgfn/` or `configs/` is upstream. Anything under `glue/`,
> `configs/glue/`, `scripts/`, `validation/`, `experiments/`, `data/` (except the
> upstream `chemistry.xlsx` + `targets/`), `Logs/`, or `docs/` is ours.**

Our own code further splits along a **second axis** — the production training
pipeline vs. the validation/benchmarking layer (see
"[Two axes](#two-axes-upstreamours-and-pipelinevalidation)" below).

This works because **gin resolves components by class name**, not file location.
A `@gin.configurable` class can live anywhere, as long as it is *imported* before
the config is parsed. So we never need to edit upstream to add a component —
we define it in `glue/`, ensure `glue.registry` imports it, and reference it by
name from a config in `configs/glue/`.

## Full layout

```
RGFN-Fork/
├── CLAUDE.md                  # agent/contributor guide (rules + how to extend)
├── docs/
│   ├── ARCHITECTURE.md        # this file (repo layout + where data/results live)
│   ├── RESEARCH_CONTEXT.md    # the science, goals, terminology + experiment index
│   ├── REFACTOR_LOG.md        # running record of the restructure + open items
│   └── PATCHES.md             # the deliberate edits we made to upstream files
│
├── rgfn/                      # UPSTREAM — pristine RGFN package. Do not edit.
├── configs/                   # UPSTREAM gin configs (+ configs/glue/ = ours)
│   └── glue/                  # OURS — new configs, overlay upstream via include
├── gin_config/                # UPSTREAM gin helpers
├── train.py, grid_search.py   # UPSTREAM entry points
├── infer.py                   # OURS — inference script (delegated to by scripts/)
├── tests/                     # UPSTREAM tests (add ours under tests/ as needed)
├── data/
│   ├── chemistry.xlsx         # UPSTREAM — reaction building blocks
│   ├── targets/               # UPSTREAM — docking target pdbqt
│   └── synthetic/             # OURS — generated datasets (gitignored outputs)
├── external/                  # setup scripts (setup_qv2gpu.sh is OURS)
│
├── glue/                      # OURS — PRODUCTION PIPELINE package (all in-loop code)
│   ├── __init__.py            #   imports glue.registry on import
│   ├── registry.py            #   imports every submodule so gin sees our classes
│   ├── oracles/               #   in-loop scoring science: docking_6td3 (gnina ternary),
│   │                          #     docking_seh (QuickVina2-GPU), mock; MD later
│   ├── rewards/               #   reward shaping (neosubstrate differential, ...)
│   ├── samplers/              #   batch-selection strategies
│   ├── proxies/               #   rgfn ProxyBase adapters (gin-registered)
│   │   ├── example_glue_proxy.py   # working template adapter
│   │   └── learned_proxy.py   #   refit-able MPNN surrogate M (the in-loop reward)
│   ├── datasets/              #   input loaders + the accumulating oracle-labelled D
│   ├── metrics/               #   pipeline-side metrics (e.g. scaffold counts)
│   └── active_learning/       #   the multi-round loop driver (loop.py, [bengio2021gflownet] Alg.1)
│
├── scripts/                   # OURS — generic launch layer (pipeline-wide entry points only)
│   ├── train.py               #   imports glue, then runs root train.py
│   ├── infer.py               #   imports glue, then runs root infer.py
│   ├── active_learning.py     #   the active-learning loop entry point (drives any AL config)
│   └── submit.sh              #   generic Balam SLURM submit (training)
│   # experiment-SPECIFIC scripts live with their experiment under experiments/<group>/<run>/, not here
│
│   # ── VALIDATION (imports glue/+rgfn/; NEVER imported by them) ──
├── validation/                # OURS — the whole comparative-evaluation world
│   ├── generators/            #   entrants — baseline generators (thin adapters):
│   │   ├── rgfn/              #     adapter → our glue/ pipeline
│   │   ├── synflownet/        #     reaction-based GFlowNet baseline
│   │   ├── fraggfn/           #     fragment-based GFlowNet (non-synthesizable)
│   │   └── vae_bo/            #     VAE + Bayesian optimization
│   ├── oracles/              #   validation-only oracles (never in-loop):
│   │   └── boltz2/           #     Boltz-2 co-folding high-fidelity check
│   ├── suites/               #   benchmark tasks/metrics:
│   │   ├── pmo/              #     Practical Molecular Optimization (external)
│   │   └── glue_suite/      #     our own glue benchmarks (run on every entrant)
│   ├── harness/             #   runner that drives entrants through suites + eval metrics
│   ├── configs/             #   run specs (entrants × suite × oracle × budget × seeds)
│   └── results/             #   committed tables + plots (small artifacts only)
│   # heavy baseline code installed via external/setup_{synflownet,fraggfn,vae_bo}.sh
│
├── data/                      # OURS — the single INPUTS dir (+ upstream chemistry.xlsx/targets/)
│   ├── chemistry.xlsx         #   UPSTREAM — reaction building blocks (path hardcoded upstream)
│   ├── targets/               #   UPSTREAM — docking receptor pdbqt (path hardcoded upstream)
│   ├── models/                #   structures (.cif), generated docking tiers, checkpoints (gitignored)
│   ├── validation-molecules/  #   curated known-glue sets (validation inputs)
│   └── synthetic/             #   generated datasets (gitignored outputs)
├── experiments/               # OURS — one self-contained dir PER RUN, grouped by type
│   ├── active_learning/       #   AL runs: seh/ 6td3/ (+ mock/ probe/ smoke tests)
│   ├── oracle_validation/     #   docking discrimination: docking_6td3/ docking_crbn/ + shared prep
│   └── ablations/             #   metric studies: pose_selection/ sixway/ mw/
│   # each run dir = its code + seeds + small results + README; timestamped run
│   # outputs land alongside (git-ignored). See experiments/README.md.
└── Logs/                      # OURS — experiment logs (index lives in RESEARCH_CONTEXT.md)
    └── references/             #   paper shelf (bib + git-ignored PDFs)
```

## Two axes: upstream/ours AND pipeline/validation

Our code is organized on two independent axes. The first keeps us mergeable with
upstream; the second keeps the thing we *ship* separate from the machinery that
*measures* it.

| | **Upstream** | **Ours — production pipeline** | **Ours — validation** |
|---|---|---|---|
| What | pristine RGFN | the glue generator we ship | how we prove it's good |
| Where | `rgfn/`, `configs/` | `glue/`, `configs/glue/`, `scripts/` | `validation/` |
| Examples | GFlowNet core | in-loop oracle, proxy, reward, AL loop | baseline generators, PMO, Boltz-2, result tables |

The validation axis is enforced by a **one-way dependency rule**:

> `validation/` may import from `glue/` and `rgfn/`. The production pipeline
> (`glue/`, `scripts/train.py`, `configs/glue/`) must **never** import from
> `validation/`.

Consequences this buys us:

- **No accidental slow rewards.** A validation-only oracle (Boltz-2, co-folding,
  MD) physically cannot be wired into the in-loop reward, because `glue/` can't
  see `validation/`. In-loop oracles live in `glue/oracles/`; post-hoc scorers
  live in `validation/oracles/`.
- **A shippable core.** `glue/` reads as the pipeline alone — none of the baseline
  or benchmark code leaks into it.
- **Honest baselines.** The RGFN benchmark entrant (`validation/generators/rgfn/`)
  is a *thin adapter* over the real `glue/` pipeline, so it can't drift from what
  we ship. Other baselines (SynFlowNet, FragGFN, VAE-BO) are thin adapters too;
  their heavy upstream code is installed via `external/setup_*.sh`, never vendored.

If you ever want `glue/` to import something from `validation/`, the abstraction
is in the wrong place — push it down into `glue/` (or `glue/metrics/`) instead.

## Component flow (validation)

```
spec (validation/configs/*)
  └─ names entrants + suite + oracle + budget/seeds
       └─ validation/harness  ── import ──▶ validation/generators/<entrant>
            │                                   └─ rgfn/ adapter ──▶ glue/ + rgfn/ pipeline
            │                                   └─ synflownet/ … (pkg from external/setup_*.sh)
            ├─ runs each entrant on a validation/suites/<suite> at equal oracle budget
            ├─ optionally scores top-k with validation/oracles/<scorer> (e.g. boltz2)
            └─ writes summarized tables/plots ──▶ validation/results/  (+ a Logs/ entry)
```

The arrow only ever points *into* `glue/`+`rgfn/`, never back out.

## Component flow (training)

```
config (configs/glue/*.gin)
  └─ references @YourProxy by name
       └─ scripts/train.py  ── import glue ──▶ glue.registry imports glue.proxies
                                                   └─ @gin.configurable YourProxy registered
       └─ delegates to root train.py (upstream) ── gin.parse_config ──▶ resolves @YourProxy
                                                   └─ Trainer runs RGFN with our proxy
```

The proxy (`glue/proxies/`) wraps an oracle (`glue/oracles/`) in the upstream
`ProxyBase.compute_proxy_output` contract. Keep oracle science separate from the
proxy plumbing so oracles stay unit-testable without the training stack.

## Why these specific homes

- **`glue/` as a package (not loose folders):** makes our code importable
  (`from glue.oracles import ...`), gives gin a single registration entry point,
  and creates an unambiguous boundary against `rgfn/`.
- **`configs/glue/` overlay (not editing upstream configs):** keeps `configs/`
  mergeable; our configs `include` the upstream base and override.
- **`scripts/` = the generic launch layer, not a junk drawer.** It holds *only*
  pipeline-wide entry points: the launchers that `import glue` (so gin sees our
  components) before delegating to upstream `train.py`/`infer.py` via `runpy` — the
  one way to register our components without editing upstream — plus
  `active_learning.py` (the AL-loop driver that runs *any* `configs/glue/*` AL
  config) and a generic Balam `submit.sh`. The test for "does it belong in
  `scripts/`?" is **generality**: if a script is tied to one experiment/system, it
  lives with that experiment under `experiments/<group>/<run>/` instead (see the
  `ablations/pose_selection/` and `active_learning/seh/` precedents). This keeps
  `scripts/` small and meaningful — "here is how you launch the pipeline."
- **`experiments/<group>/<run>/` — one self-contained dir per run.** Each run/
  experiment owns a directory holding its code (seed-gen, submit, analysis), its
  committed inputs (seeds, small result CSVs), its README, and its (git-ignored,
  timestamped) run outputs. Dirs are **grouped by type** — `active_learning/`,
  `oracle_validation/`, `ablations/` — so the group tells you the kind of
  experiment and the leaf names the system/variant. Reusable, pipeline-wide
  science graduates into `glue/`; the experiment dir keeps only the one-off
  harness around it (e.g. the sEH oracle ships in `glue/oracles/`, while its
  seed-gen/validation/submit tooling stays in `experiments/active_learning/seh/`).
  Run outputs are co-located by `scripts/active_learning.py`, which routes
  `active_learning_<x>.gin` to `experiments/active_learning/<x>/<timestamp>/`.
- **`data/` — the single inputs dir.** All pipeline inputs live under `data/`
  (`models/`, `validation-molecules/`, `synthetic/`, plus the upstream
  `chemistry.xlsx` + `targets/`). It stays named `data/` because the upstream
  files' paths are hardcoded in `rgfn/` + `configs/`; large/regenerable artifacts
  are git-ignored. See `data/README.md`.

## Data, datasets & results

Where the oracle's inputs and outputs live (single source of truth — the science
narrative in `RESEARCH_CONTEXT.md` points here rather than re-listing locations).

**Curated input datasets** (`data/validation-molecules/`):
- `DDB1_CDK12_Glues.csv` — 175 real CDK12-CCNK/DDB1 glues (161 unique, 123
  purine-based). The 6TD3 KNOWN+ positives.
- `CRBN_GSPT1_Glues.csv` — 200 real CRBN/GSPT1 glues (199 active, 188
  glutarimide-bearing; MW up to ~1009, so a few are likely bivalent/PROTAC-range,
  not pure glues). The CRBN KNOWN+ positives.
- `Enamine_CRBN_Molecular_Glue_Library_*.smiles` — purchasable SCAFFOLD library
  (potential glues, **not** validated degraders); git-ignored (large). Used in
  early CRBN runs; superseded by the curated set above for the "known" positives.

**Docking results:**
- **Committed CSVs** (small, per-molecule scores):
  `experiments/oracle_validation/docking_6td3/{known,decoy_cdk}_results.csv` and
  `experiments/oracle_validation/docking_crbn/{known,decoy}_crbn_results.csv`.
  (`*.csv` is git-ignored by default; these are force-tracked via `*_results.csv` — see `.gitignore`.)
- **Full run outputs + SLURM logs** (Balam scratch, not in the repo):
  `/scratch/markymoo/rgfn_runs/dock_6td3_<jobid>/`, `dock_crbn_<jobid>/`, and the
  `dock6td3-<jobid>.out` / `dockcrbn-<jobid>.out` job logs.
- **Cross-system comparison:** `experiments/oracle_validation/compare_systems.py`
  (per-system known-vs-decoy + cross-system); the six-signal ablation and
  MW-control analyses live in `experiments/ablations/{sixway,mw}/`.
- **Active-learning loop outputs:** per-round datasets + Top-K under each run's
  `experiments/active_learning/<system>/<timestamp>/active_learning/` (git-ignored).
