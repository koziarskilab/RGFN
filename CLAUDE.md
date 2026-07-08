# CLAUDE.md — repository guide for AI agents and contributors

This is a research fork of **RGFN** (Reaction-GFlowNet) for generating novel
**molecular glue degraders**. We extend upstream RGFN with new oracles, reward
shaping, batch-selection strategies, benchmarks, and dataset tooling.

**Read this file and `docs/ARCHITECTURE.md` before making structural changes.**
For the science/goals, read `docs/RESEARCH_CONTEXT.md`.

---

## The one rule that governs everything: old vs. new

| Path | Owner | Rule |
|---|---|---|
| `rgfn/` | **Upstream** | Pristine RGFN. **Do not edit.** Extend it from `glue/`. |
| `configs/` (except `configs/glue/`) | **Upstream** | Treat as pristine. New configs go in `configs/glue/`. |
| `gin_config/`, `train.py`, `grid_search.py`, `tests/` (upstream parts), `data/chemistry.xlsx`, `data/targets/`, `external/setup_{gnina,gneprop,reinvent,shared}.sh` | **Upstream** | Leave as-is. |
| `glue/` (**production pipeline**) | **Ours** | All new pipeline Python: oracles, rewards, samplers, proxies, datasets, the active-learning loop. The thing we ship. |
| `configs/glue/` | **Ours** | All new gin configs (overlay upstream via `include`). |
| `scripts/` | **Ours** | **Generic** launch layer only: pipeline-wide entry points (`train.py`/`infer.py` wrappers, `active_learning.py`) + generic `submit.sh`. Experiment-specific scripts go in `experiments/<group>/<run>/`, not here. |
| `validation/` (**validation**) | **Ours** | The whole comparative-evaluation world: baseline generators (SynFlowNet, FragGFN, VAE-BO, RGFN adapter), validation-only oracles (Boltz-2), benchmark suites (PMO + our own), the harness, and committed results. |
| `data/` | **Ours** (+ upstream files) | The single **inputs** dir. Ours: `data/models/` (structures/checkpoints), `data/validation-molecules/` (curated known-glue sets), `data/synthetic/` (generated, git-ignored). Upstream (don't move): `data/chemistry.xlsx`, `data/targets/`. |
| `experiments/` | **Ours** | One self-contained dir **per run/experiment**, grouped by type: `active_learning/`, `oracle_validation/`, `ablations/`. Holds that run's code + seeds + small results + README; timestamped run outputs land alongside (git-ignored). Reusable science graduates into `glue/`. See `experiments/README.md`. |
| `Logs/`, `docs/` | **Ours** | Experiment logs + project documentation. |
| `Logs/references/` | **Ours** | Canonical bibliography for papers we build on. Cite by key (`[koziarski2024rgfn]`); annotated index in its `README.md`. PDFs in `pdfs/` are git-ignored. |

We keep `rgfn/` and `configs/` mergeable with upstream. The only deliberate edits
to upstream files are three small operational overrides documented in
**`docs/PATCHES.md`** — read that before "fixing" anything odd in those files.

### The second axis: production pipeline vs. validation

Beyond upstream-vs-ours, our own code splits along a second axis — the **training
pipeline** (`glue/`) vs. the **validation/benchmarking layer** (`validation/`) —
kept apart by one rule:

> **The dependency arrow points one way.** `validation/` may import from `glue/`
> and `rgfn/`. The production pipeline — `glue/`, `scripts/train.py`,
> `configs/glue/` — must **never** import from `validation/`.

This is what keeps slow validation-only oracles (Boltz-2, co-folding, MD) out of
the in-loop reward by construction, and keeps the shipped pipeline understandable
without dragging in every baseline. Everything comparative lives under the single
`validation/` umbrella (generators, oracles, `suites/`, `harness/`, `results/`);
baseline generators are **thin adapters** in `validation/generators/`, their heavy
upstream code installed via `external/setup_*.sh`, not vendored. See
`validation/README.md` and `docs/ARCHITECTURE.md` for the full picture.

---

## How to extend the system (don't edit `rgfn/`)

0. **Understand before you change.** Ground the work in the science *first*: read
   `docs/RESEARCH_CONTEXT.md` (goals, the validated oracle/metric, terminology),
   the relevant experiment logs in `Logs/`, and the source papers in
   `Logs/references/`. Base each consequential choice — architecture, metric, sign
   convention — on the file that *recorded* the decision (paper section, upstream
   `rgfn/` code, or our own log/analysis script), **never** on what merely seems
   plausible. Confirm the change is consistent with prior results and the
   project's direction, and **ask the user to clarify** anything ambiguous before
   building. A design that looks reasonable but silently diverges from the
   publications or our findings is the costliest mistake here — and it passes
   compile/import/smoke tests, so only this step catches it.
1. **New oracle** → implement the science in `glue/oracles/`.
2. **Expose it to training** → write a `@gin.configurable` adapter in
   `glue/proxies/` subclassing upstream `CachedProxyBase`/`ProxyBase`. See
   `glue/proxies/example_glue_proxy.py` for the working template.
3. **New reward / sampler** → `glue/rewards/` or `glue/samplers/`, subclassing the
   upstream base classes in `rgfn/api/`.
4. **Register it** → make sure the module is imported by `glue/registry.py`
   (directly or via its subpackage `__init__`). Gin finds classes by name only
   after they are imported.
5. **Configure it** → add a gin config in `configs/glue/` that `include`s the
   upstream base and references your class as `@YourClass`.
6. **Run it** → `python scripts/train.py --cfg configs/glue/<cfg>.gin`. The
   wrapper imports `glue` first so gin can resolve your component; the root
   `train.py` (upstream) only knows about `rgfn`.

If gin says *"No configurable matching @X"*, the defining module wasn't imported
on the startup path — fix `glue/registry.py`, not the config.

---

## Project goals (what we're building toward)

- New **oracles** (ternary docking / neosubstrate differential; possibly MD).
- New **rewards** (differential reward isolating glue cooperativity).
- Possibly expanded **batch selection**.
- **Benchmarks** comparing RGFN vs. baselines and across protein systems.
- **Models & datasets** as input; **synthetic datasets** as output.

---

## Working notes for agents

- **Compute:** Heavy stack (torch-geometric, openbabel, meeko, gnina) + GPU
  docking runs on **Balam** (SciNet). Balam may be down; when it is, work on a
  Mac laptop and **validate what you can locally** (imports, `py_compile`, gin
  config-include integrity, `bash -n`) — full train/dock validation happens on
  Balam. State clearly in your summary what you did vs. couldn't verify.
- **Login-node smoke tests (Balam *or* Trillium):** before any interactive smoke
  test that imports `glue`/`rgfn` (pulls in dgl) or runs the GPU docking oracle,
  prefix the command with `source ~/bin/rgfn-smoke-env.sh &&`. That one helper
  activates the `rgfn` env and sets the `LD_LIBRARY_PATH` (torch-bundled CUDA libs
  for dgl + QuickVina2-GPU boost libs) + `GNINA` — and works **unchanged on both
  login nodes** (Balam is SciNet-legacy; **Trillium** is an Alliance cluster where
  `module load cuda/11.8.0` does not exist). Balam and Trillium **share
  `/scratch` + `/home`**, so the conda envs and `$SCRATCH/vina_gpu`/`gnina` builds
  are identical from either. **Jobs still submit to Balam compute only** — the
  `submit_*.sh` headers are Balam-specific (SLURM account/partition/`--exclude`)
  and the helper does not touch them.
- **Document as you go:** record structural changes and anything left unverified
  in `docs/REFACTOR_LOG.md` so the next agent can continue or repair the work.
- **Experiment logs:** use the `experiment-log` skill for any real computational
  experiment.
