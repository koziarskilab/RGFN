# Refactor log — 2026-06-24 repo restructure

A running record of the restructure that introduced the `glue/` package and the
upstream/ours boundary. **If something is broken, start here**: this documents
what changed, why, what was verified, and what could not be verified (Balam was
down; work was done on a Mac laptop, so no GPU/docking/training runs were possible).

Owner: Mark + Claude. Goal context: `docs/RESEARCH_CONTEXT.md`. Layout: `docs/ARCHITECTURE.md`.

---

## Why

The repo had fused upstream RGFN with our additions at the top level with no clear
boundary. We separated them: `rgfn/` + `configs/` stay pristine/mergeable; all new
work lives in `glue/` and clearly-owned top-level dirs. See `CLAUDE.md`.

Key finding during planning: the upstream package was *already* ~pristine. The
proxies/docking code/configs that looked like ours are actually upstream (2024
commits). Our only edits to upstream files were **three operational overrides**,
now documented in `docs/PATCHES.md`. So this was mostly **additive scaffolding +
moving our own files**, not extracting code out of `rgfn/`.

---

## What changed

### Added (new, zero risk to upstream)
- `glue/` package: `oracles/`, `rewards/`, `samplers/`, `proxies/`, `datasets/`,
  plus `registry.py` (import hub for gin discovery) and `__init__.py`.
  - `glue/proxies/example_glue_proxy.py` — working template adapter (returns QED)
    showing the `CachedProxyBase` + `@gin.configurable` pattern.
- `configs/glue/README.md` — overlay pattern for new configs.
- `scripts/train.py`, `scripts/infer.py` — wrappers that `import glue` (so gin
  sees our components) then delegate to the root entry points via `runpy`.
- `scripts/README.md`, `scripts/hpc/` (holds `submit.sh`).
- `benchmarks/README.md`, `models/README.md` + `.gitkeep`,
  `data/synthetic/README.md` + `.gitkeep`.
- `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/PATCHES.md`, this file.

### Moved
- `pre-processing/`  →  `research/preprocessing/` (whole tree via `git mv`, so
  internal `__file__`-relative and `../test-data` paths are preserved).
- `submit.sh`  →  `scripts/hpc/submit.sh` (and its train line now calls
  `scripts/train.py`).

### Edited (path fixups caused by the moves)
- `research/preprocessing/docking_gnina/submit_dock_crbn.sh` — `$REPO/pre-processing/...`
  → `$REPO/research/preprocessing/...` (2 refs).
- `research/preprocessing/docking_6td3/submit_dock_6td3.sh` — same (2 refs).
- `research/preprocessing/docking_gnina/analysis/plot_discrimination_curves.py` —
  `ROOT` now `parents[4]`; `PRE = ROOT / "research" / "preprocessing"`.
- `docs/RESEARCH_CONTEXT.md` — "Where things live" updated + restructure note.
- `Logs/000_TEMPLATE.md` — `pre-processing/` → `research/preprocessing/` prefixes.
- `scripts/hpc/submit.sh` — train line → `python scripts/train.py ...`.

### Upstream files (deliberately left edited — see docs/PATCHES.md)
- `rgfn/trainer/logger/wandb_logger.py`, `configs/loggers/wandb.gin`,
  `configs/rgfn_seh_docking.gin`. Not reverted; documented as patches.

### Not changed
- `rgfn/` (apart from the documented wandb one-liner), `gin_config/`, upstream
  `configs/`, `train.py`, `grid_search.py`, upstream `data/` inputs, `tests/`.
- Historical experiment logs `Logs/001`–`005` keep their original
  `pre-processing/` paths as point-in-time records (noted in RESEARCH_CONTEXT).

---

## Validation

Environment: macOS laptop, **Balam down**. Could NOT run training, docking, or
anything needing the GPU/heavy deps. Verified what is checkable statically.

### Verified (static, on the laptop — 2026-06-24)
- **`py_compile`** of every `.py` under `glue/`, `scripts/`, `research/` → all compile.
- **No dangling `pre-processing` references** in any `*.py` / `*.sh` / `*.gin` /
  `*.toml` (only historical `Logs/001`–`005` still mention the old path, by design).
- **`bash -n`** on `scripts/hpc/submit.sh` and both moved `submit_dock_*.sh` → OK.
- **gin `include` integrity**: every `include '...'` in `configs/**/*.gin` resolves
  to an existing file.
- **glue import chain**: `glue/__init__` → `registry` → all 5 subpackages →
  `example_glue_proxy` all exist and reference real upstream symbols
  (`CachedProxyBase`, `ReactionStateEarlyTerminal`).
- **Upstream untouched this session**: `git status` shows no modifications to
  `rgfn/` or upstream `configs/` (only `configs/glue/` added). The 3 documented
  patches in `docs/PATCHES.md` are prior commits, not new edits.

### Could NOT verify locally (heavy deps not installed → do on Balam)
- `import glue` fails with `ModuleNotFoundError: No module named 'gin'` — this is
  an **environment** issue (gin-config/torch-geometric/etc. aren't installed on the
  laptop), **not a code issue**. Re-run `python -c "import glue"` in the `rgfn`
  conda env to confirm registration works.

### NOT verified (do on Balam when it's back)
- `python scripts/train.py --cfg configs/glue/<cfg>.gin` actually trains.
- `import glue` succeeds in the real env (depends on torch-geometric etc., which
  may not be installed on the laptop — a local import failure here is likely an
  environment issue, not a code issue; confirm on Balam).
- The moved docking scripts (`research/preprocessing/...`) run end-to-end with
  gnina on a full node.
- The example proxy produces a sensible reward in a real run.

---

## How to continue / repair

- **gin "No configurable matching @X":** the module defining `X` isn't imported on
  startup. Ensure it's imported by `glue/registry.py` (or its subpackage
  `__init__`), and that you launched via `scripts/train.py` (not root `train.py`).
- **A moved docking script can't find a file:** check it's reading relative to its
  own dir (`HERE`) or `../test-data`; repo-root-relative refs were updated to
  `research/preprocessing/` — grep for any stragglers: `grep -rn pre-processing .`
- **An upstream patch vanished after a merge:** re-apply from `docs/PATCHES.md`.
- **Adding new components:** follow the steps in `CLAUDE.md` ("How to extend").

---

# 2026-06-24 — Active-learning loop (branch `active-learning-loop`)

First implementation of the multi-round active-learning loop from
`[bengio2021gflownet]` Alg. 1 (transcribed in `docs/RESEARCH_CONTEXT.md`), wired
to our 6TD3 docking oracle and kept oracle-agnostic for future oracles (MD, etc.).

## Grounding decision (important — re-checked against the papers)

An earlier draft proposed a hand-rolled **MLP** proxy. That was wrong: the proxy
in `[bengio2021gflownet]` A.4 is an **MPNN over the RDKit atom graph** (NNConv +
GRU ×12 → Set2Set → 3-layer MLP, dim 64, LeakyReLU). RGFN already ships *exactly*
that network as `MPNNet` (the class behind `SehMoleculeProxy`, which loads
Bengio's pretrained sEH weights). Key point that resolves the "RGFN isn't
atom-by-atom" worry: RGFN's **reaction/fragment graph drives only the policy/flow
predictor**; the **proxy scores a finished molecule from its atom graph**
(`mol2graph(SMILES)`), identically in both `[bengio2021gflownet]` and RGFN's own
sEH proxy. So we **reuse `MPNNet` + `mol2graph`/`mols2batch` verbatim** rather
than reinventing an architecture.

**We reuse the architecture, NOT the weights.** `LearnedGlueProxy` never calls
`seh_proxy.load_original_model` / `SEHProxyWrapper` — it constructs a fresh
random-init `MPNNet` and trains it from scratch on `D_0` (and refits each round),
exactly as `[bengio2021gflownet]` does ("initializes an MPNN proxy"). Loading the
pretrained sEH weights would be wrong anyway: that proxy predicts sEH affinity,
not our DDB1 differential. (Verified at runtime: our init weights ≠ the pretrained
weights, and differ across seeds.)

## Added (all under `glue/`, zero edits to `rgfn/`)

- `glue/oracles/base.py` — `GlueOracle` ABC (`score(smiles) -> list[float]`): the
  modular seam so the loop is oracle-agnostic.
- `glue/oracles/mock_oracle.py` — `MockGlueOracle`: cheap CPU oracle (QED × MW
  band) so the whole loop runs on a laptop without gnina/GPU. Test fixture, not
  science.
- `glue/oracles/docking_6td3_oracle.py` — `Docking6TD3Oracle`: real two-tier
  gnina docking returning the DDB1 neosubstrate differential (`ddb1_dcnnaff`).
  **Mirrors `research/preprocessing/docking_6td3/dock_cluster.py`** (identical
  gnina flags, best-pose-by-CNNscore, Tier-1 `--score_only`, same differential);
  differs only in orchestration (one in-process batch vs sharded multi-GPU job).
- `glue/proxies/learned_proxy.py` — `LearnedGlueProxy`: trainable wrapper around
  RGFN's `MPNNet` with a `.fit()` (the in-loop reward `M`).
- `glue/datasets/oracle_labeled.py` — `OracleLabeledDataset`: the accumulating
  `D_i = D̂_i ∪ D_{i-1}` store (canonical-SMILES keyed, seed-CSV load, Top-K).
- `glue/active_learning/{__init__,loop}.py` — `ActiveLearningLoop`: the outer
  Alg. 1 orchestrator. New subpackage (registered in `glue/registry.py`).
- `scripts/active_learning.py` — entry point (imports `glue` **and** `Trainer`,
  parses an `configs/glue/` config, builds the loop, runs it).
- `configs/glue/active_learning_mock.gin` (local smoke) and
  `configs/glue/active_learning_6td3.gin` (Balam real run).
- `experiments/active_learning_6td3/` — worked example: `README.md`,
  `build_seeds.py`, `seed_6td3.csv` (408 real validated-docking labels = 160
  known + 248 decoys), `seed_mock.csv` (14 mols for the smoke test).
- Registrations: `glue/registry.py` (+`active_learning`), and the `oracles` /
  `proxies` / `datasets` subpackage `__init__.py` exports.

## How the loop works (and the one invariant)

Each round: **fit `M` on `D_{i-1}`** → `clear_cache()` → **`Trainer.train()`**
against `M(x)^β` → **sample query batch** from the trained forward policy →
**score with `O`** → **`D_i = D̂_i ∪ D_{i-1}`**. The trainer's reward and the loop
share **one** `LearnedGlueProxy` singleton, so refitting updates the reward RGFN
trains on. Invariant preserved: oracle labels enter training **only** by
retraining `M`, never as a direct RGFN reward. Critically, the in-loop reward is
the *cheap MPNN proxy*, so GFN training uses the `reaction` env (not
`reaction_docking`) — **docking never runs in the inner loop**, only on the
per-round query batch.

## Correction (2026-06-25): oracle uses the VINA differential, lower-is-better

The first draft of `Docking6TD3Oracle` computed the **CNNaffinity** differential
(`ddb1_dcnnaff`) with `higher_is_better=True`. That was wrong on both counts.
Checking log 002 + `compare_systems.py` + the result CSVs: the validated +78pt
discrimination is on **`ddb1_dvina` = `Vina(Tier2) − Vina(Tier1)`**, where *more
negative = better glue* (known median −2.20 vs decoy −0.60; 85.6% vs 7.3% below
−1.5). The CNNaffinity differential does **not** discriminate (known +0.23 vs
decoy +0.04, decoy range up to +1.80). CNNscore is still used — but only to pick
the pose (`max cnnsc`), matching `dock_cluster.py`. Fixed (user chose: store raw
`ddb1_dvina`, lower-is-better):
- `docking_6td3_oracle.py` now returns `vina_t2 − vina_t1`; `higher_is_better=False`.
- `LearnedGlueProxy.higher_is_better` is now a constructor arg (set `False` in
  `active_learning_6td3.gin`); dropped the positive output clip (the base config's
  exponential boosting makes the reward positive for any prediction) in favour of
  a symmetric `±clip` sanity bound; invalid molecules get a sign-aware worst-case.
- `ActiveLearningLoop.run()` now asserts `proxy.higher_is_better ==
  oracle.higher_is_better` to catch this class of bug.
- `seed_6td3.csv` / `build_seeds.py` rebuilt from the `ddb1_dvina` column.
- **Regression guard added:** `glue/tests/test_oracle_discrimination.py` — the
  "science validated" counterpart to the wiring smoke test. It checks the metric
  feeding the loop actually separates known glues from decoys (reproduces Log
  002: known 85.6% vs decoy 7.3% strong-bonus, +78pts), and a contrast check
  shows the Vina differential discriminates (+78pts) while CNNaffinity does not
  (+0pts). Verified non-vacuous: it FAILS if the seed is built from
  `ddb1_dcnnaff`. Runs on a laptop from committed data (no gnina); also runnable
  standalone (`python glue/tests/test_oracle_discrimination.py`).

**Process lesson (root cause of both this and the MLP/MPNN error):** each
consequential scientific choice was filled by inference instead of reading the
file that recorded the decision. The fix is to read all three where they exist —
the paper section, the upstream `rgfn/` code, AND our own `Logs/` experiment log
+ analysis script (`compare_systems.py`) — and to treat "smoke test passed" as
"wiring works", never "science is correct". The discrimination test above
operationalises the latter.

## Update (2026-06-25, cont.): Logs 006 + 007 confirm the Vina ΔT2−T1 choice

After the metric correction above, experiment logs **006** (six-way signal
ablation) and **007** (molecular-weight control) were written and independently
confirm `ddb1_dvina` (Vina Tier2−Tier1, lower-is-better) is the right oracle
signal — so **no reward-signal code change was needed**; the implementation was
already aligned. What changed is documentation + the guard:
- 006: ranked all six candidate signals on the entry-002 poses; **Vina ΔT2−T1
  wins, AUROC 0.946** (Cohen's d 2.38); its Youden cut −1.58 matches our −1.5.
  CNNaffinity differential is 0.850; absolute Vina Tier 1 is 0.69.
- 007: after matching glues/decoys on molecular weight, Vina ΔT2−T1 stays top
  (0.946 → 0.866) while absolute Vina Tier 1 falls below chance (0.38) — the
  differential isn't a size artifact.
- Citations to 006/007 added to: `docking_6td3_oracle.py` docstring,
  `test_oracle_discrimination.py`, the example README, and the gin config.
- Fixed two stale comments that still said `ddb1_dcnnaff` (config + build_seeds).
- Strengthened the discrimination test with an **AUROC check** (asserts > 0.90;
  reproduces 006's 0.946) and an AUROC-based contrast (Vina 0.946 vs CNN 0.850).
  Caught a sign slip while doing so: CNNaffinity is higher-is-better (pK), Vina is
  lower-is-better (energy) — using one orientation for both flips AUROC to
  1−AUROC. Now oriented per metric; both reproduce 006 exactly.

## Deliberate divergences from the publications (validate/revisit on Balam)

1. **Proxy target** — predicts our docking neosubstrate *differential*
   (`ddb1_dvina`, lower-is-better), not AutoDock sEH affinity. The project's
   novel oracle. Signal choice justified by Logs 002/006/007.
2. **Trainable proxy** — `SehMoleculeProxy` is inference-only with frozen
   weights; we add `.fit()`. This is what Alg. 1 requires; the shipped proxy just
   doesn't expose it.
3. **Random init each round** — we rebuild `MPNNet` weights every `fit()` rather
   than annealing from the previous round (avoids compounding drift; revisit).
4. **Label scaling** — we standardise labels at `fit` time and clip predictions
   to `[min_value, max_value]` at inference (mirrors the paper's renormalisation
   to a positive reward and `SehMoleculeProxy`'s `clip(1e-4, 100)`).
5. **Scale** — paper uses `|D_0|=2000` and 200 mols/round; our `D_0` is 408
   (all the validated labels we have). `query_batch_size=200` matches the paper.
6. **Validation split** — paper uses a fixed 3000-mol val set for early stopping;
   we hold out a fraction (`val_fraction`) — matters for small `D`.
7. **Replay buffer reset** — we clear it each round (its priorities go stale once
   `M` is refit). Best-effort, guarded against upstream layout changes.

## Verified locally (Mac, `rgfn` conda env — has torch/rdkit/gin/torch_geometric)

- `py_compile` of all new modules; `import glue` registers every new component.
- Both AL configs parse (after the `Trainer`-import fix below).
- `LearnedGlueProxy` fits on toy data, predicts positive rewards, handles invalid
  SMILES and early-terminal states.
- **Full mock loop end-to-end** via `scripts/active_learning.py` (2 rounds, CPU,
  `WANDB_MODE=offline`): seed `|D_0|=13` → fit → `Trainer.train()` → sample 16 →
  score → accumulate → `|D|=29` → round 2 → `|D|=45` → `top_k.csv` written. EXIT 0.
  RGFN-generated (reaction-assembled) molecules out-scored seeds under the mock
  oracle and rose to the top of the Top-K. (Smoke artifacts deleted, not committed.)
- `Docking6TD3Oracle` imports without gnina and raises a clear `FileNotFoundError`
  when receptors are absent (laptop) — Balam-only by design.

### Two bugs found and fixed during local validation
- **`scripts/active_learning.py` didn't register `Trainer`** → gin "No
  configurable matching 'Trainer'". Upstream `train.py` imports it explicitly;
  added `from rgfn.trainer.trainer import Trainer`.
- **`LearnedGlueProxy._compute_proxy_output` assumed `.molecule`** → crashed on
  `ReactionStateEarlyTerminal` after the loop's post-`fit` `clear_cache()` wiped
  the pre-seeded entry. Now guards early-terminal states → `min_value`.

## NOT verified — needs Balam (gnina + GPU + prepared receptors)

- `Docking6TD3Oracle.score()` against real `6TD3_tier{1,2}.pdbqt` + `crystal_RC8.pdb`
  (git-ignored, absent on laptop). **Reconcile/cross-validate it against
  `dock_cluster.py`** — the two duplicate the docking logic; `dock_cluster.py` is
  the source of truth. Consider unifying them into one shared `dock_batch_6td3()`.
- `scripts/active_learning.py --cfg configs/glue/active_learning_6td3.gin` full run.
- Hyperparameters: `β`, per-round `Trainer.n_iterations`, proxy `max_epochs` /
  `patience`, `query_batch_size` — all first-guess defaults, untuned.
- Reward-scale sanity: with `β=8` and standardised proxy outputs, confirm the
  exponential reward boosting doesn't explode/vanish on the real differential.
- The headline plots reviewers will want: Top-K-vs-oracle-calls curve and a
  random-acquisition baseline (`[bengio2021gflownet]` Fig. 7 analog).

---

# 2026-06-25 (cont.): documentation consolidation

Reduced duplication across the project docs (goal: one home per fact, fewer
chances for versions to drift):
- **Moved `Logs/RESEARCH_CONTEXT.md` → `docs/RESEARCH_CONTEXT.md`** (`git mv`);
  updated all ~14 referencing files (code docstrings, READMEs, CLAUDE.md, configs).
- **Slimmed `Logs/README.md`** to a short intro + pointers. Its duplicated
  "The systems" and "Common methodology" sections were deleted (already covered by
  RESEARCH_CONTEXT's systems table + terminology; 5HXB anchoring specifics remain
  in log 001). Its experiment **Index** moved into `RESEARCH_CONTEXT.md` (links now
  `../Logs/`). Its "Where results live" + "Datasets" moved into `ARCHITECTURE.md`.
- **`ARCHITECTURE.md`** is now the single source for repo layout + data/result
  locations (new "Data, datasets & results" section); RESEARCH_CONTEXT's
  "Where things live" trimmed to a pointer to it.
- Net: systems/terminology live once (RESEARCH_CONTEXT); locations live once
  (ARCHITECTURE); the experiment index lives once (RESEARCH_CONTEXT).
- **Path-corrected the historical logs** (001–005): their `pre-processing/` path
  references are now `research/preprocessing/` (real, clickable paths), so the
  "read these as research/preprocessing/" translation note was **removed** from
  both `RESEARCH_CONTEXT.md` and `CLAUDE.md`. Only verbatim *commit messages* in
  the logs' "Relevant Versions" blocks still say "pre-processing" — left intact,
  since those quote real historical commits.

---

# 2026-06-26: validation/benchmarking scaffold (second axis)

Introduced a **second organizing axis** for our own code — **production pipeline
vs. validation** — on top of the existing upstream/ours boundary. Goal: get the
structure ready for the comparative studies (multiple generators, PMO, our own
benchmarks, VAE-BO, Boltz-2) **before** Balam is back, keeping the validation
layer as separate as possible from the training pipeline. **Scaffolding only — no
implementation code imported yet** (per the user's request).

### Decisions (chosen by the user)
- **One umbrella folder, not two.** Initially scaffolded as two top-level dirs
  (`validation/` for entrants/scorers + `benchmarks/` for suites/harness/results),
  but the split was thin and tightly coupled (the harness only exists to run the
  generators) and produced two `configs/` dirs. Consolidated into a single
  `validation/` holding everything; the "benchmark" idea survives as
  `validation/suites/`. (`benchmarks/` was created and then removed in the same
  session — it was never committed.)
- **Validation oracles separate from in-loop oracles.** Boltz-2 / co-folding /
  high-fidelity checks go in `validation/oracles/` (never in-loop). In-loop
  oracles stay in `glue/oracles/`. Short-MD as a *complementary in-loop* oracle
  (CRBN ceiling, Objective 3) would still go in `glue/`, not here.
- **Baselines = thin adapters + external installs:** each baseline is a thin
  adapter in `validation/generators/<name>/`; heavy upstream code installs via
  `external/setup_<name>.sh` (not vendored).

### The rule that enforces the split
> `validation/` may import from `glue/` and `rgfn/`; the production pipeline
> (`glue/`, `scripts/train.py`, `configs/glue/`) must **never** import from
> `validation/`. The dependency arrow points one way.

### Created (directories + READMEs / `.gitkeep` only)
- `validation/` + `README.md` (boundary rule, full single-folder layout,
  what-goes-where guide, and the "why reviewers care" rationale).
  - `validation/generators/{rgfn,synflownet,fraggfn,vae_bo}/` + `generators/README.md`
    (entrant table, planned common `Generator` interface).
  - `validation/oracles/boltz2/` + `oracles/README.md`
    (in-loop vs. validation-oracle contrast table).
  - `validation/suites/{pmo,glue_suite}/` + `suites/README.md`.
  - `validation/harness/README.md`, `validation/configs/README.md`,
    `validation/results/` (`.gitkeep`).
- `external/setup_{synflownet,fraggfn,vae_bo}.sh` — **placeholder stubs** (valid
  bash, `exit 1` with a TODO; mirror `setup_reinvent.sh` when implemented).

### Docs updated
- `CLAUDE.md`: ownership table now tags `glue/` as *production pipeline* and adds
  a single `validation/` row as *validation*; new "second axis" subsection with
  the dependency rule.
- `docs/ARCHITECTURE.md`: layout tree expanded with the `validation/` subtree;
  new "Two axes" section + "Component flow (validation)" diagram.

### Verified
- `bash -n` on all three new `external/setup_*.sh` stubs (pass).
- Directory tree + placeholder files created as listed; `benchmarks/` removed.

### NOT done / deferred (by design — structure only)
- No `Generator` base class, adapters, harness, suite, or oracle code written.
- `__init__.py` files intentionally omitted until real code lands (each Python
  subtree gets one then, so the harness can import it).
- Config format for `validation/configs/` (gin vs. YAML) not decided yet.
- The setup stubs do not actually install anything.

---

## 2026-06-26 — sEH GPU-docking oracle for the active-learning loop

Added a fast docking oracle so the multi-round AL loop can actually finish (the
CPU-bound 6TD3 gnina oracle was killed by the login-node CPU cap; exp `009`/`010`).

### Created (ours)
- `glue/oracles/docking_seh_oracle.py` — `DockingSEHOracle(GlueOracle)`. Real sEH
  docking via QuickVina2-GPU-2.1. **Composes** upstream `DockingMoleculeProxy` and
  calls its `dock_batch_qv2gpu` directly on SMILES — the docking is byte-for-byte
  the upstream path (no duplication, unlike the 6TD3 oracle). `higher_is_better=
  False` (Vina energy); returns `nan` on failure per the `GlueOracle` contract;
  the heavy proxy is built lazily so the module imports on a laptop.
- `configs/glue/active_learning_seh.gin` — AL loop wiring (mirrors
  `active_learning_6td3.gin`): `LearnedGlueProxy` (MPNN) as `M`, `DockingSEHOracle`
  as `O`, `reaction` env (docking stays out of the inner loop).
- `research/active_learning_seh/` — sEH-specific run tooling (sibling of
  `research/preprocessing/`; bundles scripts + submit + README like the
  `pose_selection_ablation/` precedent): `make_seh_seed.py` (builds `D_0` by
  sampling the untrained RGFN policy and docking → `seed_seh.csv`; reports
  s/mol), `validate_seh_oracle.py` (quick live smoke), `submit_al_seh.sh` (Balam
  compute-node run; cuda module + boost libs; regenerates a 300-mol seed on
  `$SCRATCH`), `README.md`. The generic `scripts/active_learning.py` entry point
  stays in `scripts/` (shared with 6TD3).
- `experiments/active_learning_seh/seed_seh.csv` — committed starter seed.

### Edited (ours)
- `glue/oracles/__init__.py` — export/register `DockingSEHOracle`.
- `docs/RESEARCH_CONTEXT.md` — index row 010 + Objective 1 note.

### Verified (on balam-login01 A100, `rgfn` env)
- `py_compile`; import via `glue.registry`; gin parse of the config + class
  resolves by name.
- **Live GPU dock**: aspirin −6.30 (matches the recorded sEH validation),
  ibuprofen −6.90, caffeine −6.50; invalid SMILES → `nan`; ~3.7 s/mol.
- **Seed gen**: 36/40 RGFN-sampled molecules docked; dataset loads |D_0|=36;
  proxy/oracle sign check passes (both `higher_is_better=False`).
- `bash -n` on `submit_al_seh.sh`.

### NOT done / deferred
- The **full multi-round loop** has not been run yet (needs a compute node;
  `submit_al_seh.sh` is ready). No top-k-vs-oracle-calls curve / random baseline.
- Oracle outputs not yet cross-checked against an actual
  `configs/rgfn_seh_docking.gin` run on the same molecules (numbers should match).

---

## 2026-06-26 — per-phase timing for the active-learning loop

Experiment 009 could not say where the loop spent its time (only eyeballed
per-step rates; the run died at the oracle step with no number for it). Added
intrinsic, always-on phase timing.

### Added (ours)
- `glue/active_learning/timing.py` — `PhaseTimer`: a context-manager that times
  the four per-round phases (`fit_proxy`, `train_gfn`, `sample_batch`,
  `oracle_score`). On each phase exit it prints (`[AL]`), logs to the trainer
  logger under the `timing` prefix (→ wandb), and **appends** to
  `<run_dir>/active_learning/phase_timings.csv`. The append-before-next-phase
  ordering means a mid-run crash (the SIGXCPU that ended exp 009) still leaves a
  record of every phase that finished. `report_total()` ranks phases by share of
  total wall-clock — the "where to save time" answer.

### Edited (ours)
- `glue/active_learning/loop.py` — wrapped the four phases in `timer.phase(...)`,
  added `report_round`/`report_total`. No behavioural change to the algorithm;
  recorded via `finally`, so timing is robust to a raising phase.

### Verified (laptop, no heavy deps)
- `py_compile` on both files. Standalone smoke test of `PhaseTimer`: `_fmt`
  formatting, CSV append, crash-phase recorded via `finally`, share-ranked
  summary.

### NOT done / deferred
- Not yet run inside a real loop on Balam (needs the `rgfn` env / GPU). The four
  phase labels and their wiring are unverified against a live `Trainer.train()`.
- Overhead is negligible (one `perf_counter` + a CSV append per phase, 4/round),
  but unconfirmed under the real loop.

---

## 2026-06-26 — directory reorg: experiments/ (per-run), data/ (inputs), flat scripts/

Reorganized the top-level layout for clarity (one home per concept). No `glue/`
or `rgfn/` logic changed; this is moves + path-rewiring + docs.

### Moved / merged
- **`research/` dissolved into `experiments/`**, now **grouped by type, one dir
  per run**: `experiments/{active_learning,oracle_validation,ablations}/<run>/`.
  - `research/active_learning_{seh,6td3}/` → `experiments/active_learning/{seh,6td3}/`
    (merged with the matching `experiments/active_learning_*` seeds/outputs).
  - `research/preprocessing/{docking_6td3, docking_gnina→docking_crbn, clean*.py,
    compare_systems.py}` → `experiments/oracle_validation/…`.
  - `research/preprocessing/{pose_selection_ablation→pose_selection,
    full_comparison→sixway, full_comparison_mw→mw}` → `experiments/ablations/…`.
  - Old flat AL output dirs (`active_learning_mock`, `al_probe→probe`,
    `active_learning_6td3_inner→6td3_inner`) → under `experiments/active_learning/`.
  - `experiments/oracle_validation/` chosen over `validation/` to avoid clashing
    with the top-level `validation/` benchmarking layer.
- **`models/` + `research/preprocessing/test-data/` folded into `data/`** (the
  single inputs dir): `data/models/`, `data/validation-molecules/` (renamed from
  "test-data" — they're validation *inputs*, not fixtures). `data/` keeps its name
  because upstream hardcodes `data/chemistry.xlsx` + `data/targets/`.
- **`scripts/hpc/` removed**; the one generic `submit.sh` → `scripts/submit.sh`.
  Run-specific submit scripts already live with their experiment.

### Rewired
- All `git mv`/`mv` done so git detected pure **renames** (history preserved).
- Code paths: `glue/oracles/docking_6td3_oracle.py` `_DOCK_DIR`,
  `glue/tests/test_oracle_discrimination.py` (`SEED_CSV`/`DOCK_DIR`), the moved
  docking/analysis scripts (`../test-data`→`data/validation-molecules`,
  `docking_gnina`→`docking_crbn`, `models/`→`data/models/`), `clean*.py` in/out dirs.
- Configs: 5 AL `seed_csv` paths repointed to `experiments/active_learning/<run>/`.
- `scripts/active_learning.py`: run_name now groups outputs under
  `experiments/active_learning/<run>/<ts>/` so outputs co-locate with each run's
  committed material.
- `.gitignore`: replaced the blanket `experiments/*` ignore with a **pattern-based**
  scheme — ignore timestamped run-output dirs (`experiments/**/YYYY-MM-DD_*/`),
  force-track `seed_*.csv` / `*_results.csv`; updated `models/`→`data/models/`,
  `test-data`→`data/validation-molecules`, cluster_out/violins paths.

### Verified (balam-login01, `rgfn` env)
- `git status` shows clean renames; no committed file under a timestamped run dir;
  `git check-ignore` confirms run outputs ignored / seeds+results+code tracked.
- `py_compile` of all moved scripts; `import glue`; all 5 AL configs parse **and
  their seed paths resolve**; `pytest glue/tests/test_oracle_discrimination.py` (2
  passed); `bash -n` on all 6 submit scripts.
- READMEs updated/added: `experiments/README.md` (structure guide), per-group/run
  READMEs, `data/README.md` + `data/{models,validation-molecules}/README.md`,
  `scripts/README.md`; plus `CLAUDE.md`, `docs/ARCHITECTURE.md`, README/context.

### NOT done / deferred
- Balam-side `$SCRATCH` paths (e.g. where the user keeps `.cif`/receptors) may need
  the same `models/`→`data/models/` move on Balam; the repo-relative refs are fixed.
- The 6TD3 docking-oracle pdbqt receptors live (git-ignored) at the new
  `experiments/oracle_validation/docking_6td3/` path on this login node; confirm
  they're present at that path on Balam compute nodes before the next 6TD3 run.

---

## 2026-06-27 — `top_k` deliverable sort fix (`glue/datasets/oracle_labeled.py`)

`OracleLabeledDataset.top_k()` sorted `reverse=True` (largest label first), but
oracle labels are *more-negative = better* (`GlueOracle.higher_is_better = False`,
for both the 6TD3 `ddb1_dvina` differential and the sEH Vina energy), so the
"Top-K deliverable" returned the **worst** molecules. Found while reviewing the
first multi-round 6TD3 run (Logs/011, job 69445), whose `top_k.csv` listed +0.68
at rank 1. Fixed to sort ascending. Scope: deliverable file only — training's
reward/proxy path is separate and was correctly oriented (the run's generated
molecules dock *well*). Regenerated `top_k.csv` for that run from
`dataset_round_003.csv` (best now −4.92).

**Verified:** `py_compile` of the module. **Not verified here:** no unit test
covers `top_k`, and gin/rdkit/conda aren't on this node, so the loop wasn't
re-run; the regenerated `top_k.csv` was produced by replicating the (trivial)
ascending-sort logic inline. A `test_top_k_orientation` unit test would be cheap
insurance.

---

## 2026-06-28 — docking oracle sub-step timing (`OracleStepTimer`)

Added per-substep wall-clock timing inside the expensive docking oracle so the
`oracle_score` phase (35% of loop wall-clock, Logs/011) is no longer a black box.

- **New** `glue/oracles/step_timing.py::OracleStepTimer` — disabled-by-default
  (silent no-op) companion to `glue/active_learning/timing.PhaseTimer`. Appends
  `(round, step, seconds, n_molecules)` rows on each sub-step's completion
  (crash-safe), prints a `[dock]` line live.
- `glue/oracles/docking_6td3_oracle.py` — new `enable_step_timing(csv_path)`;
  `score()`'s four sub-steps wrapped: `embed` / `tier2_dock` / `pose_select` /
  `tier1_rescore`. **Docking logic unchanged** (like-for-like with Logs/011).
- `glue/active_learning/loop.py` — opt-in hook: if the oracle has
  `enable_step_timing`, the loop points it at
  `<run>/active_learning/docking_timings.csv`. Loop stays oracle-agnostic; mock /
  sEH oracles (no hook) are unaffected (`getattr` + `callable` guard).

**Verified:** `py_compile` (3 files); in the `rgfn` env `import glue` resolves the
oracle, disabled timer is a confirmed no-op, enabled timer writes the expected
4-column CSV with recoverable s/mol; `bash -n` on the new submit script.
**Not verified here:** a real multi-round run with the timer live — that is
**Logs/012** (job 69450, queued this session). The sEH oracle does **not** yet
implement `enable_step_timing`; add it there when that loop is next run.

---

## 2026-06-29 — FragGFN baseline (first `validation/` entrant) + the oracle bridge

Implemented the **FragGFN** non-synthesizable baseline (Recursion's `gflownet`
fragment environment), the foil for RGFN's synthesizability claim (Objective 4/5;
`Logs/015`). This is the first real code under `validation/`, so it sets the
pattern for future entrants.

**Key constraint that shaped the design.** Recursion's `gflownet` (pinned
`da999404`) hard-pins **python 3.10 / torch 2.1.2 / torch-geometric 2.4.0**,
incompatible with the `rgfn` env (3.11 / torch 2.3 / dgl). So FragGFN runs in its
**own `fraggfn` conda env**, and — since `glue/__init__` eagerly imports the
`rgfn`-dependent registry — it **cannot import `glue`/`rgfn` at all**. The shared
docking oracle is therefore reached **across the env boundary** via a CLI bridge.

**The oracle bridge — `scripts/score_batch.py` (generic, reusable).** Runs in the
`rgfn` env; scores a SMILES file with a named glue oracle and writes the **standard
candidate-dataset format** (`glue.datasets.candidates` + `glue.metrics.dataset_metrics`,
the same code RGFN's `SuggestionLog` uses), `has_route=0` for non-synthesizable
entrants. Per-round shards + `--finalize` give crash-safety across the per-round
subprocess calls. This is now the single shared scoring standard every baseline
will use; it lives in `scripts/` because it's pipeline-wide, not FragGFN-specific.

**Added:**
- `external/setup_fraggfn.sh` — builds the `fraggfn` env (py3.10), clones gflownet
  to `external/gflownet/` (git-ignored via the new `external/*/` rule — clones are
  **not vendored**), installs cu118 torch + pyg wheels + gflownet, pins `numpy<2`.
- `scripts/score_batch.py` — the oracle bridge (above).
- `validation/generators/fraggfn/{proxy,task,al_loop,run_fraggfn_al}.py` + README —
  the thin adapter: `AtomMPNNProxy` (the Bengio-2021 MPNN reused from gflownet's
  own `bengio2021flow`, same architecture as `LearnedGlueProxy`), `FragGFNTrainer`
  over `FragMolBuildingEnvContext`, and `FragGFNActiveLearningLoop` mirroring
  `glue/active_learning/loop.py`. Deliberately does **not** import `glue`/`rgfn`.
- `validation/{__init__,generators/__init__}.py` — make the subtree importable
  (the convention noted in `validation/README.md`).
- `validation/configs/fraggfn_6td3.yaml` (+ `fraggfn_smoke.yaml`) — budget matched
  field-for-field to `configs/glue/active_learning_6td3_gpu.gin`.
- `experiments/active_learning/fraggfn_6td3/{README.md,submit_fraggfn_6td3.sh}` —
  Balam run scaffolding; **reuses** `../6td3/seed_6td3.csv` (identical `D_0`).

**Verified (login node A100, this session):** `fraggfn` env builds + imports
gflownet; `mol2graph` width 71 / `FRAGMENTS` 72; `py_compile` all new Python in
both envs; `bash -n` the submit + setup scripts; YAML configs load; the bridge
mock run produces a conformant standard dataset (`validate_candidate_dataset` →
no issues, `has_route=0`); and a **full end-to-end CPU dry run** of the loop
(2 rounds, mock oracle via the bridge) — fit M → train fragment-GFN → sample →
cross-env score → accumulate → finalize → Top-K.
**Not verified here:** the real 3-round **GPU docking** run (Balam compute node;
`sbatch experiments/active_learning/fraggfn_6td3/submit_fraggfn_6td3.sh`) and the
FragGFN-vs-RGFN comparison numbers — pending, tracked in `Logs/015`.

## Active-learning loop robustness (post job 69481, 2026-06-30)

Two fixes prompted by the entry-014 GPU run, which was killed by a Balam node
failure (balam009 returned all-`no_pose` in round 1, then died mid-round-2):

- **All-NaN-round abort guard** (`glue/active_learning/loop.py`): if an entire
  round's oracle batch comes back NaN (no labels), the loop now raises a clear
  `RuntimeError` instead of silently refitting M on an unchanged D and burning
  another full GFN training run. The round's provenance (`suggestions/`,
  `dataset_round_NNN.csv`) is written *before* the abort, so the failure is fully
  inspectable. Triggers only on a wholesale oracle failure (non-empty batch, zero
  valid scores) — partial failures pass through as before.
- **Manifest provenance** (`SuggestionLog` / `CandidateDataset` / loop / driver):
  `ActiveLearningLoop` gained `system` + `seed` args, forwarded into the candidate
  manifest (were `null`). `scripts/active_learning.py` binds
  `ActiveLearningLoop.seed` from `--seed`; `configs/glue/active_learning_6td3_gpu.gin`
  sets `ActiveLearningLoop.system='6td3'`.

Verified on the Trillium login node (`source ~/bin/rgfn-smoke-env.sh`): `py_compile`;
`glue` import + `ActiveLearningLoop` signature carries `system`/`seed`; gin parse of
`active_learning_6td3_gpu.gin` with the `--seed` binding; and a `SuggestionLog`
round-trip confirming `system`/`seed` land in `manifest.json`. Not verified: a full
multi-round loop run (needs a GPU compute node; Balam submission unavailable).

## RxnFlow synthesizable baseline entrant (exp 016, 2026-06-30)

Added a **second** synthesis-aware benchmark entrant — **RxnFlow** (`[seo2024rxnflow]`,
ICLR 2025; reaction-template + building-block GFlowNet, action-space subsampling) — as
the *synthesizable peer* to RGFN, opposite the non-synthesizable FragGFN foil. (Briefly
scoped as SynFlowNet first, then switched to RxnFlow; the SynFlowNet bib/reference/PDF
work was reverted.) RxnFlow is built on a bundled Recursion `gflownet`, so it drops into
the FragGFN two-env precedent almost verbatim. New / changed:

- `validation/generators/rxnflow/` — thin adapter mirroring `fraggfn/`:
  `proxy.py` (re-exports FragGFN's `AtomMPNNProxy` so `M` is identical across entrants),
  `task.py` (`RxnFlowTask`/`RxnFlowGlueTrainer` over RxnFlow's `BaseTask`/`RxnFlowTrainer`,
  reward = `M`, constant β, `num_workers=0`), `al_loop.py`
  (`RxnFlowActiveLearningLoop` = `[bengio2021gflownet]` Alg.1 + the all-NaN guard +
  `extract_route`), `run_rxnflow_al.py`, `README.md`, `__init__.py`.
- `scripts/score_batch.py` — generalized the shared oracle bridge to be **route-aware**:
  new `--routes` (per-round JSONL `{"smiles":…, …route…}`) stored next to the shards and
  joined onto candidates by SMILES on `--finalize` → synthesizable entrants emit
  `has_route=1` + `routes.jsonl`. FragGFN's path (no `--routes`) is byte-unchanged; the
  hardcoded "FragGFN/non-synthesizable" finalize note is now generator-aware.
- `validation/configs/rxnflow_{6td3,smoke}.yaml` — budget/oracle/proxy matched
  field-for-field to `configs/glue/active_learning_6td3_gpu.gin`; a `rxnflow:` block
  carries the generator knobs (`env_dir`, `max_reactions`, `action_sampling_ratio`).
- `experiments/active_learning/rxnflow_6td3/{README.md,submit_rxnflow_6td3.sh}` — Balam
  run scaffolding; **reuses** `../6td3/seed_6td3.csv` (identical `D_0`).
- `external/setup_rxnflow.sh` — `rxnflow` conda env (py3.12, torch 2.5.1+cu121) + clone +
  `pip install -e` + env-dir prep (public ZINCFrag blocks + `templates/hb_edited.txt`).
- References: reverted SynFlowNet; added `[seo2024rxnflow]` to `references.bib` +
  `Logs/references/README.md`; PDF at `pdfs/seo2024rxnflow.pdf`.

**Verified (this session, no heavy stack locally):** `py_compile` all new Python;
`bash -n` the setup + submit scripts; both YAMLs load and their `loop`/`oracle` blocks
match `fraggfn_6td3.yaml`; the route-aware bridge round-trips with **and** without
`--routes` (`validate_candidate_dataset` → no issues; `has_route` flips 1↔0 correctly).
**Flagged for Balam validation (heavy stack is cluster-only):**
- the RxnFlow upstream API — `rxnflow.config.Config` schema + `env_dir` field path,
  `RxnFlowTrainer`/`BaseTask` signatures (`setup_task`, `set_default_hps`), and the
  per-action attributes `extract_route` reads (block / reaction template / product).
  These are written best-effort from the RxnFlow docs/examples; confirm against the
  cloned repo and tighten `extract_route` so routes are chemically faithful.
- the cu121 (rxnflow) vs cu118 (rgfn bridge) coexistence — separate procs/envs, torch
  cu121 wheels are RPATH-self-contained; needs a GPU driver ≥525 (CUDA 12.1) on the node.
- `external/setup_rxnflow.sh` step 4 (env-dir prep) has a `TODO` for the exact RxnFlow
  building-block preprocessing command (per the cloned repo's `data/README.md`).
- finish: pin `RXNFLOW_COMMIT`, run the CPU mock smoke, then the 3-round GPU run; the
  RGFN-vs-RxnFlow-vs-FragGFN comparison numbers — tracked in `Logs/016`.

---

## 2026-06-30 — Synthesizability metric (AiZynthFinder + SA) in the harness

Added the first `validation/harness/` evaluation metric: a post-hoc
**synthesizability** report over a candidate dataset, the analogue of the `AiZynth`
column in `[koziarski2024rgfn]`/`[gainski2025scent]` and RxnFlow's "Synthesizability %"
(`[seo2024rxnflow]`). It runs uniformly on **every** entrant because they all emit the
one standard candidate-dataset format (`docs/CANDIDATE_DATASET_FORMAT.md`).

Design choices (confirmed with Mark before building):
- **Stock/templates:** AiZynthFinder's standard **public dataset** (USPTO expansion +
  ZINC in-stock) — reproducible and directly comparable to the published RGFN/SCENT
  numbers. Stock/expansion/filter keys are CLI-overridable.
- **Metrics:** the full paper set — fraction-solved (AiZynth success rate), mean #steps
  over solved, and the SA-score distribution.
- **Isolation:** AiZynthFinder in its own `aizynth` conda env
  (`external/setup_aizynthfinder.sh`), invoked post-hoc as a standalone CLI — same
  "one env per tool, one on-disk standard" pattern as the fraggfn/rxnflow oracle bridge.

Added:
- `validation/harness/synthesizability.py` — reads `candidates.csv`/`manifest.json`
  **directly** (csv/json, no `glue` import, so it runs in the lean `aizynth` env);
  parallel AiZynth driver (one finder per worker); dedups by canonical SMILES; writes
  `synthesizability.csv` (per-molecule) + `synthesizability_summary.json` (aggregate,
  incl. the by-construction self-report cross-check). CLI: `--dataset/--config/--nproc/
  --top-k/--no-dedup/--time-limit/--iteration-limit`.
- `validation/harness/__init__.py` (kept import-light on purpose).
- `validation/harness/test_synthesizability.py` — dependency-free unit tests
  (monkeypatch RDKit + AiZynth) for dedup / top-k / scatter-back / aggregation.
- `external/setup_aizynthfinder.sh` — `aizynth` env + `download_public_data` + smoke.
- References: `[genheden2020aizynth]` + `[ertl2009sascore]` in `references.bib` and the
  references `README.md` (new "Evaluation — synthesizability metrics" section).
- `validation/harness/README.md` — documents the landed metric + run command.

**Verified (Mac/login, no heavy stack):** `py_compile` the module/test/init; `bash -n`
the setup script; `python -m unittest validation.harness.test_synthesizability` (3/3
pass — dedup denominator is unique-molecule level, top-k picks best-by-score, files
written).
**Flagged for `aizynth`-env validation (not runnable without the install):**
- the AiZynthFinder ≥4 API surface the driver uses — `AiZynthFinder(configfile=…)`,
  `.stock/.expansion_policy/.filter_policy.select(key)` + `.items`, `target_smiles`,
  `tree_search()`/`build_routes()`, and the `extract_statistics()` keys (`is_solved`,
  `number_of_steps`, `number_of_solved_routes`, `top_score`) — written from the docs;
  confirm against the installed version and adjust key names if the schema differs.
- `download_public_data` output layout (that it writes `config.yml` with `zinc`/`uspto`
  keys) — the `--stock/--expansion/--filter` defaults assume this; the driver falls back
  to the first available key if a name is absent, but verify on a real install.
- timing/throughput on the cluster (retrosynthesis is seconds–minutes/molecule); decide
  whether to default to `--top-k 500` like the papers rather than the full set.

## SCENT cost-aware baseline entrant (exp 017, 2026-06-30)

Added the **cost-aware** benchmark entrant — **SCENT** (`[gainski2025scent]`,
arXiv:2506.19865). SCENT is a **fork of RGFN from the same lab** (its package is named
`rgfn`, same py3.11/torch2.3/dgl/gin stack, same `rgfn.api`/`Trainer`/reaction env) that
adds Recursive Cost Guidance + Exploitation Penalty + Dynamic Library. RGFN is one of
SCENT's *own* baselines, so this is the tightest of the three comparisons.

Design choices (confirmed with Mark before building, via AskUserQuestion):
- **Variant:** *full* SCENT (all three mechanisms ON) — the paper's headline method.
- **Library:** SCENT's **SMALL** building-block set, which *ships* the building-block
  prices (`fragment_to_real_cost.json`) + reaction yields (`templates_yields.csv`) the
  cost model needs — the only way to run *true* cost-aware SCENT without sourcing Enamine
  pricing. Reward/oracle/seed/budget/β stay identical to the RGFN run; only the generator
  + its blocks differ (inherent to a different generator).

Key difference from the FragGFN/RxnFlow precedent — **why the adapter is gin-driven, not
programmatic:** SCENT is configured entirely through its own gin (cost guidance,
exploitation penalty, dynamic library, env, policies are all gin-wired singletons), so
reproducing it programmatically (the `gflownet` `Config` route fraggfn/rxnflow take) is
infeasible. Instead the runner parses a gin AL config that `include`s SCENT's
`scent_base.gin` and swaps the sEH proxy for ours — mirroring `scripts/active_learning.py`.

**Namespace hazard (the load-bearing subtlety):** SCENT's package is *named* `rgfn`, so in
the `scent` env `import rgfn` must hit SCENT's installed package, not our repo-local
`rgfn/`. The runner therefore (a) never puts the repo root on `sys.path` (uses **sibling**
imports `import proxy`/`import al_loop`/`import route`, off the runner's own dir), and (b)
`chdir`s into `external/scent` so SCENT's relative gin includes + `data/small/*` resolve —
with every *our*-side path (config, seed, run dir, the repo root the bridge runs in) made
absolute first. This is why the adapter modules use plain (not package-relative) imports
and are NOT imported by `validation/generators/scent/__init__.py`.

New:
- `validation/generators/scent/` — `proxy.py` (`LearnedDockingProxy`: line-for-line the
  same MPNN as `glue.proxies.LearnedGlueProxy`/fraggfn's `AtomMPNNProxy`, here subclassing
  **SCENT's** `CachedProxyBase` + importing **SCENT's** bundled `MPNNet`/`mol2graph`),
  `route.py` (self-contained copy of `glue.active_learning.route`), `al_loop.py`
  (`ScentActiveLearningLoop` = `[bengio2021gflownet]` Alg.1, gin Trainer + bridge scoring
  + route JSONL + all-NaN abort guard; `LabelStore` = `D`), `run_scent_al.py` (entry
  point), `README.md`, `__init__.py`.
- `validation/configs/scent_6td3.gin` (+ `scent_smoke.gin`) — `include`s `scent_base.gin`
  + `small.gin`; budget/seed/β matched to `configs/glue/active_learning_6td3_gpu.gin`;
  oracle bridge params (the same `Docking6TD3GpuOracle` knobs) passed to the loop.
- `external/setup_scent.sh` — clone `koziarskilab/SCENT@af1fee5` + `scent` conda env
  (py3.11.8/torch2.3/dgl2.2.1, `pip install -e .`) + import smoke test.
- `experiments/active_learning/scent_6td3/submit_scent_6td3.sh` — Balam submit (OpenCL
  gate, `--exclude=balam008`, `$SCRATCH` outputs; `scent` env, bridge re-enters `rgfn`).
- References (`[gainski2025scent]`), generator table, `Logs/017`.
- The shared **route-aware** bridge (`scripts/score_batch.py`) is reused **unchanged**
  (the RxnFlow `--routes` extension already covers synthesizable entrants).

**Verified (Trillium login, no heavy stack):** `py_compile` all SCENT Python; `bash -n`
both shell scripts; static gin-integrity check — every `include` in `scent_{6td3,smoke}.gin`
resolves into the clone, every symbol the adapter imports exists in SCENT's source, and
`data/small/{fragments,templates,fragment_to_real_cost,templates_yields}` are present;
confirmed `CachedProxyBase.compute_proxy_output` wraps a `List[float]` into `ProxyOutput`
(matches `LearnedDockingProxy`'s return), `Trainer.{train,close,train_forward_sampler,
train_batch_size}` + the `Trajectories`/`Reaction*` API the loop binds to all exist.
**Resolved on Balam (job 69513, `COMPLETED` 2 h 03 m) — three bring-up fixes:**
- `import rgfn` died on `pkg_resources` (wandb dep; py3.11 env had setuptools≥81 which
  removed it) → pinned `setuptools<81` (now in `setup_scent.sh`).
- gin "No configurable matching 'Trainer'" → `rgfn/trainer/__init__` doesn't import
  `trainer.py`; runner now does `from rgfn.trainer.trainer import Trainer` (as SCENT's own
  `train.py` does).
- SCENT's sEH-tuned `ScaffoldCost` metric calls `.molecule` on any state with `proxy>8`
  without a terminal-type guard; our lower-is-better proxy gives invalid states `+clip=+10`
  → crash. Overrode `train_metrics` to a sign-safe subset (kept `@TrajectoryCost`); these
  are wandb-only diagnostics, so no effect on generation or the comparison.
- Confirmed working: runner's `chdir`-into-clone + sys.path handling resolves `import rgfn`
  to SCENT (not our repo-local `rgfn/`); the multi-line `oracle_args` dict literal parses;
  the cost-guided graph constructs from `data/small/*`; `has_route=96/96` with tree routes.
- Result: SCENT matches RGFN/FragGFN on glue quality (median dvina −2.12, best −5.81),
  fully synthesizable, and its cost-awareness appears to counteract the size drift — full
  numbers in `Logs/017`.

## sEH benchmark: GPU-pipeline parity + validation-baseline coverage (2026-06-30)

**Why:** sEH is the canonical GFlowNet docking benchmark ([bengio2021gflownet],
RGFN's `configs/rgfn_seh_docking.gin`); we want it as a *reproduction check* before
trusting the harness on the novel 6TD3 glue task. The sEH oracle (`DockingSEHOracle`,
exp 010) and a loop config already existed, but the config lagged the 6TD3 GPU
pipeline and no validation baseline could target sEH. This entry brings sEH to parity.

**Re-validated live (Trillium login H100):** `DockingSEHOracle` (QuickVina2-GPU sEH
docking) reproduces aspirin −6.3 (known value), ibuprofen −6.9, anthracene −8.5,
caffeine −6.5, at ~0.9 s/mol; invalid SMILES → `nan`. The Balam-built QV2-GPU stack
runs unchanged on Trillium (shared FS) via `~/bin/rgfn-smoke-env.sh`. The full
bridge path `scripts/score_batch.py --oracle docking_seh` was also exercised
end-to-end: scores aspirin −6.3 and writes the standard candidate dataset
(`candidates.csv` + `manifest.json`, `has_route=0`).

Changed:
- `configs/glue/active_learning_seh.gin` — GPU-pipeline parity with
  `active_learning_6td3_gpu.gin`: added `ActiveLearningLoop.system='seh'`,
  `Trainer.valid_every_n_iterations`, and a corrected metric block. The inherited
  `@NumScaffoldsFound` (from `rgfn_base.gin`) used positive thresholds `[5,6,7,8]`
  that assume the upstream `SehMoleculeProxy`'s clipped non-negative reward; OUR
  proxy `M` predicts the RAW Vina energy (negative, lower-is-better), so that metric
  was silently inert. Swapped in `@SafeNumScaffoldsFound` with
  `proxy_higher_better=False` and Vina-scale cutoffs `[-8,-9,-10,-11]` grounded in
  the seed `D_0` distribution (median −7.0, Q1 −8.4, strong binders ≤ −10).

New (validation baselines can now target sEH — all three implemented entrants):
- `validation/configs/{fraggfn,rxnflow}_seh.yaml`, `validation/configs/scent_seh.gin`
  — sEH analogues of the `*_6td3` configs, budget/seed/β matched to
  `active_learning_seh.gin`; oracle switched to `docking_seh` (single-target Vina
  binding, `higher_is_better=false`; args `exhaustiveness/docking_batch_size/
  n_conformers/n_gpu`, **no** `num_modes`/dvina), threshold −8.0, seed
  `experiments/active_learning/seh/seed_seh.csv` (250 labels). `scent_seh.gin` is a
  surgical mirror of the validated `scent_6td3.gin` (verified by diff).
- `experiments/active_learning/{fraggfn,rxnflow,scent}_seh/` — submit scripts +
  READMEs, mirroring the `*_6td3` Balam submit scripts (OpenCL gate, `--exclude=balam008`,
  `$SCRATCH` outputs; loop in the generator's env, bridge re-enters `rgfn`).
- Fixed a stale "36-molecule starter" claim in the sEH README + submit script (the
  committed `seed_seh.csv` actually holds 250 labels).

**Verified (Trillium H100 + static):** live `DockingSEHOracle` smoke + live
`docking_seh` bridge run (above); `active_learning_seh.gin` re-parses with all
`@references` resolved (`@SafeNumScaffoldsFound` included); `py_compile` of touched
Python; `bash -n` on all three new submit scripts; YAML schema parity of
`{fraggfn,rxnflow}_seh.yaml` vs their `_6td3` siblings; `scent_seh.gin` body diff vs
`scent_6td3.gin` shows only the intended budget/oracle/threshold/system changes.
**Flagged for `scent`/`rxnflow`-env validation (not runnable without those installs):**
the gin/yaml end-to-end parse under each baseline's own env, and the real multi-round
GPU runs of the RGFN sEH loop + the three baselines on a Balam compute node.

## RxnFlow entrant — Balam validation + first run (exp 016, 2026-06-30, job 69518)

Installed and validated the RxnFlow entrant on Balam (login A100 + compute), resolving
every API/stability uncertainty flagged above. Adapter fixes made against the real
cloned repo:
- **Route extraction rewritten** to consume RxnFlow's real trajectory format
  (`ctx.read_traj` → `("FirstBlock", block)` / `("UniRxn", template)` /
  `("BiRxn", template, block)` tuples), not the guessed action-object attributes; and
  sampling switched to RxnFlow's own `algo.graph_sampler.sample_inference` +
  `ctx.object_to_log_repr`/`read_traj` (the `RxnFlowSampler` path). Verified: 8/8 samples
  render faithful multi-step routes.
- **Config path** `model.num_layers` → `model.graph_transformer.num_layers` (RxnFlow's
  ModelConfig has no `num_layers`).
- **`gdown`** added to `setup_rxnflow.sh` (imported by `rxnflow.utils.misc`, missing from
  RxnFlow's pyproject).

**The load-bearing finding — template set vs. library size.** With `hb_edited.txt` (71
templates) on the bundled 10k ZINCFrag debug library, training goes **non-finite**
(NaN policy logprobs): the library is too sparse, so many (template, reactant) states
have no compatible second block → all-masked action categories → `log(0)`. Confirmed it
was the env, not our code, by reproducing the NaN with RxnFlow's **own** stock QED task
(14/25 steps non-finite), and confirmed the fix by rebuilding with the default
**`real.txt` (109 Enamine-REAL templates)** → QED control **0/25** and our
constant-β=8 proxy loop **0/25**. So: `setup_rxnflow.sh` now defaults to `real.txt`;
switch to `hb_edited.txt` only with the full ~200k ZINCFrag library.
- **RxnFlow defaults are the stable regime**, kept as-is: `action_subsampling.sampling_ratio=0.02`,
  `train_random_action_prob=0.1`. Removed our `RxnFlowGlueTrainer.set_default_hps`
  override that forced `train_random_action_prob=0.0` (the RGFN/FragGFN value) — RxnFlow
  needs positive exploration or its TB loss diverges. Our shared-fairness levers (proxy
  `M`, oracle, seed, budget, constant β=8) are unchanged; only RxnFlow's native
  exploration/subsampling knobs differ (per-generator, as intended).
- **cu121/cu124 vs cu118 coexistence confirmed:** rxnflow torch 2.5.1+cu124 runs GPU fine
  with the `cuda/11.8.0` module + rgfn env's `LD_LIBRARY_PATH` present (needed for the
  bridge's dgl). Driver 580 on balam004 covers CUDA 12.

**Validated end-to-end (login A100):** full cross-env mock smoke — 2 rounds, RxnFlow
train (finite) → sample+routes → bridge (rgfn, mock) → standard dataset. Finalized
dataset conformant, `has_route=1`, `routes.jsonl` = 12/12 with real building blocks +
1–2 reaction steps. **Real run submitted: job 69518** (`al_rxnflow_6td3`, balam004,
3 rounds × 32-mol GPU 6TD3 docking), running alongside the matched RGFN GPU run (69517).
Results → `Logs/016`.

**COMPLETE (job 69518, 2026-07-01 00:40, 51 min, exit 0:0).** All 3 rounds ran on the
real GPU docking oracle: 96/96 candidates scored, **all 96 with 2-step synthesis routes**
(`has_route=1`), conformant standard dataset. Best `dvina` −4.22, median −1.19, 30/96
(31%) ≤ −2.0, diversity 0.86–0.90. Training stayed finite over all 900 steps (the
`hb_edited`→`real.txt` fix held). **Matched-oracle RGFN GPU run (69517) also COMPLETE**
(3h14m; RGFN in-loop GPU training ~1h/round vs the bridge baselines' ~15min), giving a
clean same-oracle GPU three-way in `Logs/016`. Two findings: (1) on glue score all three
are same-league (RGFN median/mean −2.14 best, FragGFN best single hit −4.86 & 54% past
−2.0, RxnFlow behind) — RGFN does not out-dock the non-synth foil; (2) the real split is
drug-likeness — RGFN & FragGFN both bloated/low-QED (MW 665–720, QED 0.11–0.15, RGFN
Lipinski 2/96), only RxnFlow is both synthesizable and drug-like (MW 489, QED 0.36).
Lead: add a QED/property term (or tighter block lib) to RGFN. Full numbers in `Logs/016`.

## GPU-oracle-in-loop fix + pre-flight gate (exp 014, job 69517, 2026-07-01)

The first *complete* GPU-oracle active-learning run (job 69517: 3 rounds, |D|
408→483) required fixing a real bug that three prior attempts hit — round-1
docking returning all `no_pose`:

- **Root cause: GPU-memory contention.** QuickVina2-GPU runs in a subprocess and
  allocates GPU memory via OpenCL; after GFN training, torch's CUDA caching
  allocator holds the whole A100 (free VRAM 40 GB → 1.1 GB), so docking gets 0
  poses. Node-independent (failed on balam004 too, which passed the OpenCL probe).
  Confirmed by a controlled login-A100 reproduction (fresh→docks; torch-holds→0
  poses; `empty_cache()`→docks).
- **Fix** (`glue/active_learning/loop.py`): `_free_torch_gpu_cache()` calls
  `gc.collect()` + `torch.cuda.empty_cache()` after training/sampling and before
  `oracle.score()` each round (prints free VRAM). Guarded; no-op without CUDA.
- **Pre-flight dock gate** (`experiments/active_learning/6td3/preflight_dock.py`,
  wired into `submit_al_6td3_gpu.sh`): docks 2 seed molecules at job startup and
  exits non-zero if 0 poses — catches a genuinely bad node in ~40 s instead of at
  round-1 (67 min). The OpenCL context probe alone was insufficient (balam009
  passed it but couldn't dock). `--exclude` now lists balam008,balam009.

Verified: the completed run freed ~39.7 GB before docking each round and added ~25
molecules/round; docking fell to ~1 min/round (<2% of the loop) vs entry-012's ~31
min (33%). See `Logs/014`. All changes are login-node smoke-tested; the full loop is
now confirmed on a real 3-round Balam run.

## 2026-07-03 — Fixed-reward docking matrix: per-step docking for every generator (Logs/021)

Extended the fixed-reward benchmark to run **per-step GPU docking as a fixed reward for
ALL FOUR generators** (previously RGFN-only), on the shared SMALL library. New structure:

- **`glue/proxies/oracle_reward_proxy.OracleRewardProxy`** — the missing generic adapter
  turning any `glue.oracles.GlueOracle` into an in-loop GFN reward (`clip(-raw/norm)` +
  per-step `torch.cuda.empty_cache()` so a GPU docking oracle isn't starved by torch during
  training — the Logs/014 fix moved into the reward path, the guard the raw upstream
  `DockingMoleculeProxy` lacks). Registered in `glue/proxies/__init__.py`. Symmetric third
  case next to `LearnedGlueProxy` (AL surrogate) and `ExampleGlueProxy` (template).
- **`glue/oracles/docking_seh_oracle.DockingClpPOracle`** — one-line ClpP target subclass;
  added to `scripts/score_batch.py` `ORACLES` as `docking_clpp`.
- **Baseline cross-env bridge** (the "attempt baselines too" work): `DockingBridgeReward` in
  `validation/generators/{fraggfn,rxnflow}/fixed_reward.py` + a `reward.type: docking` branch;
  `DockingBridgeProxy` (SCENT-clone `CachedProxyBase`) in
  `validation/generators/scent/docking_bridge_proxy.py`. Each docks a step's batch by shelling
  to `conda run -n rgfn score_batch.py` (benchmark 69691 showed ~10-13% amortized overhead),
  caches per canonical SMILES, `empty_cache` first. `ScentFixedRewardRun`/the FragGFN/RxnFlow
  runners now also write the raw dvina as a `raw_score` provenance column (backward-compatible).
- **Configs** (`configs/glue/fixed_reward_{6td3,clpp,drd2_stdlib}.gin`; `validation/configs/
  {fraggfn,rxnflow}_{6td3,clpp}_docking_fixed.yaml`, `scent_{6td3,clpp}_fixed.gin`,
  `rxnflow_drd2_fixed_stdlib.yaml`) — all on `glue_standard_v1`, β=4, 400 iters, and **lean
  train_metrics dropping `TanimotoSimilarityModes`** (it re-scores `%train_proxy` every
  eval step → would re-dock). Run-dir names follow `<gen>_<system>` so
  `submit_aizynth_fourway.sh SYSTEM=6td3|clpp` aggregates unchanged.
- **Orchestration**: `experiments/fixed_reward/{6td3,clpp,drd2_stdlib}/*.sh`, parameterized
  `baseline_docking/submit_{fraggfn,rxnflow,scent}_docking.sh CONFIG`, master
  `submit_matrix.sh`, benchmark + smoke dirs.

**Verified**: benchmark (69691) + 2 end-to-end Balam smoke jobs — RGFN in-env 6TD3 (69692) and
FragGFN cross-env (69693) both COMPLETED with conformant candidates and 100% per-step dock
success while torch trains. Imports/gin-parse/py_compile all pass. **NOT YET**: the full matrix
launch (awaiting user go-ahead) and a `git` commit. rgfn/ + upstream configs untouched.

## 2026-07-03 — `glue/analysis/`: trained-GFN hub-based late-stage-diversification pipeline

New **post-hoc analysis** subpackage for a *trained* reaction-GFN. Goal: select the `m`
highest-value **pre-terminal hubs** (shared `ReactionStateA` intermediates) and diversify
each in parallel via a single diverging final reaction (late-stage diversification), then
trade **diversity** against **concurrency** (# lanes) or **cost** (shared intermediate) — the
infrastructure to build that Pareto front, not the front itself.

- **`glue/analysis/`** — modular, registry-driven, NOT gin (driven by `scripts/analyze_gfn.py`
  + a Python/JSON `SweepSpec`), so intentionally NOT imported by `glue.registry`:
  - `loader.TrainedGFN` — load cfg+ckpt like `infer.py` (strict=False for sampling-cache
    buffers; imports `Trainer` so config parse resolves `Trainer.*`); sample trajectories,
    score states, expose env/policy/proxy.
  - `hub_graph` — `build_hub_graph(traj)` reduces trajectories to hubs; **flow = visit count**
    (TB trains no per-state flow), observed 1-reaction children off the penultimate `SA`.
  - `expanders` — `observed` (cheap, sampled) vs `enumerative` (env-driven exhaustive +
    proxy-scored, per-hub-key cached, capped-with-warning).
  - `hub_selectors` (`highest_flow`/`most_modes`/`highest_expected_reward`/`highest_child_reward`
    + `DiverseHubSelector`), `mol_selectors` (`top_k_reward`/`top_k_reward_diverse`/
    `scaffold_diverse_k`/`random_k`), `registry` (add a strategy = subclass + one line).
  - `batch_plan` (→ standard `CandidateDataset` + `lanes.csv`), `metrics` (diversity/
    concurrency/cost/reward; reuses `glue.metrics` + `glue.chemistry` cost), `cost` (route
    pricing from `ChemLibrary`), `select`, `sweep` (trials × strategies → `results.csv`),
    `pareto` (front extraction + trial aggregation).
- **`scripts/analyze_gfn.py`** — generic entry point (mirrors `scripts/infer.py`).
- **`experiments/diversification/`** — new experiment group; `seh_stdlib/` example (spec.json
  + submit).

**Verified** end-to-end on a real trained checkpoint (fixed-reward sEH proxy on
`glue_standard_v1`, GPU): observed sweep (2 trials × 24 configs → `results.csv`), all
selectors, both Pareto fronts (diversity-vs-concurrency and -vs-cost), enumerative expander
(single hubs → 29/87/105 products; `top_k_reward_diverse` lifts modes 48→75), cost saving
33% (observed) → 50% (enumerative k=25/lane). `py_compile` + imports pass. rgfn/ + upstream
untouched. **Placement is the recommended default** (`glue/analysis/` since `glue/metrics/`
is already post-hoc analysis `validation/` imports); the child-expansion default, cost-now,
and home are the open forks flagged to the user — a `git mv` moves the tree if they prefer
`validation/`. **NOT YET**: `git` commit; user sign-off on the forks.
