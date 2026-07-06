# sEH — same-library four-way fixed-reward benchmark (shared standard set)
**Date:** 2026-07-02, ~2pm

## Question

When we compare our synthesizable generators head-to-head, how much of the difference
comes from the *generator* versus the *library of chemical building blocks each one
happens to use — and what happens when we force them all to build from the exact same
library?

## Context & Summary

Our four-way sEH fixed-reward comparison (entry `019`) trained RGFN and three baseline
generators once each against the same frozen sEH scorer, and found RGFN's real edge is
*synthesizability* — the baselines match it on score. But that comparison had a hidden
variable: **each generator used a different building-block library.** RGFN drew on the
upstream `chemistry.xlsx` set (~350 fragments, 66 reactions), SCENT on its own priced
"SMALL" set (418 fragments, 112 reactions), RxnFlow on 10,000 ZINC blocks with 109
reaction templates. So "RGFN vs. RxnFlow" was really "RGFN-with-its-library vs.
RxnFlow-with-its-library" — the generator and its chemistry were confounded.

This experiment removes that confound for the three *reaction-based* generators. We built
one **standard library** — `glue_standard_v1`, the canonical form of SCENT's priced SMALL
set (chosen because it is the only library that ships the per-block prices and per-reaction
yields SCENT's cost-guidance needs) — and made **RGFN, SCENT, and RxnFlow all build from
it**. FragGFN can't participate: it has no reactions and its fragments need pre-defined
attachment points, so it stays in as the **non-synthesizable control**, running on its own
native fragment vocabulary (exactly its role in entry `019`). With the library held fixed,
any remaining difference between RGFN, SCENT, and RxnFlow is attributable to the generator's
action space alone.

## Answer

With the building-block library held fixed across the three synthesizable generators, the
differences from entry `019` **persist and sharpen** — so they were about the *generators*,
not their libraries. On identical chemistry the three separate into distinct corners rather
than converging: **SCENT** wins on sEH reward (median 7.63), route cost (cheapest, with a
coverage caveat), *and* external synthesizability (AiZynth 0.71); **RxnFlow** is by far the
most drug-like (QED 0.62 vs 0.22–0.29) and smallest (MW 413), at the price of the lowest
reward (6.22) and — surprisingly — the lowest AiZynth recovery (0.29) despite a 100%
by-construction route claim; **RGFN** scores well but drifts largest of the three and is the
priciest to synthesize. The **non-synthesizable foil (FragGFN)** matches on reward yet is
externally unsynthesizable (AiZynth 0.02, SA 4.32), confirming that reward parity ≠
synthesizability. Two methodological results fall out: (1) the library was *not* the
confound behind entry `019`; (2) "synthesizable by construction" (`has_route=1`) is not
uniform — an independent retrosynthesis tool recovers SCENT/RGFN routes far more often than
RxnFlow's, on the same library.

## Relevance to our Publication

A NeurIPS/Digital-Discovery reviewer comparing generators will ask the obvious control
question: *did you hold the building-block library fixed?* Entry `019` did not; this entry
does. It converts our four-way from "four systems as shipped" into a controlled comparison
where the generator's action space is the only variable across the synthesizable entrants —
the version of the benchmark that isolates what we actually claim to be measuring. It also
exercises the reusable standard-library infrastructure (`docs/CHEM_LIBRARY_FORMAT.md`) that
lets any future library (Enamine catalog, a curated glue set) be dropped in as one config
knob — itself a contribution toward the "contribution of the building-block set" ablation
(Objective 2).

## Next Experiments

**Refining for publication**
- Score all four candidate sets with AiZynthFinder + SA (entry `018`) into one comparison
  table (`validation/harness/aggregate_synthesizability.py`), directly alongside entry
  `019`'s different-library table so the library effect is visible.
- Add the **synthesis-cost axis**: price every entrant's routes retroactively with SCENT's
  cost model over `glue_standard_v1` (the `validation/harness/cost.py` evaluator, currently
  a stub — see `docs/CHEM_LIBRARY_FORMAT.md` §6), so RGFN/RxnFlow get the cost number SCENT
  optimizes natively. This is the entry-`017` follow-up.
- ≥3 seeds per generator for error bars (currently seed 42 only).

**Next steps in project**
- Run the same shared-library comparison on the validated **6TD3 glue oracle** (not just the
  sEH surrogate), so the controlled comparison holds on the real glue objective.
- Build a **curated glue standard set** (glue warheads + drug-like blocks) as a second
  library and re-run, testing whether library choice moves the size/drug-likeness drift.

# Re-creation

## Relevant Files

Root: `./` (repo root). Run outputs on `/scratch/markymoo/rgfn_runs/` (Balam; `$HOME` is
read-only on compute nodes).

**Standard-library infrastructure (ours, new):**
- `./glue/chemistry/library.py` — `ChemLibrary`: loads/exports a fragment+reaction library
  across formats (`from_scent_small`, `from_chemistry_xlsx`, `from_canonical_dir`;
  `to_canonical_dir`, `to_chemistry_xlsx`, `to_rxnflow_inputs`) + cost/yield maps.
- `./glue/chemistry/reaction_data_factory.py` — `GlueReactionDataFactory`: subclasses
  upstream `ReactionDataFactory` to source RGFN's action space from a canonical library dir
  instead of `data/chemistry.xlsx`, keeping `rgfn/` pristine. Reused via `glue/registry.py`.
- `./docs/CHEM_LIBRARY_FORMAT.md` — the format + design (the standard-set spec).
- `./validation/harness/cost.py` — retroactive synthesis-cost evaluator (stub; §6 of the doc).

**The standard library (ours, generated):**
- `./data/libraries/glue_standard_v1/{fragments.csv,reactions.csv,manifest.json}` — canonical
  form of SCENT's SMALL set: **418 fragments (all priced), 112 reactions (all with yields)**.
  Built from `external/scent/data/small/` via `ChemLibrary.from_scent_small`.
- `./data/models/rxnflow_env_stdlib/` — RxnFlow env dir baked from `glue_standard_v1`
  (`building_block.smi` = 412 retained blocks, `template.txt` = 112 templates, + `bb_*.npy`
  caches). `./data/models/rxnflow_env_stdlib_src/` — the raw blocks/templates it was baked from.

**Configs (ours):**
- `./configs/glue/fixed_reward_seh_proxy_stdlib.gin` — RGFN entrant; `include`s
  `fixed_reward_seh_proxy.gin` and swaps the data factory to `@GlueReactionDataFactory` on
  `glue_standard_v1`.
- `./validation/configs/rxnflow_seh_fixed_stdlib.yaml` — RxnFlow entrant; `env_dir` points at
  `data/models/rxnflow_env_stdlib`.

**Submit scripts (ours):**
- `./experiments/fixed_reward/stdlib_seh/submit_fixed_rgfn_stdlib_seh.sh` — job **69616**
  (3-day walltime, lean metrics; supersedes cancelled 69613 (24 h) and 69615 (full metrics)).
- `./experiments/fixed_reward/stdlib_seh/submit_fixed_rxnflow_stdlib_seh.sh` — job 69614.

**Reward generator (upstream, pristine):**
- `./rgfn/gfns/reaction_gfn/proxies/seh_proxy.py::SehMoleculeProxy` — the shared frozen
  Bengio-2021 sEH MPNN reward (same checkpoint every entrant trains against).

**Reused entrants (from entry `019`, same shared library / foil by construction):**
- SCENT: job **69608** (`fixed_reward/scent_seh/2026-07-02_10-23-09`) — trains on
  `external/scent/data/small/`, which `glue_standard_v1` is the byte-identical canonical copy
  of, so SCENT is already on the shared library (its Dynamic Library then *grows* fragments
  from that same seed — a property of SCENT's method, kept on).
- FragGFN: job **69606** (`fixed_reward/fraggfn_seh/2026-07-02_10-23-21`, COMPLETED, 1000
  candidates) — native 72-fragment vocabulary; the non-synthesizable control.

**Results:**
- New: `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/seh_proxy_stdlib/<ts>/` (RGFN,
  69615) and `.../rxnflow_seh_stdlib/<ts>/` (RxnFlow, 69614).
- `./validation/results/seh_fixed_reward_stdlib/comparison.{csv,md}` — four-way table (pending).

**Job Logs:** `/scratch/markymoo/rgfn_runs/fr_*_stdlib_seh-<jobid>.{out,err}`.

## Relevant Versions

Branch `GPU-Dock`. New files (`glue/chemistry/`, `docs/CHEM_LIBRARY_FORMAT.md`,
`data/libraries/glue_standard_v1/`, `configs/glue/fixed_reward_seh_proxy_stdlib.gin`,
`validation/configs/rxnflow_seh_fixed_stdlib.yaml`, `validation/harness/cost.py`,
`experiments/fixed_reward/stdlib_seh/`) and the one-line `glue/registry.py` edit are **not
yet committed**. [TODO — add commit hash after pushing.] (The baked `data/models/
rxnflow_env_stdlib*` caches are large/regenerable — commit the `.smi`/`.txt` sources, not
the `.npy`.)

## Relevant Resources

**Sources**
- RGFN (`[koziarski2024rgfn]`, arXiv:2406.08506) §4.1 — the fixed-reward sEH setup.
- SCENT (`[gainski2025scent]`, arXiv:2506.19865) — the cost-aware generator + the priced
  SMALL library (`external/scent/data/small/`) `glue_standard_v1` is derived from.
- RxnFlow (`[seo2024rxnflow]`, ICLR 2025) — the reaction-template + building-block generator;
  env baked via `external/RxnFlow/data/scripts/b_create_env.py`.

**Packages**
- `rgfn` (env `rgfn`, py3.11) — RGFN entrant + `GlueReactionDataFactory` + candidate ingest.
- `rxnflow` (env `rxnflow`, py3.12) — RxnFlow entrant; `scent` (env `scent`) — SCENT entrant;
  `gflownet` (env `fraggfn`, py3.10) — FragGFN foil.
- Balam login helper `~/bin/rgfn-smoke-env.sh` for the local smoke tests.

## Method

1. Built the standard library: `ChemLibrary.from_scent_small('external/scent/data/small')`
   → `to_canonical_dir('data/libraries/glue_standard_v1')`. Verified round-trip (418 frags /
   112 reactions, all priced/yielded) and cost/yield lookups.
2. Verified RGFN's action space from it: `GlueReactionDataFactory(library_dir=
   'data/libraries/glue_standard_v1')` → 418 fragments, 112 reactions, 206 anchored
   reactions; `ReactionEnv` builds.
3. Baked the RxnFlow env: `ChemLibrary.to_rxnflow_inputs` → `building_block.smi` + `template.txt`,
   then (in the `rxnflow` env) `external/RxnFlow/data/scripts/b_create_env.py -b … -t … -o
   data/models/rxnflow_env_stdlib` → 412/418 blocks retained, mask shape (188, 412).
4. Smoke-tested both (login-node A100): RGFN 3 iters → 15 routed candidates from the shared
   library; RxnFlow 5 steps → finite loss (1990→2878, no NaN) → 10 routed candidates. Both
   emit conformant candidate datasets.
5. Queued on Balam (2026-07-02): `sbatch` RGFN-stdlib and RxnFlow-stdlib → **69614** (1-GPU).
   SCENT (69608) + FragGFN (69606) reused. **Walltime correction:** the first RGFN submit
   (69613, 24 h) ran at ~54 s/iter — the shared library's 112 reactions (vs chemistry.xlsx's
   66) roughly double RGFN's CPU-bound per-iteration cost, so 5000 iters need ~70 h and 69613
   would have timed out at ~step 1600. Cancelled 69613 and resubmitted as **69615** with a
   3-day walltime and `TanimotoSimilarityModes.max_modes` capped at 1500 (a wandb-only
   diagnostic, capped to flatten the per-iter rate; does not affect training/candidates). The
   5000-step budget (matched to the other entrants) is preserved. RxnFlow (69614) runs at
   ~3 s/step (its shrunk 412-block env is fast) and fits 24 h easily.
6. **Metric audit + lean resubmit (69616).** Investigating RGFN's slow per-iter rate,
   measured the training-metric overhead directly (login GPU, 12 iters, full vs
   `[@StandardGFNMetrics()]` only, 12 iters): full median 48.3 s/it vs minimal 43.1 s/it →
   metrics are only ~11% of per-iter, NOT the dominant cost. Both full *and* minimal rate *rise* over iterations →
   the rise is in the trajectory rollout (consistent with molecule size-drift: bigger
   molecules → costlier rollout/scoring each step), not the metrics; SCENT stays fast partly
   because its cost guidance counteracts size-drift (entry `017`: MW 694→579) and it samples
   fewer forward trajectories (64 vs our 100). Applied the audit's conclusion: RGFN-stdlib
   now runs a **lean** `train_metrics = [StandardGFNMetrics, FractionEarlyTerminate,
   UniqueMolecules]` — keep only the must-run-live scalars + the cheap `{smiles: score}` log;
   DEFER QED / NumScaffoldsFound / modes / diversity to the post-hoc `dataset_metrics.py`
   over the saved candidates (identical recipes). Resubmitted as **69616** (the earlier
   `max_modes=1500` cap in 69615 was ineffective — `TanimotoSimilarityModes` is gated to
   every `evaluation_step=500`, so it wasn't the per-step cost).
7. **Pending:** on completion, score all four with AiZynth (entry `018`) + aggregate into
   `validation/results/seh_fixed_reward_stdlib/comparison.{csv,md}`.

## Results

**Smoke tests (login-node A100, 2026-07-02) — machinery validated:**

| entrant | steps | outcome |
|---|---|---|
| RGFN-stdlib | 3 | 15 unique valid candidates, `has_route=1`, built from `glue_standard_v1` (e.g. seed block `O=Cc1ccccc1Br`); best sEH score 4.74 |
| RxnFlow-stdlib | 5 | loss finite 1990→2878 (**no NaN** on the 412-block library), 10 unique valid candidates, `has_route=1` |

**Standard library `glue_standard_v1`:** 418 fragments (418 priced), 112 reactions (112 with
yields); RxnFlow env retained 412 blocks. Source: `external/scent/data/small/`.

**Four-way comparison (RGFN / SCENT / RxnFlow on `glue_standard_v1`; FragGFN native foil).**
All four runs COMPLETE (1000 candidates each): RGFN 69616 (2d01h), RxnFlow 69614 (3h13m),
SCENT 69608 (9h25m), FragGFN 69606 (2h05m).

*Descriptors + reward (1000 candidates each):*

| generator | has_route | sEH score (med / max) | MW (med) | QED (med) |
|---|---|---|---|---|
| SCENT | 1000/1000 | **7.63** / 8.39 | 515 | 0.26 |
| RGFN | 1000/1000 | 7.26 / 8.35 | 551 | 0.22 |
| FragGFN (foil) | 0/1000 | 7.23 / 8.42 | 688 | 0.16 |
| RxnFlow | 1000/1000 | 6.22 / 7.34 | **415** | **0.66** |

*Synthesis cost (retroactive `PathCostProxy` pricing on `glue_standard_v1`, top-100 by score;
`validation/harness/cost.py`; lower = cheaper):*

| generator | priced | top-100 cost (med / mean) | route len | any-fallback |
|---|---|---|---|---|
| SCENT | 1000/1000 | 2.88 / 7.33 | 2.71 | 0.744 |
| RxnFlow | 1000/1000 | 7.78 / 14.4 | 2.98 | 0.000 |
| RGFN | 1000/1000 | 34.15 / 36.9 | 3.94 | 0.000 |
| FragGFN | 0/1000 | — (no routes) | — | — |

Cost **coverage caveat:** RGFN and RxnFlow price at 0% fallback (every route component is a
base `glue_standard_v1` block/reaction → exact). SCENT hits ~74% fallback because its
**dynamic library** promotes synthesized intermediates to building blocks absent from the
base library; those are imputed at the default cost (1.0), so SCENT's cost is an approximate
lower bound, not directly comparable to the fully-priced RGFN/RxnFlow. Full table:
`validation/results/seh_fixed_reward_stdlib/cost_comparison.{csv,md}`.

*Synthesizability (AiZynthFinder + SA, top-100; job 69867, COMPLETE 14m25s;
`validation/results/seh_fixed_reward_stdlib/comparison.{csv,md}`):*

| generator | self-route (by-construction) | AiZynth success | steps | SA |
|---|---|---|---|---|
| SCENT | 1.00 | **0.71** | 4.07 | 3.13 |
| RGFN | 1.00 | 0.66 | 4.08 | 3.36 |
| RxnFlow | 1.00 | **0.29** | 4.07 | 3.46 |
| FragGFN (foil) | 0.00 | 0.02 | 3.50 | 4.32 |

The **by-construction claim (`self-route`) vs the independent AiZynth verdict diverge**, and
that divergence is the interesting result: SCENT/RGFN's routes are largely recovered by an
external retrosynthesis tool (0.71 / 0.66), FragGFN's are not (0.02 — it never claimed any),
but **RxnFlow's are recovered only 29% of the time despite a 100% by-construction claim** —
its small, drug-like molecules use block/connectivity choices AiZynth's USPTO+ZINC stock
often can't reproduce. So "synthesizable by construction" is strongest for SCENT/RGFN and
weakest (externally) for RxnFlow, on identical chemistry.

**Headline (same-library, generator is the only variable across the synthesizable three):**
on identical chemistry, the three synthesizable generators separate cleanly — SCENT posts
the best sEH reward at moderate size and the cheapest routes (with the dynamic-library
caveat); RxnFlow is the most drug-like (QED 0.66) and smallest at a modest reward cost and
0%-fallback cheap routes; RGFN scores well but drifts largest of the three and priciest to
synthesize. FragGFN (non-synth foil) matches on reward but carries no route (has_route=0).
