# SCENT / 6TD3 — cost-aware baseline head-to-head with RGFN
**Date:** 2026-06-30, ~5pm

## Question

When a competing generator actively tries to keep its molecules **cheap to
synthesize**, does our RGFN glue generator still produce comparably good 6TD3 glue
candidates — and what does that cost-awareness buy in diversity and synthesis cost —
given the same scoring oracle and the same budget?

## Context & Summary

**Context.** Entries `015` (FragGFN, the non-synthesizable foil) and `016` (RxnFlow,
a synthesizable peer) built out the baseline comparison reviewers will demand. SCENT
(`[gainski2025scent]`, arXiv:2506.19865) is the most pointed comparison of the three:
it is a **fork of RGFN from the same lab** that keeps RGFN's synthesizable
reaction-template machinery and adds explicit **cost-awareness** — it learns to steer
generation toward molecules whose synthesis routes use **cheap building blocks** and
**high-yield reactions**, plus two supporting tricks (a penalty that stops it from
over-exploiting the cheapest routes, and a "dynamic library" that promotes useful
intermediates to building blocks). RGFN is literally one of SCENT's own published
baselines, so this is the closest thing to a controlled test of "RGFN vs. RGFN + cost
guidance" on our glue task.

**Summary.** We run SCENT through the **same** active-learning loop, the **same** 6TD3
docking oracle, the **same** seed set of 408 labelled molecules, the **same** budget
(3 rounds × 32 molecules/round, 300 generator steps/round), the **same**
inverse-temperature β=8, and the **same** learned proxy reward as the RGFN run — so the
*only* thing that changes is the generator's internal machinery. SCENT runs its **full
cost-aware method** on its **SMALL building-block library**, which ships the
building-block prices and reaction yields its cost model needs (so cost guidance is
genuinely active, not a no-op). Because SCENT's python package is *named* `rgfn` (it is
an RGFN fork), it runs in its own software environment and reaches our docking oracle
through the shared scoring bridge — the same cross-environment setup FragGFN/RxnFlow
use, here recording each molecule's synthesis route so SCENT's output sits directly
alongside RGFN's.

## Answer

SCENT's cost-aware generator produces 6TD3 glue candidates on par with our RGFN run and
the FragGFN baseline, while keeping every molecule synthesizable — so cost-awareness does
**not** cost glue quality here. Over 3 rounds it generated 96 molecules (all with a
synthesis route, `has_route=96/96`) with a median docking differential of **−2.12** and a
best of **−5.81** (the strongest single candidate of any entrant so far), 51% in the
glue-range (≤ −2.0). The most striking difference is a **beneficial side effect of cost
guidance**: across rounds the mean molecular weight *fell* (694 → 635 → 579) and QED
*rose* (0.11 → 0.17 → 0.21) — the reverse of the size-bloat/QED-collapse drift seen in the
RGFN (entries `011`/`014`) and FragGFN (entry `015`) runs. Steering toward cheap, high-yield
routes appears to pull the generator toward smaller, more drug-like molecules "for free" —
the same drug-like region RxnFlow's block library confines it to (entry `016`, MW 489 /
QED 0.36), but reached here by cost pressure rather than a constrained action space. So on
the four-way matched-oracle comparison (below), SCENT and RxnFlow are the two entrants that
escape the RGFN/FragGFN size drift. (One seed — a trend to confirm with ≥3 seeds, and by
pricing the RGFN/FragGFN routes with SCENT's own cost model for a like-for-like cost
comparison.)

## Relevance to our Publication

This is the **cost-aware** arm of the baseline comparison (`docs/RESEARCH_CONTEXT.md`,
Objectives 4–5). FragGFN (`015`) controls for synthesizability, RxnFlow (`016`) controls
for a different synthesis-aware action space, and SCENT controls for **explicit cost
optimization** — letting us answer the reviewer question "is RGFN leaving easy
synthesis-cost wins on the table?" Because SCENT is RGFN's own published successor,
showing RGFN remains competitive (or showing exactly where cost-awareness changes the
candidate set) on the same oracle and budget is a strong, directly-citable result for a
methods venue (Digital Discovery / JCIM). SCENT also reports the AiZynth success rate we
adopt as our synthesizability metric (its Table 1, up to ≈0.75), tying the comparison to
`validation/harness/synthesizability.py`.

## Next Experiments

**Refining for publication.**
- Run ≥3 seeds for RGFN and SCENT (and FragGFN/RxnFlow) so the head-to-head carries error
  bars.
- **Matched-oracle RGFN GPU anchor — DONE:** entry `014`'s GPU run has since completed on a
  healthy node (job 69517), so all four entrants now share the identical GPU oracle and the
  head-to-head below is a clean four-way (no CPU placeholder).
- Report **average synthesis cost** of the top-k glues (SCENT's headline axis) alongside
  glue score + diversity — the dimension SCENT is built to win.
- In-codebase ablation: rerun SCENT with cost guidance / exploitation penalty / dynamic
  library toggled off (the `include`s in SCENT's `configs/scent_base.gin`) to attribute
  any difference to the specific mechanism.

**Next steps in project.**
- Fold SCENT into the validation harness's top-k-vs-oracle-calls curve (the
  `[bengio2021gflownet]` Fig. 7 analogue) alongside RGFN, RxnFlow, FragGFN, and a
  random-acquisition baseline.
- Extend the comparison to a second protein system once a second oracle is validated
  (Objective 0 / 4).

# Re-creation

> **STATUS: COMPLETE.** Ran end-to-end on Balam (job 69513, balam009, 2 h 03 m,
> `COMPLETED`). Three fixes were needed during bring-up (setuptools pin, explicit
> `Trainer` import, sign-safe `train_metrics`) — see "Balam bring-up" below.

## Relevant Files

Root: repository root unless noted.

**Scripts / adapter** (`./validation/generators/scent/`):
- `run_scent_al.py` — entry point (runs in the `scent` env): absolutises our paths,
  `chdir`s into the SCENT clone, registers the gin classes, parses the gin config, runs
  the loop. Deliberately keeps the repo root OFF `sys.path` so SCENT's installed `rgfn`
  isn't shadowed by our repo-local `rgfn/` (uses sibling imports instead).
- `al_loop.py` — `ScentActiveLearningLoop`: the `[bengio2021gflownet]` Alg. 1 loop (fit
  proxy → train SCENT → sample batch + routes → label via bridge → accumulate → Top-K),
  with the all-NaN oracle-failure abort guard; `LabelStore` is the accumulating `D`.
- `proxy.py` — `LearnedDockingProxy`: the in-loop reward `M`. The same Bengio-2021
  atom-graph MPNN as RGFN's `glue.proxies.LearnedGlueProxy`, here subclassing **SCENT's**
  `CachedProxyBase` and importing **SCENT's** bundled `MPNNet`/`mol2graph` (identical
  symbols from its own `rgfn` fork). Fresh init, refit each round.
- `route.py` — self-contained copy of `glue.active_learning.route` (`extract_route`) for
  the `scent` env; reconstructs each molecule's synthesis route from its trajectory.

**Shared oracle bridge:**
- `./scripts/score_batch.py` — scores a batch of SMILES with a named glue oracle and
  writes the standard candidate-dataset format; route-aware (`--routes` per-round JSONL →
  joined into `routes.jsonl` on `--finalize`, `has_route=1`). Runs in the `rgfn` env.
  (Shared with the RxnFlow entrant; not modified here.)

**Install / run scaffolding:**
- `./external/setup_scent.sh` — clones `koziarskilab/SCENT` (pinned `af1fee5`) under
  `external/scent`, creates the `scent` conda env (py3.11.8, torch 2.3.0+cu118, dgl
  2.2.1+cu118), `pip install -e .`, and import-smoke-tests that SCENT's `rgfn` resolves.
- `./experiments/active_learning/scent_6td3/submit_scent_6td3.sh` — Balam SLURM submit
  (OpenCL health gate, `--exclude=balam008`, `$SCRATCH` outputs); activates `scent`, the
  bridge subprocess re-enters `rgfn`.

**Config / seed:**
- `./validation/configs/scent_6td3.gin` — gin AL config: `include`s SCENT's
  `scent_base.gin` (full cost-aware method) + `small.gin` (SMALL library + shipped
  cost/yield data), swaps the sEH proxy for `@LearnedDockingProxy`, and matches the RGFN
  run's budget/seed/β. `./validation/configs/scent_smoke.gin` — tiny mock-oracle dry run.
- `./external/scent/data/small/{fragments.txt,templates.txt,fragment_to_real_cost.json,templates_yields.csv}`
  — SCENT's SMALL building blocks (418), reaction templates (112), building-block
  **prices**, and reaction **yields** — the inputs that make Recursive Cost Guidance
  active. Shipped in the SCENT repo (git-ignored clone).
- `./experiments/active_learning/6td3/seed_6td3.csv` — the shared seed `D_0` (408
  validated docking labels), reused identically by RGFN, FragGFN, RxnFlow, and SCENT.

**Results** (git-ignored, on `$SCRATCH`):
- `/scratch/markymoo/rgfn_runs/experiments/active_learning/scent_6td3/<timestamp>/` —
  per-round `dataset_round_NNN.csv`, `top_k.csv`, and `suggestions/` (standard
  `candidates.csv` + `manifest.json` + `routes.jsonl`, `batch_metrics.csv`). Concrete run
  (job 69513): `…/scent_6td3/2026-06-30_19-46-47/`. SLURM logs:
  `/scratch/markymoo/rgfn_runs/al_scent_6td3-69513.{out,err}`.

## Relevant Versions

```
08da97c Working GPU Oracle RGFN, FGFN, RxnFlow, SCENT, AiZynthFinder
ded1c0d checkpoint for FGFN, RxnFlow, SCENT, AiZynthFinder, sEH
cdf3f78 GPU loop + FGFN loop
```

The SCENT adapter (`validation/generators/scent/*`), `external/setup_scent.sh`, and the
first `validation/configs/scent_{6td3,smoke}.gin` landed in **`ded1c0d`** ("checkpoint …")
and were finalised in **`08da97c`** ("Working GPU Oracle …") along with the three bring-up
fixes (setuptools pin, explicit `Trainer` import, sign-safe `train_metrics`),
`experiments/active_learning/scent_6td3/*`, and the reference/doc updates
(`Logs/references/{README.md,references.bib}`, `validation/generators/README.md`,
`docs/REFACTOR_LOG.md`). `08da97c` is also the commit that carries the RGFN GPU anchor
(job 69517) this run is compared against.

## Relevant Resources

**Sources.**
- `[gainski2025scent]` — SCENT: *Scalable and Cost-Efficient de Novo Template-Based
  Molecular Generation*, arXiv:2506.19865. Code: https://github.com/koziarskilab/SCENT.
- `[bengio2021gflownet]` — the active-learning loop (Alg. 1) every entrant follows.
- `[koziarski2024rgfn]` — RGFN, the generator SCENT forks and is compared against.
- `[slabicki2020cr8]` — the 6TD3 system (DDB1·CDK12–cyclinK·CR8) the oracle scores.

**Packages.**
- SCENT (its own `rgfn` fork) — the cost-aware generator; used by
  `validation/generators/scent/{proxy,route,al_loop}.py`.
- gnina + QuickVina2-GPU — the docking oracle backend, reached via
  `scripts/score_batch.py` (`Docking6TD3GpuOracle`).
- RDKit — SMILES canonicalization + descriptors, throughout.

## Method

> Commands below are the ones actually run for job 69513.

1. `bash external/setup_scent.sh` — clone SCENT, build the `scent` env, install it.
2. CPU smoke: `conda run -n scent python validation/generators/scent/run_scent_al.py
   --cfg validation/configs/scent_smoke.gin --seed-csv
   experiments/active_learning/6td3/seed_6td3.csv --root-dir /tmp/scent_smoke`
   (mock oracle; validates the full loop + route logging; needs `module load cuda/11.8.0`
   for the bridge's `rgfn` env).
3. Real run: `sbatch experiments/active_learning/scent_6td3/submit_scent_6td3.sh` —
   3 rounds, GPU docking oracle, on a healthy node.

## Results

**Job 69513** (balam009, `COMPLETED`, 2 h 03 m). 3 rounds × 32 query/batch, 300 SCENT
steps/round, seed `D_0`=408, β=8. Docking succeeded on 90/96 sampled molecules (2 dock
failures/round — normal). Wall-clock per round: proxy fit 10–17 s, SCENT GFN train
36–41 min, sample 8–9 s, GPU docking 69–80 s. (SCENT's cost-guidance makes its GFN step
~7× slower than plain RGFN's ~5 min/round — the dominant cost, as for every entrant.)

**Per-round trend (standard `batch_metrics.csv`):**

| Round | \|D\| | oracle mean (dvina) | best | internal diversity | mean MW | mean QED |
|---|---|---|---|---|---|---|
| 1 | 438 (+30) | −2.19 | −5.81 | 0.86 | 694 | 0.11 |
| 2 | 468 (+30) | −1.94 | −3.65 | 0.88 | 635 | 0.17 |
| 3 | 498 (+30) | −2.01 | −4.86 | 0.87 | 579 | 0.21 |

MW falls / QED rises across rounds — opposite to the RGFN/FragGFN drift.

**Full 96-candidate set (`candidates.csv`):** median dvina **−2.117**, mean **−2.049**,
best **−5.807**; **51%** ≤ −2.0, 21% ≤ −3.0; `has_route` = **96/96**; `num_reactions`
mostly 4 (90), with 3 (2) and 1 (4) — tree-structured routes from the dynamic library.

**Head-to-head — clean matched-oracle GPU four-way.** All four entrants share the
**identical** `Docking6TD3GpuOracle`, seed `D_0` (408 mol), budget (3 rounds × 32), β=8,
and proxy `M`; only the generator differs. Numbers are **generated-only** (the 96 suggested
molecules; `top_k.csv` pools the shared seed and is not a generator metric). RGFN = job
**69517** (entry `014`), FragGFN = job 69482 (entry `015`), RxnFlow = job 69518 (entry `016`),
SCENT = job **69513** (this run) — all on the GPU oracle on healthy nodes.

| Metric | RGFN (69517) | FragGFN (69482) | RxnFlow (69518) | **SCENT** (69513) |
|---|---|---|---|---|
| best `dvina` | −3.90 | −4.86 | −4.22 | **−5.81** |
| median `dvina` | **−2.14** | −2.06 | −1.19 | −2.12 |
| mean `dvina` | −2.14 | −1.84 | −1.41 | −2.05 |
| frac ≤ −2.0 | 42/96 (44%) | **52/96 (54%)** | 30/96 (31%) | 49/96 (51%) |
| frac ≤ −3.0 | 8/96 (8%) | **20/96 (21%)** | 5/96 (5%) | 20/96 (21%) |
| int. diversity | 0.85 | 0.87 | **0.89** | 0.87 |
| synthesizable route | yes (by constr.) | **none** (`has_route=0`) | yes 96/96 (2-step) | yes 96/96 (tree) |
| mean MW | 665 | ≈720 | **489** | ≈636 (694→579↓) |
| mean QED | 0.12 | ≈0.15 | **0.36** | ≈0.16 (0.11→0.21↑) |
| log | `014` | `015` | `016` | `017` |

**Reading it.** On glue score the four are in the same league (medians −1.2…−2.1), with
SCENT taking the single strongest hit (−5.81) and matching RGFN's central tendency; RGFN and
FragGFN reproduce the size-bloat/low-QED drift (MW 665–720, QED 0.12–0.15), while **SCENT and
RxnFlow are the two entrants that stay drug-like** — RxnFlow by its constrained block library
(MW 489, QED 0.36), SCENT by cost pressure pulling MW down and QED up *across rounds* toward
that same region. This matches entry `016`'s finding that RGFN's synthesizability is real but
does not by itself buy drug-likeness; SCENT shows a second route to it (cost guidance) beyond
RxnFlow's block constraint. (Caveat carried from `016`: single-seed, small 3×300-step budget —
deltas need ≥3 seeds before they are load-bearing.)

SCENT's own per-round synthesis-cost diagnostic (`@TrajectoryCost.forward_mean_cost`) is
logged to the offline wandb history under the run dir (not the summary). A like-for-like
"average synthesis cost of the good glues" comparison needs SCENT's cost model applied to
the RGFN/FragGFN routes too — deferred (Next Experiments).

### Balam bring-up (2026-06-30) — three fixes found running it for real
Building/validating in the `scent` env surfaced three issues, all fixed:
1. **`wandb` needs `pkg_resources`** — the py3.11 env shipped `setuptools>=81` (which
   removed `pkg_resources`), so `import rgfn` (→ its wandb logger) died with
   `ModuleNotFoundError: pkg_resources`. Fix: pin `setuptools<81` (added to
   `external/setup_scent.sh`).
2. **`@Trainer` not registered at gin-parse** — `import rgfn` runs `rgfn/trainer/__init__`
   which imports `artifacts/logger/metrics/optimizers` but NOT `trainer.py`, so gin saw
   "No configurable matching 'Trainer'". Fix: the runner now does an explicit
   `from rgfn.trainer.trainer import Trainer` (exactly what SCENT's own `train.py` does).
3. **SCENT's sEH-tuned metrics crash on our proxy's sign** — `ScaffoldCost` (via
   `ScaffoldCostsList`) records any state with `proxy > threshold(=8)` and calls
   `.molecule` on it *without* a terminal-type guard. Our docking proxy is
   `higher_is_better=False` and assigns invalid/early-terminal states the worst value
   `+clip=+10`, which trips `10>8` → `AttributeError` on a `ReactionStateEarlyTerminal`.
   These are wandb-only diagnostics (not used by the AL comparison, which reads our
   `batch_metrics` via the bridge), so the configs override `train_metrics` to a
   sign-safe subset — keeping SCENT's own synthesis-cost diagnostic `@TrajectoryCost`.

**Smoke (mock oracle, `scent_smoke.gin`, login A100) — PASSED end-to-end:** gin parse →
cost-guided Trainer build → proxy fit (val MSE 0.78 on 408) → GFN train → 4 candidates
sampled *with routes* → bridge scored 4/4 under the `rgfn` env → standard dataset
finalized with `has_route=1`, `num_reactions`∈{1,3,4} (tree-structured), `routes.jsonl` +
medchem descriptors. **Real GPU run: job 69513** on balam009 (3 rounds, `docking_6td3_gpu`).

### Local static validation (pre-Balam)
- `py_compile` of all new SCENT Python — passed.
- `bash -n` of `setup_scent.sh` + `submit_scent_6td3.sh` — passed.
- Static gin-integrity check: every `include` in `scent_{6td3,smoke}.gin` resolves into
  the SCENT clone; every symbol `proxy.py`/`route.py` import (`MPNNet`,
  `NUM_ATOMIC_NUMBERS`, `mol2graph`, `mols2batch`, `_chunks`, `CachedProxyBase`,
  `ReactionState{,EarlyTerminal,Terminal}`, `ReactionAction0`, `ReactionActionC`) is
  present in SCENT's source; `SCENT/data/small/{fragments,templates,
  fragment_to_real_cost,templates_yields}` all present (418 / 112 / 420 / 123 lines).
- `CachedProxyBase.compute_proxy_output` wraps a `List[float]` from
  `_compute_proxy_output` into `ProxyOutput(value=…)` — matches `LearnedDockingProxy`'s
  return type (same contract as SCENT's own `SehMoleculeProxy` and our `LearnedGlueProxy`).
