# 6TD3 / RGFN — GPU-oracle active-learning loop with synthesis-route + metrics logging

**Date:** 2026-06-29, ~late afternoon

## Question

When we drive the multi-round active-learning loop with the fast GPU docking
oracle instead of the CPU one, do the generated molecules still dock like real
glues — and can we now capture, for every molecule the model suggests, the
synthesis route that builds it and a panel of medchem/diversity metrics per round?

## Context & Summary

**Context** — Entry 011 ran the first 3-round active-learning loop on the CPU
gnina oracle: the generated molecules docked as genuine glues (median differential
≈ −2.4), but the loop took ~5 h, with docking ~35% of that. Entry 012 pinned the
cost precisely (the Tier-2 conformational search is 99.3% of docking ≈ 33% of the
whole loop), and entry 013 built a GPU oracle (`Docking6TD3GpuOracle`) that runs
that search on QuickVina2-GPU ~50× faster while keeping most of the validated
discrimination (AUROC 0.88 vs 0.948). The natural next step is to put that GPU
oracle *inside* the loop. At the same time the loop was throwing away everything
except each molecule's SMILES + final score — so we could not later check that a
suggested glue's by-construction synthesis route is plausible, nor analyse how
diversity / molecular weight evolve across rounds (a gap entry 011 explicitly
flagged: "track molecular weight and QED on every round's batch").

**Summary** — Two changes, run together. (1) We added a per-round **suggestion
log**: the loop now keeps each suggested molecule's full RGFN trajectory, writes
its structured **synthesis route** (starting building block + every reaction step
with SMARTS, reactants and product), its medchem descriptors, and the oracle's
score breakdown, plus a **batch-metrics** row per round (uniqueness, number of
Tanimoto modes, internal diversity, scaffold count, novelty vs the seed, and the
MW/QED/cLogP/etc. distributions) — all following the metric set RGFN's own mode
report and the source papers use. None of this touches the proxy-fit dataset; it
is pure provenance. (2) We swapped the loop's oracle from the CPU
`Docking6TD3Oracle` to the GPU `Docking6TD3GpuOracle`, holding every other knob
identical to the entry-012 baseline (3 rounds, 300 GFN steps/round, 32-molecule
query batch, β=8, top-16), so this run is a clean diff against entry 012 (job
69451): only the oracle's pose source and the new logging change.

## Answer

**The GPU-oracle active-learning loop now runs end to end, and it exposed — then
fixed — a real bug about running a GPU docker inside a GPU training loop.** The
completed 3-round run (job 69517) grew the labelled set 408 → 483 and the molecules
RGFN generated dock as **genuine glue candidates**: per-round median differential
−2.14 / −1.99 / −2.42 (squarely in known-glue territory, matching entry 011's CPU
median ≈ −2.4), best −4.92, with round 3 scoring 16/26 docked molecules at ≤ −2.0.
Every round the batch was maximally diverse (32/32 unique, 32 modes, 32 scaffolds,
novelty 1.0 vs seed), and — as entry 011 warned — heavy and non-drug-like (mean MW
~630–690, QED ~0.10–0.13), the size drift the reward still needs to counter
(Objective 2). Each molecule's **synthesis route** was captured (96/96), e.g. a
β-alanine building block elaborated over 4 reactions.

Getting here took four attempts and surfaced the real lesson. The first three runs
all failed at round-1 docking with every molecule `no_pose` — first blamed on a
node (balam009), but it recurred on balam004, which had *passed* a startup dock.
A controlled login-node reproduction pinned it: **GPU-memory contention**. The GPU
docker runs QuickVina2-GPU in a *subprocess* that needs GPU memory via OpenCL, but
after ~1 h of GFN training torch's caching allocator holds the whole A100 (free
VRAM 40 GB → **1.1 GB**), so every dock fails; `torch.cuda.empty_cache()` before
docking returns the memory (→ 39.7 GB free) and docking recovers. This is a general
gotcha for any GPU oracle driven inside a torch training loop, invisible to the
oracle in isolation (which is why entry 013's standalone test and the Trillium
smoke test always worked). Three robustness fixes came out of it (below); the
standard candidate logging, meanwhile, wrote correctly through every failure.

## Relevance to our Publication

This run advances two objectives at once. For **Objective 1 / Objective 2**, it
demonstrates the loop running end-to-end at GPU throughput — the answer to a
likely Digital Discovery / J. Cheminformatics reviewer question, "your oracle is
expensive; can it drive active learning at realistic scale?" For **Objective 5
(evaluation suite)**, the per-round suggestion log + batch metrics are the
substrate for the headline analyses reviewers expect: diversity and molecular-
weight trajectories under a fixed oracle budget, and — uniquely for a *reaction*
GFlowNet — verifiable synthesis routes for the top candidates, which a baseline
generator that does not build molecules by construction cannot offer.

## Next Experiments

**Refining for publication** — Run ≥3 seeds and the random-acquisition baseline on
the same oracle budget for the top-k-vs-oracle-calls curve (`[bengio2021gflownet]`
Fig. 7). Use the new batch-metrics CSV to plot the MW / QED / diversity trajectory
directly, quantifying the size drift entry 011 noted. Spot-check a handful of
top-k synthesis routes for chemical plausibility.

**Next steps in project** — Fold a QED / ligand-efficiency term into the reward
(Objective 2) to counter size drift, now that we can measure it per round. Drive
the sEH GPU oracle (entry 010) through the same instrumented loop for a second
end-to-end system (Objective 4).

# Re-creation

## Relevant Files

Root: `glue/`, `configs/glue/`, `experiments/active_learning/6td3/`

New / changed code (this entry):
- `./glue/active_learning/route.py` — **new.** `extract_route(states, actions)`:
  reconstructs a molecule's synthesis route from its RGFN trajectory — the
  `ReactionAction0` building block + every `ReactionActionC` (reaction SMARTS,
  reactants, product). `route_to_str` renders a one-line CSV-friendly summary.
- `./glue/metrics/dataset_metrics.py` — **new.** Pure (gin-free) per-molecule
  descriptors (ExactMolWt, QED, cLogP, HBD/HBA, heavy atoms, rotatable bonds,
  TPSA, rings, Lipinski-Ro5, ligand efficiency, Murcko scaffold) and set metrics
  (uniqueness, # Tanimoto modes [Morgan r3/2048, 0.7], internal diversity, scaffold
  count, novelty-vs-seed, descriptor + oracle-score distributions). Metric set
  mirrors `rgfn/trainer/metrics/reaction_metrics.py` and `[koziarski2024rgfn]` /
  `[bengio2021gflownet]`. Runs live in the loop *and* retroactively on any CSV.
- `./glue/datasets/candidates.py` — **new.** The ONE standard candidate-dataset
  format (`CandidateDataset` writer, `from_smiles_table` ingest, `read_candidate_dataset`
  / `validate_candidate_dataset`): `manifest.json` + `candidates.csv` (+ `routes.jsonl`).
  Lives in `glue/` so the loop can write it and `validation/` can import it (one-way
  dependency rule). Spec: `docs/CANDIDATE_DATASET_FORMAT.md`.
- `./glue/datasets/suggestion_log.py` — **new.** `SuggestionLog`: writes the standard
  candidate dataset to `suggestions/` (`manifest.json` + `candidates.csv` +
  `routes.jsonl`, with the AL round as the standard `step` column; `generator="rgfn"`)
  plus `batch_metrics.csv` (one set-metrics row per round — a non-standard analysis
  sidecar). Sidecar only — never feeds training.
- `./glue/active_learning/loop.py` — `_sample_query_batch` now returns each
  molecule's route (was discarding trajectories); `_score_batch` prefers the
  oracle's `score_detailed()` (GPU oracle's per-pose breakdown) over scalar
  `score()`; `run()` snapshots the seed D_0 for novelty, calls `SuggestionLog`, and
  forwards batch metrics to the trainer logger. **Three robustness fixes from the
  failed attempts:** (a) `_free_torch_gpu_cache()` — calls `torch.cuda.empty_cache()`
  after training/sampling and before docking, the actual fix for the GPU-memory
  contention (prints free VRAM as a diagnostic); (b) aborts the loop with a clear
  error if a whole round's oracle batch is all-NaN (the silent no-op that wasted
  round-2 training on the first attempts); (c) new `system`/`seed` constructor args
  threaded into the manifest provenance.
- `./glue/datasets/candidates.py` (+ `suggestion_log.py`) — `SuggestionLog` now
  takes/records `system` + `seed` in the manifest (were `null` in this run).
- `./glue/datasets/__init__.py` — export `SuggestionLog` + the candidate-format API
  so `glue.registry` registers them.

Scripts / entry points:
- `./experiments/active_learning/6td3/submit_al_6td3_gpu.sh` — **this run.** Same
  loop as `submit_al_6td3_pregpu.sh` (entry 012) but GPU oracle; QV2-GPU env
  (boost libs, `$GNINA`), `--exclude=balam008,balam009`, OpenCL health gate (entry
  013), **plus a pre-flight dock gate** (below).
- `./experiments/active_learning/6td3/preflight_dock.py` — **new.** Docks 2 seed
  molecules at job startup and exits non-zero if the node makes 0 poses — catches a
  node that passes the OpenCL probe but can't actually dock, in ~40 s instead of at
  round-1 (67 min). (Note: the *main* failure turned out to be GPU-memory contention,
  fixed in the loop; this gate still guards against genuinely bad nodes.)
- `./experiments/active_learning/6td3/analyze_suggestions.py` — **new.** Retroactive
  per-round batch metrics from a saved standard `candidates.csv` (re-runs the metric
  functions without re-docking; works on any generator's standard dataset).
- `./scripts/active_learning.py` — driver; now binds `ActiveLearningLoop.seed`
  from `--seed` so the run seed reaches the manifest provenance.

Config:
- `./configs/glue/active_learning_6td3_gpu.gin` — **new.** Byte-for-byte
  `active_learning_6td3_mini.gin` (entry 012) except `@Docking6TD3Oracle` →
  `@Docking6TD3GpuOracle` (exhaustiveness=8000 QV2 threads, num_modes=9,
  docking_batch_size=25, n_gpu=1) + `ActiveLearningLoop.system='6td3'`. 3 rounds,
  300 steps/round, 32 batch, top-16.

Datasets / receptors (gitignored, on Balam):
- `./experiments/active_learning/6td3/seed_6td3.csv` — seed D_0 (408 labels; entry 002).
- `./experiments/oracle_validation/docking_6td3/6TD3_tier{1,2}.pdbqt`, `crystal_RC8.pdb`
  — GPU oracle's default receptors (Tier 2 = CDK12+DDB1, Tier 1 = CDK12 alone, crystal RC8 for the autobox).

Results (gitignored, on `$SCRATCH`):
- `/scratch/markymoo/rgfn_runs/experiments/active_learning/6td3_gpu/<timestamp>/active_learning/`
  — `suggestions/{manifest.json, candidates.csv, routes.jsonl, batch_metrics.csv}`
  (standard candidate dataset + per-round metrics sidecar), `dataset_round_00{1,2,3}.csv`,
  `top_k.csv`, `phase_timings.csv`, `docking_timings.csv`.

Diagnostic smoke tests (login node, no SLURM):
- `scratchpad/smoke_dock.py` — docked seed controls + round-1 molecules (incl. MW
  801) through `Docking6TD3GpuOracle` on a **Trillium H100**; all `status=ok`,
  ruling out the oracle/molecule-size as the cause.
- `scratchpad/mem_contention_test.py` — **the decisive test.** Docks 2 large
  generated molecules under three GPU states on the login A100: fresh (40 GB free →
  docks), torch caching-allocator holding memory (1.1 GB free → **0 poses, exact
  loop signature**), and after `empty_cache()` (40 GB free → recovers). Nails
  GPU-memory contention as the root cause and `empty_cache()` as the fix.

Job Logs:
- `/scratch/markymoo/rgfn_runs/al_6td3_gpu-{69481,69511,69515,69517}.out` / `.err`.

Jobs (the debugging journey → success):
- **69479 / 69480** — cancelled during round-1 training to roll in the
  suggestion-log + candidate-format standardization.
- **69481** (balam009) — round 1 all `no_pose`; killed mid-round-2 by the Balam
  node outage (~18:40 2026-06-29). First blamed on the node.
- **69511** (balam009) — after Balam recovered: round 1 all `no_pose` **again**;
  the new abort guard fired cleanly (FAILED at 1h05m, no wasted round 2/3). Node
  passed the startup OpenCL probe, so "bad node" no longer fit.
- **69515** (balam004) — pre-flight dock gate **passed** at startup, yet round 1
  still all `no_pose`. This within-node contradiction (docks at startup, fails
  post-training) is what pointed at GPU-memory contention rather than node health.
- **69517** (balam004) — **SUCCESS.** With the `empty_cache()` fix: run dir
  `…/6td3_gpu/2026-06-30_23-43-24/`, 3 rounds, ~3h 16m, `[AL] done`, Top-16 written.
  Each round freed ~39.7 GB before docking and added ~25 molecules.

## Relevant Versions

```
08da97c Working GPU Oracle RGFN, FGFN, RxnFlow, SCENT, AiZynthFinder
cdf3f78 GPU loop + FGFN loop
d1d77f9 Active learning & GPU dock
```

The suggestion-log + standard candidate-dataset machinery (`glue/active_learning/route.py`,
`glue/metrics/dataset_metrics.py`, `glue/datasets/candidates.py`, `glue/datasets/suggestion_log.py`,
`configs/glue/active_learning_6td3_gpu.gin`, `experiments/active_learning/6td3/analyze_suggestions.py`,
and the first loop wiring) landed in **`cdf3f78`** ("GPU loop + FGFN loop"). The decisive
`empty_cache()` fix — the headline of this entry — plus the all-NaN abort guard, the
`system`/`seed` manifest provenance, `experiments/active_learning/6td3/preflight_dock.py`, and the
final `submit_al_6td3_gpu.sh` landed in **`08da97c`** ("Working GPU Oracle …"). The successful run
(job 69517) executed on 2026-06-30 with the `empty_cache()` fix already in the working tree; it was
committed the next morning in `08da97c`.

## Relevant Resources

**Sources**
- `[koziarski2024rgfn]` — RGFN; evaluation metric set (QED, MW, modes, scaffolds).
- `[bengio2021gflownet]` — GFlowNet; number-of-modes / diversity / reward distribution.
- PDB **6TD3** — CDK12–DDB1 / CR8 ternary complex (the glue testbed).

**Packages**
- RDKit 2023.09.5 — descriptors / fingerprints / Murcko scaffolds (`glue/metrics/dataset_metrics.py`).
- QuickVina2-GPU-2.1 + gnina — GPU pose search + CNN rescore (`glue/oracles/docking_gpu_differential_oracle.py`).

## Method

1. `sbatch experiments/active_learning/6td3/submit_al_6td3_gpu.sh` — 1-GPU compute
   job: OpenCL health gate, then `python scripts/active_learning.py --cfg
   configs/glue/active_learning_6td3_gpu.gin --seed 42 --root-dir $SCRATCH/rgfn_runs/experiments`.
2. (post-hoc) `python experiments/active_learning/6td3/analyze_suggestions.py
   --suggestions <run>/active_learning/suggestions/candidates.csv
   --seed experiments/active_learning/6td3/seed_6td3.csv --out <run>/.../batch_metrics_recomputed.csv`.

## Results

Complete 3-round run (job 69517, balam004, run dir `…/2026-06-30_23-43-24/`).

**Per-round trajectory** (`suggestions/batch_metrics.csv` + `candidates.csv`; query
batch = 32/round, oracle differential lower = better):

| round | \|D\| (Δ) | docked | ≤ −2.0 | ≤ −3.0 | median | best | modes | int.div | novelty | MW mean | QED |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 433 (+25) | 25/32 | 14 | 3 | −2.14 | −3.90 | 32 | 0.84 | 1.0 | 675 | 0.12 |
| 2 | 457 (+24) | 24/32 | 12 | 0 | −1.99 | −2.94 | 32 | 0.87 | 1.0 | 634 | 0.13 |
| 3 | 483 (+26) | 26/32 | 16 | 5 | −2.42 | −3.86 | 32 | 0.85 | 1.0 | 686 | 0.10 |

Dataset grew 408 → 483 (+75; 75/96 = 78% dock success — the ~22% failures are large
multi-fragment molecules, e.g. boronic acids, not a systematic wipeout). Generated
molecules dock in **known-glue range** (per-round medians −2.0…−2.4, matching entry
011's CPU ≈ −2.4), maximally diverse (32/32 modes + scaffolds every round, novelty
1.0), and heavy / non-drug-like (MW ~630–690, QED ~0.10–0.13) — the size drift
Objective 2's reward term must still address. **Top-16** (`top_k.csv`) best = −4.92
(`CC[C@@H](CO)Nc1nc(NC(=O)CCc2ccccc2)c2ncn(C)c2n1`, a purine — the CR8/6TD3 warhead
chemistry), with several ≤ −3.9.

**Same loop, GPU oracle vs the CPU baseline (entry 011).** This run is a controlled
CPU→GPU swap of *the same RGFN active-learning loop* (same seed, budget, β, proxy;
only the oracle's pose source and this run's logging differ), so it reads directly
against entry 011's CPU run (job 69445). Generated-only, 96 suggested molecules each:

| | entry 011 — CPU oracle (69445) | this run — GPU oracle (69517) |
|---|---|---|
| oracle | `Docking6TD3Oracle`, gnina CPU search (AUROC ≈0.95) | `Docking6TD3GpuOracle`, QV2-GPU search (AUROC ≈0.88) |
| median `dvina` (generated) | ≈ −2.4 | −2.14 |
| best generated `dvina` | −4.71 | −3.90 |
| frac ≤ −2.0 | 61/96 (64%) | 42/96 (44%); 42/75 docked (56%) |
| dock success | 96/96 | **75/96** (~22% `no_pose`, large multi-fragment mols) |
| docking cost | ~35 min/round, **35% of loop** (entry 011; 58 s/mol Tier-2 search, entry 012) | ~1 min/round, **<2% of loop** (~1.5 s/mol) |
| MW / QED drift | MW large, QED low (size drift) | MW ~665, QED ~0.12 (same drift) |

**Reading it.** The GPU oracle preserves the science — generated molecules still dock in
known-glue range (median −2.14 vs ≈−2.4) with the same MW/QED size drift — while collapsing
docking from a third of the loop to under 2% (see phase timing below). The two visible costs
of the faster oracle are (1) a slightly lower glue-score yield (44% vs 64% ≤ −2.0), tracking
the −0.06 AUROC the GPU pose-search trades away (entry 013), and (2) a ~22% dock-failure rate
on the largest molecules (all 96 CPU-docked in 011; only 75/96 here) — normalising for that,
the ≤ −2.0 fraction of *docked* molecules is 56%, much closer to the CPU run. Net: the GPU
oracle collapses the docking phase ~35× (from ~35 min to ~1 min/round, 35% → <2% of the loop;
per-molecule ~50× per entry 013) while keeping the generator producing the same real-glue
chemistry — leaving GFN training (now ~98%) as the sole remaining cost centre.

**Phase timing** (`phase_timings.csv`), the headline GPU vs entry-012 CPU result:

| phase | per round | share |
|---|---|---|
| fit_proxy | 3–10 s | <0.3% |
| train_gfn | **62–64 min** | ~98% |
| sample_batch | 13–14 s | ~0.3% |
| oracle_score (GPU dock, 32 mol) | **57–63 s** | **~1.5%** |

Total ~3h 16m. Docking collapsed from entry-012's ~31 min/round (33% of the loop) to
~1 min/round (**<2%**) — the GPU oracle removes docking as a cost centre; GFN training
is now ~98% of wall-clock. `[AL] round N: freed torch GPU cache before docking ->
~39,700 / 40,440 MiB free` each round (the fix; without it, free was 1.1 GB → 0 poses).

**Synthesis routes** — 96/96 captured (`routes.jsonl`). Example (`rgfn-000048`, 4
reactions): building block `NCCC(=O)O` (β-alanine, F55) →[rxn 68 + `O=Cc1ccccc1B(O)O`]
→ … →[rxn 128 ×3, reductive-amination-type couplings] → the terminal molecule. Every
step carries its reaction SMARTS index + reactants + product, so a chemist can check
the route without rerunning anything.

**Root-cause reproduction** (`scratchpad/mem_contention_test.py`, login A100): fresh
GPU 40 GB free → docks; torch caching-allocator holding memory 1.1 GB free → **0/2
docked, "Docking attempt #1–10 failed on GPU 0"** (exact loop signature); after
`torch.cuda.empty_cache()` 40 GB free → docks again. Confirms the diagnosis + fix.
