# 6TD3 / RGFN — single-seed pilot of the oracle-call curve (RGFN vs random acquisition)

**Date:** 2026-07-06, ~3pm

## Question

Before committing to the full multi-seed comparison, does the active-learning loop — now
instrumented to count expensive oracle calls and to record the best molecules found so far —
run end to end on the *real* docking oracle for both the learned RGFN generator and a
random-molecule baseline, producing the data behind the "how good are our molecules per
oracle call?" curve?

## Context & Summary

**Context.** One of the questions reviewers will ask (`docs/RESEARCH_CONTEXT.md`, Objective 1;
`[bengio2021gflownet]` Fig. 7) is whether RGFN actually finds better molecules *per expensive
oracle call* than simply trying molecules at random. Answering it needs two things recorded
while the loop runs and impossible to reconstruct afterwards: how many times we called the
expensive docking oracle, and what the best candidates were at each point along the way. Until
now the loop threw the call count away. Entry 014 (job 69517) established that the GPU docking
oracle runs inside the loop for the CDK12-DDB1 (6TD3) glue system over 3 rounds; this entry
reuses that exact validated setup to shake out the new instrumentation on the real oracle
before scaling up.

**Summary.** We added oracle-call counting and a running "best-so-far" record to the loop, plus
a **random-acquisition baseline** — a control that proposes molecules by sampling the same
reaction building blocks uniformly at random instead of using the trained RGFN policy, with the
proxy and generator training switched off so the only thing that changes is *how the next batch
of molecules is chosen*. This pilot runs both arms once (single seed) on the 6TD3 GPU docking
oracle: the learned RGFN loop and the random baseline, each for 3 rounds of 32 molecules (96
oracle calls). The goal is to confirm both arms complete on the real oracle and write the new
per-round trace, and to eyeball whether the learned generator is already pulling ahead of
random — not yet to make a statistically-backed claim, which needs the multi-seed run.

## Answer

Both arms completed on the real 6TD3 GPU docking oracle and the new instrumentation recorded
the oracle-call budget and best-so-far molecules correctly, so the curve data the multi-seed
campaign needs is now proven to come out of a real run. Even at this tiny single-seed budget
(96 oracle calls, 3 rounds) the intended signal is already visible: the learned RGFN generator
drives the top-16 mean docking differential down about **five times faster per oracle call**
than the random baseline (−3.21 → −3.89 vs −3.21 → −3.35), both starting from the identical
seed baseline. Neither arm beat the seed set's single best molecule in only 96 calls (both
stayed at −4.915, the CR8 purine already in D_0) — which is exactly why we log the top-k *mean*,
not just the best. One real bug surfaced: two arms of the same config launched at once collided
on one run directory (identical timestamp), so their non-appended per-round artifacts clobbered
each other; the oracle-call trace survived (it is append-only and arm-tagged), and the run-dir
naming has been fixed to include the arm + seed.

## Relevance to our Publication

This is the enabling step for the headline oracle-efficiency figure that a **Digital Discovery /
J. Cheminformatics** reviewer will expect (the analog of `[bengio2021gflownet]` Fig. 7): a plot
of best reward found versus number of expensive-oracle calls, RGFN vs a random-acquisition
baseline. The pilot de-risks the real multi-seed campaign — confirming the instrumentation and
both arms work on the true docking oracle — so the figure itself is one launch away.

## Next Experiments

**Refining for publication**
- Run the full campaign: **3 seeds × 10 rounds (320 oracle calls) × 2 arms** on 6TD3
  (`launch_curve_6td3.sh`), then draw the curve with mean±std error bands over seeds.
- Report the same curve on a second system (sEH GPU oracle, entry 010) for generalization.

**Next steps in project**
- Fold a QED / ligand-efficiency term into the reward to counter the persistent size drift
  (Objective 2), now measurable per round alongside the oracle-call trace.
- Use the trace as the substrate for the anti-gaming check (in-loop proxy vs high-fidelity
  oracle correlation on the top-k) in the evaluation suite (Objective 5).

# Re-creation

## Relevant Files

Root: `./` (repository root).

**Scripts / code**
- `./glue/active_learning/acquisition_trace.py` — `AcquisitionTrace`: writes the per-round
  `oracle_calls.csv` (cumulative oracle calls, running Top-K mean/best, best-molecule SMILES,
  round-0 seed baseline). The data behind the curve.
- `./glue/active_learning/loop.py` — records the trace each round; new `acquisition` param
  (`"policy"` = learned RGFN loop; `"random"` = uniform-policy baseline that skips proxy-fit +
  GFN-training and swaps in `RandomSampler`/`UniformPolicy` over the same reaction env).
- `./scripts/active_learning.py` — new `--acquisition {policy,random}` flag.
- `./validation/harness/acquisition_curve.py` — aggregates `oracle_calls.csv` across seeds ×
  arms → curve CSV + PNG (Okabe–Ito, ±std band).
- `./experiments/active_learning/6td3/submit_curve_6td3.sh` — parameterized submit
  (`ACQ`/`SEED`/`CFG` env vars); this pilot passes `CFG=configs/glue/active_learning_6td3_gpu.gin`.
- `./experiments/active_learning/6td3/launch_curve_6td3.sh` — 3-seed × 2-arm campaign launcher
  (for the follow-up, not this pilot).
- `./experiments/active_learning/6td3/preflight_dock.py` — startup dock gate (entry 014).

**Config**
- `./configs/glue/active_learning_6td3_gpu.gin` — the validated 3-round GPU-oracle 6TD3 loop
  (entry 014): `@Docking6TD3GpuOracle`, β=8, 300 GFN steps/round, 32-molecule query batch,
  top-k=16. Reused unchanged so the pilot is a clean shakeout of the new instrumentation.
- `./configs/glue/active_learning_6td3_gpu_curve.gin` — the 10-round / top-k=100 variant for the
  full campaign (not used by this pilot).

**Datasets / receptors** (gitignored, on Balam)
- `./experiments/active_learning/6td3/seed_6td3.csv` — seed D_0 (408 validated docking labels; entry 002).
- `./experiments/oracle_validation/docking_6td3/6TD3_tier{1,2}.pdbqt`, `crystal_RC8.pdb` — GPU
  oracle receptors (Tier 1 = CDK12; Tier 2 = CDK12+DDB1; RC8 crystal ligand for the autobox).

**Results**
- `./validation/results/6td3_acquisition_curve_pilot/` — **committed pilot output.**
  `oracle_calls_pilot.csv` (recovered + baseline-symmetrized trace, both arms),
  `acquisition_curve_topk_{mean,best}.{csv,png}` (the curve + aggregated table).
- `/scratch/markymoo/rgfn_runs/experiments/active_learning/6td3_gpu/2026-07-06_16-06-05/active_learning/`
  — the raw run dir (gitignored). `oracle_calls.csv` (interleaved both arms — the collision),
  `dataset_round_00{1,2,3}.csv`, `top_k.csv`, `suggestions/`, `phase_timings.csv`,
  `docking_timings.csv` (per-arm artifacts clobbered by the collision; see Results).

**Job Logs** (gitignored, on `$SCRATCH`)
- `/scratch/markymoo/rgfn_runs/al_6td3_pilot_{policy,random}-{69945,69946}.{out,err}`.

## Relevant Versions

```
[TODO — add commit hash after committing this session's code + this log]
```

The oracle-call instrumentation, random-acquisition arm, curve aggregator, curve config, and
submit/launch scripts are **not yet committed** (branch `GPU-Dock`). Files to commit:
`glue/active_learning/acquisition_trace.py`, `glue/active_learning/loop.py`,
`glue/active_learning/__init__.py`, `scripts/active_learning.py`,
`validation/harness/acquisition_curve.py`, `configs/glue/active_learning_6td3_gpu_curve.gin`,
`experiments/active_learning/6td3/{submit_curve_6td3.sh,launch_curve_6td3.sh}`,
`docs/REFACTOR_LOG.md`, and this log. **Can you commit these?** Once you do, tell me and I'll
fill in the hash.

## Relevant Resources

**Sources**
- `[bengio2021gflownet]` — GFlowNet active-learning loop (Alg. 1) and the oracle-efficiency /
  top-k-vs-oracle-calls comparison against random acquisition (Fig. 7) this pilot enables.
- `[koziarski2024rgfn]` — RGFN; the reaction action space the random baseline samples uniformly.
- PDB **6TD3** — CDK12–DDB1 / CR8 ternary complex (the glue testbed).

**Packages**
- QuickVina2-GPU-2.1 + gnina — GPU pose search + CNN rescore
  (`glue/oracles/docking_gpu_differential_oracle.py`).
- torch 2.3.0+cu118 — the `empty_cache()` GPU-memory-contention fix carried from entry 014.

## Method

1. Instrumentation validated locally first (login node, mock oracle, both arms): the loop wrote
   `oracle_calls.csv` with the round-0 baseline + per-round cumulative calls and Top-K; the
   random arm correctly skipped proxy-fit + GFN-training; the aggregator produced a curve CSV+PNG.
2. Submitted both pilot arms (single seed 42) on the real GPU oracle:
   ```bash
   SUB=experiments/active_learning/6td3/submit_curve_6td3.sh
   CFG=configs/glue/active_learning_6td3_gpu.gin
   sbatch --job-name=al_6td3_pilot_policy --export=ALL,ACQ=policy,SEED=42,CFG=$CFG $SUB  # job 69945
   sbatch --job-name=al_6td3_pilot_random --export=ALL,ACQ=random,SEED=42,CFG=$CFG $SUB  # job 69946
   ```
3. (post-hoc, when both COMPLETE) draw the pilot curve:
   ```bash
   python -m validation.harness.acquisition_curve \
       --runs $SCRATCH/rgfn_runs/experiments/active_learning/6td3_gpu \
       --out validation/results/6td3_acquisition_curve_pilot --metric topk_best
   ```

## Results

Jobs 69945 (policy) / 69946 (random), single seed 42, on balam010. Both **COMPLETED**.

**Oracle-efficiency curve** (top-16 mean of the 6TD3 differential over the accumulated dataset
`D`; lower = better binding; the round-0 point is the shared seed `D_0` baseline):

| cumulative oracle calls | RGFN (learned) top-16 mean | random top-16 mean |
|---|---|---|
| 0 (seed D_0) | −3.209 | −3.209 |
| 32 (round 1) | **−3.487** | −3.286 |
| 64 (round 2) | **−3.874** | −3.303 |
| 96 (round 3) | **−3.889** | −3.352 |

Improvement over the seed baseline after 96 calls: RGFN **−0.68**, random **−0.14** — the learned
acquisition improves the top-k mean ≈5× faster per oracle call. Top-16 *best* stayed at −4.915
for both arms across all rounds (D_0 already contained the best molecule, a CR8-like purine
`CC[C@@H](CO)Nc1nc(NC(=O)CCc2ccccc2)c2ncn(C)c2n1`); the 96-call budget was too small for either
arm to beat it, so the *mean* is the sensitive metric at this scale.
Plots: `validation/results/6td3_acquisition_curve_pilot/acquisition_curve_topk_{mean,best}.png`.

**Per-round docking** (query batch = 32/round; `oracle_calls.csv` + policy `.out`):

| round | RGFN labelled / 32 | RGFN top16 mean | random labelled / 32 | random top16 mean |
|---|---|---|---|---|
| 1 | 29 | −3.487 | 25 | −3.286 |
| 2 | 31 | −3.874 | 28 | −3.303 |
| 3 | 27 | −3.889 | 28 | −3.352 |

RGFN round-3 batch: 32/32 modes, MW mean 654, internal diversity 0.84 (the size drift of
entries 011/014 persists — Objective 2).

**Phase timing** (policy arm, `phase_timings.csv`): total **3h 14m 53s**; `train_gfn`
**97.7%** (3h 10m), `oracle_score` (GPU dock, 32 mol/round) **1.6%** (3m 12s), `sample_batch`
0.3%, `fit_proxy` 0.3%. Docking stays <2% of the loop — reproducing entry 014 (job 69517) on
the identical config. Random arm wall-clock **4m 27s** (no proxy-fit / no GFN training; docking
+ sampling only), i.e. the baseline arm is ~44× cheaper than the learned arm here.

**Run-directory collision (bug found + fixed).** Both jobs used the same config, so
`run_name = active_learning/6td3_gpu/<timestamp>` resolved to one shared directory
(`.../6td3_gpu/2026-07-06_16-06-05/`) at second-granularity — the two processes then wrote to
the same output paths. The append-only, arm-tagged `oracle_calls.csv` survived (both arms'
rows intact, distinguishable by the `acquisition` column; only the random arm's round-0 header
row was overwritten and was reconstructed for the plot, being identical to the shared baseline),
but the rewrite-each-round artifacts (`dataset_round_*.csv`, `top_k.csv`, `suggestions/`) reflect
only the last writer (the policy arm). **Fix:** `scripts/active_learning.py` now builds
`run_name = <config>/<acq>_seed<seed>/<timestamp>`, so concurrent arms/seeds of one config never
share a directory — required before the 6-job (`3 seed × 2 arm`) campaign. Not re-run: the pilot's
curve data is valid as-is, and the clobbered per-arm artifacts are not needed for the curve.
