# 6TD3 / RGFN — pre-GPU-exploration: instrumented docking sub-step timing baseline

**Date:** 2026-06-28, ~afternoon

## Question

Within the docking step of our active-learning loop, which sub-step actually eats the time — building the 3D molecule, the big conformational search, picking the best pose, or the second-tier rescore — and how does that cost scale with molecule size?

## Context & Summary

**Context** — The first full multi-round run (entry 011, job 69445) measured where the *loop* spends its five hours: generator training ~65%, docking ~35%, everything else <0.5%. But the docking 35% was a single black-box number — `phase_timings.csv` timed the whole oracle call, and gnina's own output was captured and thrown away, so we couldn't say what *inside* docking was slow. We strongly suspect the Tier-2 conformational search (exhaustiveness 16) dominates — the only circumstantial evidence is that per-molecule docking cost rose 52.8 → 77.8 s/mol in the round that generated the largest molecules — but we have never measured it. Before spending effort on a GPU docker to speed that step up, we need the baseline it would be measured against.

**Summary** — We added sub-step instrumentation to the 6TD3 docking oracle: it now writes `docking_timings.csv` splitting each round's docking into **embed** (RDKit 3D build), **tier2_dock** (the gnina conformational search), **pose_select** (parse + pick best pose), and **tier1_rescore** (the `--score_only` second tier), each row tagged with the round and molecule count so seconds/molecule is recoverable per step. This run re-executes the *identical* 3-round mini loop from entry 011 — same config, same seed — purely to capture that breakdown as the **pre-GPU baseline**. The next experiment will try to replace the Tier-2 search with GPU docking; this entry is the "before" it gets compared to.

## Answer

Within docking, the Tier-2 conformational search is essentially the *entire* cost: **`tier2_dock` = 99.3%** of the docking phase (58.4 s/molecule averaged over 96 molecule-docks), while embedding (0.6%), the Tier-1 rescore (0.2%), and pose selection (~0%) together come to under 1%. The four sub-steps sum to the `oracle_score` phase total *exactly* (94.1 min, 0% unaccounted), so the instrumentation accounts for the whole phase. The search cost scales with molecule size — 52–53 s/mol in rounds 1–2 rose to 70 s/mol in round 3, the round that generated the largest molecules, the signature of a conformational search (more rotatable bonds → bigger search space). The earlier worry that conformer embedding might be a hidden bottleneck (entry 004) is **refuted here**: embedding is 0.33 s/mol, ~180× cheaper than the search. The practical consequence: docking is 33.6% of loop wall-clock and `tier2_dock` is 99.3% of docking, so the Tier-2 search alone is **~33% of the entire loop** — that is the ceiling a GPU docker (replacing just that step, keeping the gnina CNN rescore per entry 008) could claw back. Embedding and rescore are rounding error and need no attention.

## Relevance to our Publication

This is groundwork, not a headline result, but it de-risks an Objective-2 decision. Entry 008's go/no-go concluded the 6TD3 oracle must keep gnina's CNN pose-picker (the 6TD3 oracle needs a GPU-dock **plus CNN-rescore** path, not pure QuickVina2-GPU), so any speed-up has to target the Tier-2 search specifically while preserving CNN rescoring. Knowing the exact sub-step budget tells us the *ceiling* on what a GPU docker can save — if Tier-2 is 90% of docking, replacing it roughly halves the loop (docking is 35% of wall-clock); if embedding is a big slice, the GPU docker alone won't help much. That number decides whether the GPU-dock work is worth it before we build it, and it feeds the per-oracle-call cost that the eventual top-k-vs-oracle-calls curve (entry 011 relevance, `[bengio2021gflownet]` Fig. 7) is plotted against.

## Next Experiments

**Refining for publication** — Once the breakdown is in hand, this timing instrumentation rides along on every future docking run for free (the loop enables it automatically), so the eventual headline run reports a real compute budget rather than an eyeballed one.

**Next steps in project** — The actual GPU-exploration follow-up: stand up a GPU docking path for the Tier-2 search (candidate: the QuickVina2-GPU build already wired for the sEH oracle, entry 010) **followed by a gnina CNN rescore** to keep the entry-008 pose-picking intact, then re-run this same mini loop and diff `docking_timings.csv` against this baseline to quantify the speed-up. If Tier-2 dominates as expected, also test whether dropping exhaustiveness (16 → 8) trades acceptable discrimination for wall-clock.

# Re-creation

## Relevant Files

Root: `glue/`, `configs/glue/`, `experiments/active_learning/6td3/`

New / changed code (this entry):
- `./glue/oracles/step_timing.py` — **new.** `OracleStepTimer`: a disabled-by-default (silent no-op) sub-step timer, companion to `glue/active_learning/timing.PhaseTimer`. Appends `(round, step, seconds, n_molecules)` on each sub-step's completion (crash-safe), prints a `[dock]` line live.
- `./glue/oracles/docking_6td3_oracle.py` — added `enable_step_timing(csv_path)` and wrapped `score()`'s four sub-steps (`embed`, `tier2_dock`, `pose_select`, `tier1_rescore`) in the timer. Docking logic itself is unchanged — like-for-like with entry 011.
- `./glue/active_learning/loop.py` — opt-in hook: if the oracle implements `enable_step_timing`, the loop points it at `<run>/active_learning/docking_timings.csv`. Loop stays oracle-agnostic (mock/other oracles unaffected).

Scripts / entry points:
- `./experiments/active_learning/6td3/submit_al_6td3_pregpu.sh` — **this run.** Same 3-round mini loop as `submit_al_6td3_mini.sh` (entry 011), distinct job name `al_6td3_pregpu`; now emits `docking_timings.csv`.
- `./scripts/active_learning.py`, `./glue/active_learning/loop.py`, `./glue/oracles/docking_6td3_oracle.py` — driver / loop / oracle (see entry 011 for roles).

Config (unchanged from entry 011, so this is a like-for-like baseline):
- `./configs/glue/active_learning_6td3_mini.gin` — 3 rounds, 300 GFN steps/round, 32-molecule query batch, β=8, gnina exhaustiveness 16 / 9 modes, top-k=16.

Datasets / receptors:
- `./experiments/active_learning/6td3/seed_6td3.csv` — seed `D_0` (408 labels; entry 002).
- `./experiments/oracle_validation/docking_6td3/6TD3_tier{1,2}.pdbqt`, `crystal_RC8.pdb` — the oracle's default receptor/crystal paths (gitignored; present on Balam).

Results (gitignored, on `$SCRATCH`):
- `/scratch/markymoo/rgfn_runs/experiments/active_learning/6td3_mini/<timestamp>/active_learning/docking_timings.csv` — **the new artifact this run exists to produce.**
- alongside it: `phase_timings.csv`, `dataset_round_00{1,2,3}.csv`, `top_k.csv` (now correctly sorted, post entry-011 fix).

Job Logs:
- `/scratch/markymoo/rgfn_runs/al_6td3_pregpu-69451.out` / `.err` — stdout/stderr; the `[dock]` sub-step lines also appear here live.

Job: **SLURM job 69451** — COMPLETED, exit 0:0, elapsed 04:41:05 on balam003; run dir `…/6td3_mini/2026-06-28_12-47-09/`. First attempt (job 69450) failed at startup in 4 s — submitted from the read-only `$HOME` repo dir, so SLURM couldn't open its `%x-%j.out`; fixed by hardcoding absolute `$SCRATCH` log paths in the submit script and resubmitting as 69451.

## Relevant Versions

```
c447490 Refactor & timing
4f1386d prep for validation workflows
556a466 prep for validation workflows
```

Committed in `d1d77f9` ("Active learning & GPU dock"): `glue/oracles/step_timing.py` (new), `glue/oracles/docking_6td3_oracle.py`, `glue/active_learning/loop.py`, `experiments/active_learning/6td3/submit_al_6td3_pregpu.sh`. (The entry-011 `top_k` fix in `glue/datasets/oracle_labeled.py` was committed earlier, in `e07d953`.)

## Relevant Resources

**Sources**
- Active-learning loop / per-oracle-call budgeting: Bengio et al., *GFlowNet Foundations* — `[bengio2021gflownet]`, Alg. 1 / Fig. 7.
- The signal being computed (Vina ΔT2−T1) and CNN pose-picking that must be preserved: entries 006 (six-way ablation), 008 (pose-selection go/no-go).
- The loop-level timing this refines: entry 011 (job 69445).

**Packages**
- gnina v1.3.2 (`/scratch/markymoo/gnina/run_gnina.sh`) — the docking engine (Tier-2 search + Tier-1 `--score_only`).
- RDKit 2023.09.5 — `embed` sub-step (3D embed + MMFF). torch 2.3.0+cu118 / dgl 2.2.1+cu118 — RGFN. Run in the `rgfn` conda env on a Balam compute node.

## Method

1. **Instrument** — add `OracleStepTimer` (`glue/oracles/step_timing.py`); wrap the four sub-steps of `Docking6TD3Oracle.score()`; add the loop's opt-in `enable_step_timing` hook.
2. **Validate locally** — `py_compile` all three files; in the `rgfn` env, `import glue` (gin resolves the oracle), confirm the disabled timer is a no-op and the enabled timer writes the expected CSV; `bash -n` the submit script. `[done this session]`
3. **Submit** — `sbatch experiments/active_learning/6td3/submit_al_6td3_pregpu.sh` (compute node, no login CPU cap).
4. **Analyze** — read `docking_timings.csv`: per-step share of the docking phase, seconds/molecule per step, and the tier2_dock-vs-molecule-size trend across rounds; cross-check the docking-phase total against entry 011's `oracle_score` to confirm the breakdown sums correctly.

## Results

3-round mini loop, identical config/seed to entry 011. Job 69451, COMPLETED, 4 h 41 m on one A100 (balam003). The loop-level phase split **reproduces entry 011**: `train_gfn` 66.1% / `oracle_score` 33.6% / `sample_batch` 0.2% / `fit_proxy` 0.1% (TOTAL 4 h 40 m). The new artifact, `docking_timings.csv`, breaks the `oracle_score` phase open.

**Docking sub-steps, summed over 3 rounds (n = 96 molecule-docks):**

| sub-step | total | % of docking | s/molecule | what it is |
|---|---|---|---|---|
| **`tier2_dock`** | **93 m 24 s** | **99.3 %** | **58.37** | gnina conformational search vs CDK12+DDB1 (exhaustiveness 16, 9 modes) |
| `embed` | 32.1 s | 0.6 % | 0.33 | RDKit 3D embed + MMFF |
| `tier1_rescore` | 9.4 s | 0.2 % | 0.10 | gnina `--score_only` vs CDK12 (no search) |
| `pose_select` | 0.1 s | 0.0 % | 0.001 | parse poses + pick max-CNNscore |
| **sum of sub-steps** | **94 m 06 s** | 100 % | 58.81 | — |

**Reconciliation:** the sub-steps sum to **5646 s**; the `oracle_score` phase total in `phase_timings.csv` is **5646 s** (28 m 31 s + 28 m 03 s + 37 m 32 s) — **0 s / 0.0% unaccounted**. The instrumentation captures the entire phase (SDF writes / temp-dir / parsing are sub-second, folded into their enclosing steps).

**`tier2_dock` scales with molecule size** (per-round s/mol):

| round | tier2_dock s/mol | note |
|---|---|---|
| 1 | 53.11 | — |
| 2 | 52.19 | — |
| 3 | **69.83** | round that generated the largest molecules (entry 011: R3 had the most-negative scores and biggest ligands) |

The ~1.3× jump in round 3, tracking molecule size, is consistent with a conformational search whose cost grows with rotatable bonds / atom count — and is the same size-driven swing seen at the phase level in entry 011 (52.8 → 77.8 s/mol there). Embedding rose only 0.27 → 0.44 s/mol over the same rounds: real but negligible.

**Implication for the GPU-dock decision.** `oracle_score` is 33.6% of loop wall-clock and `tier2_dock` is 99.3% of that → the Tier-2 search is **≈ 33.4% of the entire loop**. Replacing it with a GPU docker (and re-attaching the gnina CNN rescore that entry 008 found load-bearing) sets the speed-up ceiling at about one-third of total wall-clock; nothing else in docking is worth optimizing.
