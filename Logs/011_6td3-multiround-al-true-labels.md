# 6TD3 / RGFN — first multi-round active-learning loop with true oracle labels

**Date:** 2026-06-26 → 06-27, ~7pm start (5 h run, finished after midnight)

## Question

When we run the full active-learning loop for several rounds on a compute node — fit the fast scorer, train the generator, then actually dock the molecules it invents with the **CPU two-tier gnina oracle** (this is before we built the GPU docker) — do those generated molecules score like real molecular glues, and which parts of the loop eat the wall-clock?

## Context & Summary

**Context** — Entry 009 was the first time RGFN was wired to our validated 6TD3 oracle and trained, and it showed the generator *learns* (training loss fell ~24×, the bulk of generated molecules shifted toward better predicted scores). But that run had two gaps: it ran only **one** round, and the expensive docking step that would put *true* scores on the generated molecules was killed by the shared login node's CPU-time cap — so we had the proxy's optimistic predictions but no reality check. Entry 009's own takeaway flagged the fix: re-run on a Balam **compute** node (no CPU cap) for **multiple** rounds. This is that run.

**Summary** — We ran the 3-round active-learning loop on a Balam compute node against the validated 6TD3 docking oracle: fit the proxy on everything labelled so far, train RGFN for 300 steps against it, sample 32 fresh molecules, and **actually dock them** with the two-tier gnina oracle to get true glue scores — repeated three times, growing the labelled dataset from 408 to 504 molecules. We then asked two things: (1) are the generated molecules' *true* docking scores any good, and (2) where does the five hours of compute actually go. This is the **CPU-docking baseline** of the whole pipeline: the docking step here runs gnina's Tier-2 conformational search on the CPU (~58 s/mol), and the ~35%-of-wall-clock cost we measure below is exactly what motivated the GPU-docking arc that follows — entry 012 pins the bottleneck to the Tier-2 search, entry 013 builds a ~50× faster GPU pose-search oracle, and entry 014 re-runs *this same loop* on the GPU oracle. Read 011 → 012 → 013 → 014 as one story.

## Answer

The loop runs end-to-end on a compute node with the docking step intact, and the molecules RGFN generates dock as **genuine glue candidates** — their true two-tier differential lands squarely in known-glue territory (median ≈ −2.4, well past the −2.0 mark no molecule reached in entry 009's proxy-only view), with several scoring below −4. This is the first hard evidence that the generator produces real glue chemistry, not just molecules the proxy *thinks* are good; the proxy in entry 009 was actually *under*-selling what RGFN found. On cost: the loop is dominated by two phases — **training the generator (~65%)** and **docking the query batch (~35%)** — while fitting the proxy and sampling are together under half a percent of wall-clock, so any future speed-up has to come from those two, not the bookkeeping. One caveat carries over from entry 009: the strong-scoring molecules are large and multi-fragment, the same molecular-weight drift that motivates adding a drug-likeness term to the reward.

## Relevance to our Publication

This run delivers the core of **Objective 1 (first end-to-end RGFN run, the MVP)**: a multi-round active-learning loop that completes *with real oracle labels on generated molecules* — the prerequisite the login-node run (entry 009) couldn't clear. For Digital Discovery / J. Cheminformatics, reviewers will want the headline "top-k vs. oracle-calls, with a random-acquisition baseline" curve (cf. `[bengio2021gflownet]` Fig. 7); we now have its building block — a working multi-round loop whose per-round query batch is truly docked — plus a measured per-phase compute budget that tells us exactly how expensive each point on that curve is to produce. It also turns entry 009's open "anti-gaming" question (Objective 5) into a real datapoint: the generated molecules survive contact with the true oracle.

## Next Experiments

**Refining for publication** — Run ≥3 seeds and add a **random-acquisition baseline** on the same oracle budget to produce the top-k-vs-oracle-calls curve reviewers expect. Scale past 3 rounds (the cross-round trend here is positive but noisy at 32 molecules/round). Track molecular weight and QED on every round's batch to quantify the size drift directly. Instrument explicit **oracle-call counting** (Objective 1) into the loop output now, while runs are still short.

**Next steps in project** — Fold a **QED / ligand-efficiency term** into the reward (Objective 2) to counter the observed size drift, exactly the remedy the source papers prescribe (entry 009 amendment). Drive the **sEH** system (the fast GPU-docking oracle from entry 010) through the same multi-round loop so we have a second system end-to-end (Objective 4). Given that generator training and docking each dominate ~half the loop, profile whether shorter per-round GFN training or GPU-accelerated 6TD3 docking (Objective 2's go/no-go from entry 008 says we still need GPU-dock **plus** CNN rescore) shortens the loop without hurting the candidates.

# Re-creation

## Relevant Files

Root: `configs/glue/`, `glue/`, `experiments/active_learning/6td3/`

Scripts / entry points:
- `./scripts/active_learning.py` — driver for the outer active-learning loop (`[bengio2021gflownet]` Alg. 1); imports `glue` so gin can resolve our oracle/proxy/dataset/loop, then builds and runs `ActiveLearningLoop`.
- `./glue/active_learning/loop.py` — the loop itself (fit proxy → train RGFN → sample batch → score batch with oracle → accumulate). Writes per-round `dataset_round_*.csv`, `phase_timings.csv`, and the final `top_k.csv`; prints the per-phase timing summary.
- `./glue/proxies/` (`LearnedGlueProxy`) — the fast learned in-loop reward (MPNN over the atom graph), refit each round on the full labelled history and shared as one singleton between the GFN reward and the loop.
- `./glue/oracles/docking_6td3_oracle.py` — the expensive two-tier gnina oracle `O` (Tier 2 dock → CNN pose pick → Tier 1 rescore → Vina ΔT2−T1).
- `./glue/datasets/oracle_labeled.py` — the accumulating dataset `D`. **Fixed this run:** `top_k()` previously sorted `reverse=True` (largest label first), but labels are *more-negative = better* (`GlueOracle.higher_is_better = False`), so the "deliverable" was inverted — it returned the **worst** molecules. Now sorts ascending. The bug was cosmetic (deliverable file only); it never touched training, whose reward/proxy path is separate and correctly oriented (confirmed: generated molecules dock *well*, not badly).

Config:
- `./configs/glue/active_learning_6td3_mini.gin` — **this run.** 3 rounds, 300 inner GFN steps/round, 32-molecule query batch, β=8, gnina exhaustiveness 16 / 9 modes, top-k=16.

Datasets:
- `./experiments/active_learning/6td3/seed_6td3.csv` — seed `D_0`: 408 validated docking labels (160 known glues + 248 decoys), `label` = Vina ΔT2−T1 (entry 002, job 69271).

Receptors (gitignored binaries; staged into the oracle's expected path):
- `./research/preprocessing/docking_6td3/6TD3_tier{1,2}.pdbqt` — CDK12 only (Tier 1) / CDK12+DDB1 (Tier 2).
- `./research/preprocessing/docking_6td3/crystal_RC8.pdb` — native CR8 autobox reference.

Results (all gitignored, on `$SCRATCH`):
- `/scratch/markymoo/rgfn_runs/experiments/active_learning/6td3_mini/2026-06-26_19-04-50/` — the run dir.
  - `active_learning/dataset_round_00{1,2,3}.csv` — accumulating `D` after each round (440 / 472 / 504 rows incl. header).
  - `active_learning/phase_timings.csv` — per-round, per-phase seconds (source of all timing tables below).
  - `active_learning/top_k.csv` — **regenerated** with the fixed ascending sort (best = −4.92).
  - `train/checkpoints/`, `modes/`, `unique_molecules/`, `all_molecules/` — RGFN training artefacts.

Job Logs:
- `/scratch/markymoo/rgfn_runs/al_6td3_mini-69445.out` / `.err` — full stdout/stderr incl. the `[AL]` per-round timing trace.

## Relevant Versions

```
c447490 Refactor & timing
4f1386d prep for validation workflows
556a466 prep for validation workflows
```

The loop's timing instrumentation (`phase_timings.csv`, the `[AL] timing summary` print) is in `c447490`. The `top_k()` sort fix in `glue/datasets/oracle_labeled.py` was committed in `e07d953` ("small AL loop logs"). The config `configs/glue/active_learning_6td3_mini.gin` was staged in entry 009. Receptor `.pdbqt`/`.pdb` are intentionally gitignored; run outputs live on `$SCRATCH`.

**Committed:** the `top_k` fix in `glue/datasets/oracle_labeled.py` landed in `e07d953` ("small AL loop logs").

## Relevant Resources

**Sources**
- Active-learning loop / multi-round protocol and the GFlowNet-vs-random argument: Bengio et al., *GFlowNet Foundations* — `[bengio2021gflownet]`, Alg. 1 / A.5.2 (their MPNN-proxy-on-AutoDock molecule experiment is the template).
- RGFN generative model: Koziarski et al. — `[koziarski2024rgfn]` (reward `exp(β·score)`, sEH β=8 — the β this run uses).
- 6TD3 / CR8 system and seed labels: entry 002 (Balam job 69271); CR8 structure Słabicki et al., *Nature* 2020 (doi:10.1038/s41586-020-2133-z), `[slabicki2020cr8]`.

**Packages**
- torch 2.3.0+cu118, dgl 2.2.1+cu118, torch-geometric 2.5.3 — RGFN policy/proxy. dgl Graphbolt needs `module load cuda/11.8.0`.
- gnina v1.3.2 (`/scratch/markymoo/gnina/run_gnina.sh`) — the docking oracle.
- RDKit 2023.09.5 — embedding + QED. gin-config; wandb 0.26.1 (offline).
- Run in the `rgfn` conda env on Balam compute node **balam003** (1× A100, 32 CPU allocated).

## Method

1. **Submit to a compute node** — `sbatch experiments/active_learning/6td3/submit_al_6td3_mini.sh` (Balam job **69445**, partition `compute`, 1 GPU, 8 h limit, `--exclude=balam008`). The script redirects all run outputs / caches / wandb onto `$SCRATCH` (`$HOME` is read-only on compute nodes), `module load cuda/11.8.0`, `conda activate rgfn`, sets `GNINA=/scratch/markymoo/gnina/run_gnina.sh`, then runs `scripts/active_learning.py --cfg configs/glue/active_learning_6td3_mini.gin --seed 42 --root-dir $SCRATCH/rgfn_runs/experiments`.
2. **Loop, 3 rounds** — each round: refit `LearnedGlueProxy` on the full labelled `D`; train RGFN 300 steps against `M(x)^8`; sample 32 unique candidates; dock all 32 with the two-tier gnina oracle (exhaustiveness 16, 9 modes); accumulate `D_i = D̂_i ∪ D_{i-1}`.
3. **Post-run, two analyses** (off-cluster, no GPU): (a) per-round true-label distributions by diffing the cumulative `dataset_round_*.csv` (rows 0–407 = seed, 408–439 = round 1, …); (b) per-phase timing from `phase_timings.csv`.
4. **Fix + regenerate** — corrected `OracleLabeledDataset.top_k()` to sort ascending, and regenerated `top_k.csv` from `dataset_round_003.csv` (best-first).

## Results

3 rounds; proxy `M` = `LearnedGlueProxy` (MPNN), refit each round on the full history; β=8; query batch 32; oracle = `Docking6TD3Oracle` (Vina ΔT2−T1, more-negative = better glue). Hardware: 1× A100 on balam003. Job 69445, COMPLETED, exit 0:0, RunTime 05:03:01.

**The loop completed with the docking step intact.** Dataset grew **408 → 440 → 472 → 504** (+32 true-labelled molecules/round). No SIGXCPU (the entry-009 failure mode); the compute node has no per-process CPU-time cap.

**Generated molecules dock as real glues** (true ΔT2−T1; *generated-only*, i.e. the 32 new molecules each round, excluding seed):

| set | n | best (min) | median | mean | frac < −0.5 |
|---|---|---|---|---|---|
| seed `D_0` (known + decoy) | 408 | −4.92 | −1.03 | −1.25 | 0.72 |
| round 1 generated | 32 | −4.71 | **−2.37** | −2.45 | 1.00 |
| round 2 generated | 32 | −4.20 | −1.97 | −1.97 | 0.88 |
| round 3 generated | 32 | −4.56 | **−2.59** | −2.59 | 0.97 |

Known glues in the seed reach −2.2 to −2.6; **61 of 96** generated molecules score ≤ −2.0, the threshold entry 009's proxy predicted *nothing* would cross. The generated median (~−2.4) beats the seed median (−1.03). Cross-round trend is positive but non-monotonic at this batch size (R1 −2.37 → R2 −1.97 → R3 −2.59). Caveat: the top scorers are large, multi-fragment molecules (indole/tryptophan-like motifs, boronic acids), consistent with the entry-007/009 molecular-weight drift.

**Corrected top-k deliverable** (`top_k.csv`, over the full `D_504`, so it mixes seed and generated): best = **−4.92** (`CC[C@@H](CO)Nc1nc(NC(=O)CCc2ccccc2)c2ncn(C)c2n1`, a seed molecule), 16th = −3.84. The best *generated* molecule is −4.71 (`Cc1cc(N(CCc2c[nH]c3ccccc23)C(=O)c2ccc(F)c(F)c2F)cnc1N(C(=O)c1ccccc1C=O)c1nccc(C)c1N`). Before the fix, `top_k.csv` listed the worst molecules (rank 1 = +0.68).

**Proxy fit per round** (best validation MSE, from the `[AL]` log): round 1 = 0.263 (n=408), round 2 = 0.367 (n=440), round 3 = 0.476 (n=472). MSE rises as freshly generated, larger molecules enter `D` and broaden the label distribution.

### Timing breakdown (the headline cost picture)

Total wall-clock **5 h 01 m 48 s** (18,108 s). Four phases, in order each round: `fit_proxy → train_gfn → sample_batch → oracle_score`.

**By phase, summed over all 3 rounds:**

| phase | total | share | what it is |
|---|---|---|---|
| `train_gfn` | 3 h 15 m 12 s | **64.7 %** | training RGFN 300 steps/round against the proxy |
| `oracle_score` | 1 h 45 m 27 s | **34.9 %** | docking the 32-molecule query batch (two-tier gnina) |
| `sample_batch` | 42.0 s | 0.23 % | drawing 32 molecules from the trained policy |
| `fit_proxy` | 27.0 s | 0.15 % | refitting the MPNN proxy on the full history |
| **TOTAL** | **5 h 01 m 48 s** | 100 % | |

**Two phases are everything.** Generator training + docking = **99.6 %** of wall-clock; proxy fitting and sampling together are **69 s** across the whole run — rounding error. Any speed-up must come from GFN training or docking.

**Per round, and derived rates:**

| round | fit_proxy | train_gfn | sample | oracle_score | round total | gfn /step | oracle /mol |
|---|---|---|---|---|---|---|---|
| 1 | 10.0 s | 64.6 min | 12.7 s | 35.8 min | 100.8 min | 12.91 s | 67.2 s |
| 2 | 6.4 s | 64.9 min | 14.9 s | 28.1 min | 93.4 min | 12.98 s | 52.8 s |
| 3 | 10.6 s | 65.7 min | 14.5 s | 41.5 min | 107.6 min | 13.14 s | 77.8 s |
| **avg** | 9.0 s | 65.1 min | 14.0 s | 35.2 min | 100.6 min | **13.01 s** | **65.9 s** |

Readings: (1) **GFN training is steady** — ~13.0 s/step regardless of round, so its cost scales purely with `n_iterations` (300 here). (2) **Docking is the variable phase** — 52.8 → 77.8 s/mol across rounds, a ~1.5× swing; the slowest round (R3, 77.8 s/mol) is also the one that generated the largest, best-scoring molecules, consistent with bigger ligands docking slower. (3) At these rates, the levers are clear: halving GFN steps would cut ~32 min/round; the docking cost is set by query-batch size × per-molecule docking time (the entry-008 GPU-dock + CNN-rescore path is the route to shrinking the latter).
