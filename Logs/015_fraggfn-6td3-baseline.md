# 6TD3 / FragGFN — non-synthesizable fragment-GFlowNet baseline through the active-learning loop
**Date:** 2026-06-29, ~5pm (run completed ~5:48pm; Balam job 69482, 27m33s elapsed)

## Question

Can we generate molecular-glue candidates for 6TD3 with a **fragment-based** GFlowNet — the non-synthesizable baseline from the RGFN paper — driven through the *same* active-learning loop, oracle, and budget as our RGFN pipeline, so the two can be compared head-to-head?

## Context & Summary

**Context.** Our central claim is that RGFN's *synthesizability by construction* (it builds molecules from real chemical reactions, so every candidate comes with a synthesis route) is worth having. The way reviewers test that claim is to ask: does it actually beat a generator that has **no** such constraint, on the same scoring function and the same compute budget? The RGFN paper's own foil for this is a fragment-based GFlowNet ("FGFN") that snaps fragments together at arbitrary points — chemically valid graphs, but with no guaranteed way to make them. This is Objective 4/5 in `docs/RESEARCH_CONTEXT.md` ("add a non-synthesizable baseline generator on the same oracle and budget"). We had scaffolding for it (an empty `validation/generators/fraggfn/` and a stub install script) but no implementation. Entry 014 produced the RGFN side of this comparison (the GPU-oracle active-learning run with per-round suggestion logging); this entry builds the matching FragGFN entrant.

**Summary.** We implemented FragGFN using Recursion's `gflownet` (its fragment-assembly environment), wired it to the *same* learned proxy `M` and the *same* docking oracle as RGFN, and ran it through a faithful copy of the active-learning loop (`[bengio2021gflownet]` Alg. 1). Because Recursion's library needs an older Python/torch than our main environment, FragGFN lives in its own conda env and reaches the shared docking oracle across the environment boundary through a small "bridge" — a command-line scorer that runs in our main env and is the single shared scoring standard every benchmark entrant will use. Every molecule FragGFN suggests, and every round, is logged in the project's standard candidate-dataset format (marked non-synthesizable) so its output drops in next to RGFN's for a direct comparison. This entry covers the implementation, a full local dry-run on a cheap stand-in oracle, **and the completed headline run on Balam with the real GPU docking oracle** (job 69482), whose results anchor the head-to-head below.

## Answer

The FragGFN baseline is implemented and the full 3-round loop **ran to completion on Balam with the real GPU docking oracle** (job 69482): it seeded from the identical 408-molecule 6TD3 dataset RGFN uses, trained the fragment generator against the refit proxy, sampled a query batch of fragment-assembled molecules each round, docked them through the shared oracle bridge (32/32 scored every round), accumulated the labels, and emitted a conformant standard dataset with `has_route=0` (the headline difference from RGFN).

The science answer: **the non-synthesizable baseline generates molecules that dock as real glues, competitively with RGFN** — best differential −4.86 (vs RGFN-GPU's −3.90, entry 014), median −2.06 (vs −2.14), 54% of candidates beating the −2.0 glue cutoff (vs 44%), all fully diverse (32 distinct scaffolds per 32-molecule round, novelty 1.0). It also reproduces RGFN's two known weaknesses — large molecules (MW ≈720) and low QED (≈0.15). So on raw oracle score FragGFN is in the same league as RGFN; the differentiator RGFN must lean on is **synthesizability** (FragGFN's molecules have no route), not glue-score superiority. Crucially, this comparison is now **apples-to-apples**: the matched-oracle RGFN GPU run has since completed (entry 014, job 69517), so FragGFN-GPU is compared against RGFN on the *identical* `Docking6TD3GpuOracle` (AUROC ≈0.88) rather than against the earlier CPU-oracle RGFN run (entry 011, ≈0.95). Entry 016 folds RxnFlow into this same matched-oracle set for the full three-way.

## Relevance to our Publication

This is the baseline the synthesizability story rests on. For Digital Discovery / JCIM (and any workshop version), reviewers will ask whether "synthesizable" is doing real work or is just a label — the only convincing answer is RGFN vs. a non-synthesizable generator on one oracle and one budget, the analogue of the GFlowNet-beats-random comparison in `[bengio2021gflownet]` Fig. 7. By holding the oracle, seed, budget, proxy, and β identical and changing only the generator's action space, this entry makes that comparison clean: any difference in glue quality is attributable to the reaction-grounded action space, not to a different reward or more compute.

## Next Experiments

**Refining for publication.**
- **Matched-oracle head-to-head — DONE:** the RGFN GPU run on a healthy node has since completed (entry 014, job 69517), so FragGFN-GPU vs RGFN-GPU is now apples-to-apples on the identical oracle (see the updated table below). The remaining plotting step is to put both entrants' per-round `dvina` distributions + Top-K on one figure.
- Repeat over ≥3 seeds for error bars (Objective 4).
- Add the synthesizability-advantage panel: FragGFN has no routes by construction; quantify RGFN's by-construction route coverage / SA-score advantage on the same candidates (Objective 5) — this is where RGFN should win even though glue scores are comparable.

**Next steps in project.**
- Bring the other planned baselines (SynFlowNet, VAE-BO) online through the same oracle bridge, then run the full validation harness that reads every entrant's standard dataset and produces the comparison tables.

# Re-creation

## Relevant Files

Root: `./` (repo root).

**Install**
- `external/setup_fraggfn.sh` — creates the `fraggfn` conda env (Python 3.10), clones Recursion's `gflownet` at the pinned commit into `external/gflownet/` (git-ignored, not vendored), installs torch 2.1.2+cu118 + the cu118 PyG extension wheels + gflownet, pins `numpy<2` (torch 2.1.2 ABI), and import-smokes the env.

**Oracle bridge (shared scoring standard; runs in the `rgfn` env)**
- `scripts/score_batch.py` — scores a SMILES file with a named glue oracle (`docking_6td3_gpu`/`docking_seh`/`mock`), writes a labels CSV, and (optionally) appends a per-round shard + `batch_metrics` row in the standard candidate-dataset format; `--finalize` assembles shards into `candidates.csv`+`manifest.json` with `has_route=0`. Reusable by any baseline entrant from any env.

**FragGFN adapter (`validation/generators/fraggfn/`; runs in the `fraggfn` env)**
- `proxy.py` — `AtomMPNNProxy`, the in-loop reward `M`: the Bengio-2021 atom-graph MPNN imported from `gflownet.models.bengio2021flow` (same architecture as RGFN's `glue.proxies.LearnedGlueProxy`), refit from scratch each round; `reward()` maps standardized prediction → positive reward.
- `task.py` — `FragGFNTask` (a gflownet `GFNTask` whose reward is `M`) + `FragGFNTrainer` (a `StandardOnlineTrainer` over `FragMolBuildingEnvContext`, `num_workers=0`, constant temperature β). `build_constant_temperature` sets the fixed β.
- `al_loop.py` — `FragGFNActiveLearningLoop` (mirrors `glue/active_learning/loop.py`) + `LabelStore` (the accumulating dataset `D`, mirrors `glue.datasets.OracleLabeledDataset`). Labels each round's batch by shelling out to the bridge.
- `run_fraggfn_al.py` — entry point: load YAML, build proxy + trainer + loop, run.
- `README.md` — the two-env design, faithfulness table, run + smoke instructions.

**Config**
- `validation/configs/fraggfn_6td3.yaml` — the headline run config; budget matched field-for-field to `configs/glue/active_learning_6td3_gpu.gin`.
- `validation/configs/fraggfn_smoke.yaml` — tiny CPU mock-oracle smoke config.

**Run scaffolding**
- `experiments/active_learning/fraggfn_6td3/submit_fraggfn_6td3.sh` — Balam compute-node SLURM script (loads CUDA-11.8, OpenCL health gate, runs the loop under `fraggfn`; the bridge subprocess re-enters `rgfn`).
- `experiments/active_learning/fraggfn_6td3/README.md` — the run's documentation + RGFN-vs-FragGFN comparison table.

**Datasets**
- `experiments/active_learning/6td3/seed_6td3.csv` — `D_0`, **reused unchanged** from the RGFN run (408 validated docking labels: 160 known glues + 248 decoys, `dvina` differential). Identical seed is what makes the comparison fair.

**Results (git-ignored scratch)**
- `/scratch/markymoo/rgfn_runs/experiments/active_learning/fraggfn_6td3/2026-06-29_17-20-46/` — the completed run (job 69482): `active_learning/{dataset_round_00N.csv, top_k.csv}` + `active_learning/suggestions/{shards/, round_00N_{batch.smi,labels.csv}, batch_metrics.csv, candidates.csv, manifest.json}`.
- `/scratch/markymoo/rgfn_runs/al_fraggfn_6td3-69482.out` — SLURM job log.
- RGFN reference: the matched-oracle RGFN **GPU** run is entry 014 (job 69517, completed); the earlier CPU-oracle RGFN run is entry 011 (job 69445). Entry 014's failed early attempts (incl. job 69481) are documented there.

## Relevant Versions

```
08da97c Working GPU Oracle RGFN, FGFN, RxnFlow, SCENT, AiZynthFinder
cdf3f78 GPU loop + FGFN loop
d1d77f9 Active learning & GPU dock
```

Committed in **`cdf3f78`** ("GPU loop + FGFN loop"): the FragGFN adapter `validation/generators/fraggfn/*`; the bridge `scripts/score_batch.py`; `validation/configs/fraggfn_*.yaml`; `validation/{__init__,generators/__init__}.py`; the submit script + experiment README; the `.gitignore` rule for `external/*/`; and this log. The all-NaN abort guard mirrored into `validation/generators/fraggfn/al_loop.py` (from `glue/active_learning/loop.py`) landed in **`08da97c`** ("Working GPU Oracle …"). `external/gflownet/` is intentionally git-ignored (installed via `external/setup_fraggfn.sh`, not vendored).

## Relevant Resources

**Sources**
- Koziarski et al. 2024, *RGFN: Synthesizable Molecular Generation Using GFlowNets* — `[koziarski2024rgfn]` (the FGFN baseline + synthesizability framing).
- Bengio et al. 2021, *Flow Network based Generative Models…* — `[bengio2021gflownet]` (the active-learning loop, Alg. 1).

**Packages**
- Recursion `gflownet` @ commit `da999404e997a302a81773eb183b1da6ec2a4449` (`FragMolBuildingEnvContext`, `StandardOnlineTrainer`, `bengio2021flow.MPNNet`) — installed in the `fraggfn` env by `external/setup_fraggfn.sh`; used by `validation/generators/fraggfn/{proxy,task}.py`.
- `glue` (this repo, `rgfn` env) — `glue.oracles.Docking6TD3GpuOracle`, `glue.datasets.candidates`, `glue.metrics.dataset_metrics` — used by `scripts/score_batch.py`.

## Method

1. **Install** (login node, conda + network): `bash external/setup_fraggfn.sh` → created the `fraggfn` env, installed torch 2.1.2+cu118, PyG extension wheels, gflownet 0.1.14, pinned numpy 1.26.4. Verified `mol2graph` node-feature width = 71, `FRAGMENTS` = 72.
2. **Bridge unit check** (`rgfn` env, after `module load cuda/11.8.0`): scored two mock batches with `scripts/score_batch.py --oracle mock --suggestions-dir … --step {1,2}`, then `--finalize`. `glue.datasets.candidates.validate_candidate_dataset` returned no issues; `has_route=0`.
3. **End-to-end CPU dry run** (`fraggfn` env): `python validation/generators/fraggfn/run_fraggfn_al.py --cfg validation/configs/fraggfn_smoke.yaml --seed-csv experiments/active_learning/6td3/seed_6td3.csv --device cpu --root-dir /tmp/fraggfn_smoke` → 2 rounds × 5 GFN steps × 6-molecule batch, mock oracle via the bridge. Loop completed: fit `M` → train fragment-GFN → sample → bridge-score → accumulate → finalize → Top-K.
4. **Headline run (DONE, Balam job 69482, balam009, 27m33s):** `sbatch experiments/active_learning/fraggfn_6td3/submit_fraggfn_6td3.sh` (3 rounds × 300 GFN steps × 32-molecule batch, real `Docking6TD3GpuOracle`). Completed cleanly before the cluster outage at ~18:40; all 3 rounds docked 32/32. (Per round: ~7.5 min fragment-GFN training + ~80 s GPU docking.)
5. **Analysis (Trillium login, shared FS):** recomputed per-round + pooled metrics from the run's `candidates.csv` with the *RGFN* analysis script `experiments/active_learning/6td3/analyze_suggestions.py` (unchanged — it reads the standard format), confirming cross-entrant reuse.

## Results

**Per-round (FragGFN, job 69482, real GPU oracle).** `dvina` = Vina(Tier2)−Tier1, lower = better glue.

| Round | \|D\| (+added) | oracle mean | median | best | frac ≤ −2.0 | modes/scaffolds | int. diversity | novelty | MW | QED |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 440 (+32) | −2.12 | −2.05 | −4.08 | 56% | 32/32 | 0.87 | 1.00 | 704 | 0.16 |
| 2 | 472 (+32) | −1.84 | −1.98 | −4.86 | 50% | 32/32 | 0.87 | 1.00 | 748 | 0.15 |
| 3 | 503 (+32) | −1.55 | −2.23 | −4.02 | 56% | 32/32 | 0.87 | 1.00 | 752 | 0.15 |

**Pooled over all 96 suggested molecules:** best **−4.86**, median **−2.06**, mean −1.84, **52/96 (54%) ≤ −2.0**, 20/96 (21%) ≤ −3.0. Top-16 deliverable written; rank-1 = `CC[C@@H](CO)Nc1nc(NC(=O)CCc2ccccc2)c2ncn(C)c2n1` at −4.92.

**Head-to-head vs RGFN — matched GPU oracle.** Equal seed `D_0`, budget (3×32, top-16),
β=8, proxy architecture, **and now the identical `Docking6TD3GpuOracle`** — the only thing
that differs is the generator's action space. RGFN = entry 014's completed GPU run (job
69517); numbers are generated-only (96 suggested molecules each). The earlier CPU-oracle RGFN
run (entry 011) is shown as a third column for continuity with the pre-GPU baseline.

| | FragGFN (this run, 69482) | RGFN-GPU (entry 014, 69517) | RGFN-CPU (entry 011, pre-GPU) |
|---|---|---|---|
| best `dvina` | **−4.86** | −3.90 | −4.71 |
| median `dvina` | −2.06 | −2.14 | ≈ −2.4 |
| frac ≤ −2.0 | 52/96 (54%) | 42/96 (44%) | 61/96 (64%) |
| diversity / scaffolds | 0.87 / all distinct | 0.85 / all distinct | diverse |
| MW / QED drift | MW ≈720, QED ≈0.15 | MW ≈665, QED ≈0.12 | same drift |
| synthesizable route | **none** (`has_route=0`) | yes, by construction | yes, by construction |
| oracle used | GPU `Docking6TD3GpuOracle` (AUROC ≈0.88) | **same** GPU oracle (≈0.88) | CPU `Docking6TD3Oracle` (≈0.95) |

**On the comparison.** The FragGFN-vs-RGFN row is now clean: both ran the identical GPU oracle
on a healthy node, so any difference is attributable to the action space, not the oracle. On
that matched oracle FragGFN and RGFN are in the same league on glue score (FragGFN edges the
single best hit and the ≤ −2.0 fraction; RGFN has the slightly better median), and both show
the same MW/QED size drift — so the differentiator RGFN must lean on is synthesizability, not
glue quality. Historical note: getting the matched RGFN-GPU run took several tries — entry
014's *first* attempts (e.g. job 69481) died at docking (round 1: "Docking attempt #1–10 failed
on GPU 0", `|D|=408 (+0)`), which is what motivated the all-NaN abort guard now mirrored into
this FragGFN loop; the root cause (GPU-memory contention, fixed with `empty_cache()`) and the
successful run (69517) are written up in entry 014.

**Implementation / validation checks (all passed):**

| Check | Result |
|---|---|
| `fraggfn` env build + gflownet import | OK (torch 2.1.2+cu118, numpy 1.26.4, gflownet 0.1.14) |
| End-to-end loop on real GPU oracle | OK (3 rounds, 96/96 docked, completed in 27m33s) |
| Standard candidate dataset conformance | `validate_candidate_dataset` → no issues; `generator=fraggfn`, `has_routes=False` |
| Cross-env oracle bridge | OK (`conda run -n rgfn ... score_batch.py`, 32/32 per round) |
| RGFN `analyze_suggestions.py` on FragGFN output | OK unchanged → cross-entrant reuse confirmed |
| Budget parity with `active_learning_6td3_gpu.gin` | exact across all 9 params (rounds/batch/top-k/β/steps/oracle args) |
