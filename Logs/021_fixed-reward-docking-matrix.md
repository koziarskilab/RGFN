# Fixed-reward docking matrix — per-step GPU docking for every generator (ClpP + 6TD3)
**Date:** 2026-07-03

## Question

Can we run a single coherent **fixed-reward** (single-shot) benchmark of **four reward
generators** — sEH proxy, DRD2 proxy, **ClpP docking**, **6TD3 differential** — across
**four generators** (RGFN, RxnFlow, SCENT on the shared SMALL library; FragGFN on its own
fragments)? The two docking rewards are the new question: RGFN can dock in-process, but the
baselines run in separate conda envs and reach docking only across the env boundary. Is
**per-step docking** (docking called every training step, not per AL round) feasible for a
*single* run — with GPU pose-gen + CPU gnina + batching?

## Context & Summary

Prior work made docking-fixed-reward **RGFN-only** (`fixed_reward_seh_docking.gin`) because
per-round cross-env docking in the AL loop looked too slow to call every step. The user
asked to attempt per-step docking for the baselines too, reasoning that a *single* GFN run
(not many rounds) with GPU acceleration + batching might amortize. **Benchmark first, then
build** — and the numbers said go.

**Phase 0 — docking-throughput benchmark (job 69691, 23 min).** Measured warm in-env vs cold
cross-env (`scripts/score_batch.py`) throughput at batch sizes {8,32,100}, 100% dock success:

| reward | warm in-env (batch 100) | cold cross-env fit | cold @ batch-100 | overhead |
|---|---|---|---|---|
| `docking_seh` (single-target) | 1.36 s/mol | 33 s startup + 1.16 s/mol | ~1.48 s/mol | ~9 % |
| `docking_6td3_gpu` (differential) | 0.83 s/mol | 44 s startup + 0.51 s/mol | ~0.95 s/mol | ~13 % |

The subprocess startup amortizes over a ~100-molecule step, so per-step cross-env docking
adds only ~10-13 %. Projection (400 iters × 100 mol): sEH ~16.6 h, 6TD3 ~10.5 h — both
comfortably inside the SLURM walltime. **Verdict: reuse `score_batch.py` per step (no
persistent server needed); cap docking runs at 400 iters.**

**Phase 1 — reusable core (the modular abstraction).**
- `glue/proxies/oracle_reward_proxy.OracleRewardProxy` — generic adapter turning **any**
  `glue.oracles.GlueOracle` into an in-loop GFN reward: `reward = clip(sign·raw/norm, 0, ∞)`
  (upstream `DockingMoleculeProxy` convention), raw score kept as a `raw_score` component,
  and **`torch.cuda.empty_cache()` before each dock** so QV2-GPU is not starved by torch
  during training (the Logs/014 fix, applied per step — the guard the raw `DockingMoleculeProxy`
  lacked, the likely cause of the sEH-docking job-69567 failure). Any future oracle (MD, …)
  becomes a fixed reward for free.
- `glue/oracles/docking_seh_oracle.DockingClpPOracle` — one-line ClpP target (prepared
  `data/targets/ClpP.pdbqt` + box already in the upstream receptor tables); registered in
  `score_batch.py` as `docking_clpp`.
- Baseline cross-env bridge: `DockingBridgeReward` in `validation/generators/{fraggfn,rxnflow}/
  fixed_reward.py` (a `reward.type: docking` branch) and `DockingBridgeProxy` in the SCENT
  clone (`validation/generators/scent/docking_bridge_proxy.py`, a `CachedProxyBase`). Each
  shells a step's batch to `conda run -n rgfn score_batch.py`, `empty_cache` first, parses the
  label (Vina/dvina), converts to the same reward, caches per canonical SMILES. Provenance:
  score = reward **value** (higher-is-better), raw dvina kept as a `raw_score` column.

**Phase 2 — configs (SMALL library everywhere).** RGFN: `fixed_reward_6td3.gin`,
`fixed_reward_clpp.gin` (both `OracleRewardProxy` + `GlueReactionDataFactory` on
`glue_standard_v1`, β=4, 400 iters, lean metrics — **dropping `TanimotoSimilarityModes` so
metrics never re-dock**), `fixed_reward_drd2_stdlib.gin`. Baselines:
`{fraggfn,rxnflow}_{6td3,clpp}_docking_fixed.yaml`, `scent_{6td3,clpp}_fixed.gin` (also drop
`TanimotoSimilarityModes`), `rxnflow_drd2_fixed_stdlib.yaml`.

**Phase 3 — orchestration.** Per-run submit scripts + parameterized baseline docking submits
(`experiments/fixed_reward/baseline_docking/submit_{fraggfn,rxnflow,scent}_docking.sh CONFIG`)
+ master launcher `experiments/fixed_reward/submit_matrix.sh`. Run-dir names align to the
`<gen>_<system>` convention so the already system-parameterized
`submit_aizynth_fourway.sh SYSTEM=6td3|clpp` aggregates with no change.

## Results — both mechanisms validated end-to-end (smoke tests)

- **RGFN in-env** (job 69692, `fixed_reward_6td3_smoke.gin`, 5 iters, 13 min): trained
  (loss 175→90), docked every step with **no torch-vs-QV2 starvation** (freed 39.7/40.4 GB
  before sampling), emitted 15 conformant candidates **with synthesis routes** + `raw_score`
  (dvina) column. ~137 s/iter ⇒ full 400-iter run ≈ 15 h (fits 20 h walltime).
- **Baseline cross-env** (job 69693, FragGFN `docking_6td3_gpu`, 3 steps, 4.7 min): each step
  shelled to `score_batch.py`, docked 64/63/64 SMILES at **100 % success while torch trained**,
  loss decreasing; emitted 15 conformant candidates (`score`=clip(-dvina) higher-is-better,
  `raw_score`=dvina, `has_routes=False` for the native foil).

## Files

- Benchmark: `experiments/fixed_reward/docking_benchmark/{bench_docking_throughput.py,submit_bench.sh}`;
  results `$SCRATCH/rgfn_runs/docking_benchmark/69691/benchmark_results.json`.
- Core: `glue/proxies/oracle_reward_proxy.py`, `glue/oracles/docking_seh_oracle.py`
  (`DockingClpPOracle`), `scripts/score_batch.py` (`docking_clpp`).
- Baseline bridges: `validation/generators/{fraggfn,rxnflow}/fixed_reward.py` +
  `run_*_fixed.py` (docking branch), `validation/generators/scent/docking_bridge_proxy.py`
  + `fixed_reward.py`/`run_scent_fixed.py`.
- Configs: `configs/glue/fixed_reward_{6td3,clpp,drd2_stdlib,6td3_smoke}.gin`;
  `validation/configs/{fraggfn,rxnflow}_{6td3,clpp}_docking_fixed.yaml`,
  `scent_{6td3,clpp}_fixed.gin`, `rxnflow_drd2_fixed_stdlib.yaml`, `fraggfn_6td3_docking_smoke.yaml`.
- Submit: `experiments/fixed_reward/{6td3,clpp,drd2_stdlib,baseline_docking,smoke}/*.sh`,
  `experiments/fixed_reward/submit_matrix.sh`.

## Status & next

**LAUNCHED 2026-07-03 at 400 iters** — `bash experiments/fixed_reward/submit_matrix.sh` fired
10 jobs (**69695** RGFN 6td3, **69696** RGFN clpp, **69697/69698** FragGFN 6td3/clpp,
**69699/69700** RxnFlow 6td3/clpp, **69701/69702** SCENT 6td3/clpp, **69703** RGFN drd2-stdlib,
**69704** RxnFlow drd2-stdlib). Add-only scope: sEH/DRD2 proxy four-ways not re-run (job 69616
sEH-stdlib still running). Both RGFN docking jobs cleared the OpenCL gate + started training
clean on balam004.

A **4000-iter (publication) run with checkpoint + auto-continuation** was scoped but dropped as
not worth the cost right now: RGFN/SCENT resume natively, but FragGFN/RxnFlow would need custom
gflownet checkpoint/resume, and at 4000 steps the baselines' per-step cross-env `score_batch.py`
startup (~40 s) balloons to ~50% of wall-clock (a persistent docking server would be needed to
tame it). Revisit via a persistent server + resume-chains if the 400-iter results justify it.

After runs finish: `sbatch --export=ALL,SYSTEM=6td3 experiments/fixed_reward/submit_aizynth_fourway.sh`
(and `SYSTEM=clpp`) → synthesizability tables (run-dir names align, no change needed). Caveat:
the DRD2-on-SMALL runs use `*_stdlib` run-dir names → need an aizynth `SUBDIR` tweak (shared with
the existing sEH-stdlib four-way).

## Results (2026-07-06) — 8/10 emitted; 1 resumed; 1 proxy still running

Docking raw score = binding (lower better); 6TD3 in the glue range is ≤ −2. **The baseline
per-step cross-env docking worked** — all three baselines docked end-to-end and produced
glue-range molecules, and the AL-era pattern reproduced under fixed-reward: SCENT strongest on
the 6TD3 differential, RxnFlow the drug-like one, RGFN strongest raw binding.

| gen | ClpP Vina best/med (≤−2) | 6TD3 dvina best/med (≤−2) | routes |
|---|---|---|---|
| RGFN | −14.0 / −8.8 (199) | *resumed → 69868* | ✓ |
| FragGFN (foil) | −10.7 / −6.8 (196) | −4.95 / −2.17 (111) | ✗ |
| RxnFlow | −10.0 / −6.8 (198) | −4.41 / −1.44 (50) | ✓ (MW 379/QED .45 on ClpP) |
| SCENT | −12.7 / −9.75 (198) | −5.81 / −2.51 (145) | ✓ |

DRD2 (SMALL): RxnFlow activity median 0.895 (737/825 >0.5); RGFN-DRD2 (69703) still training.

**RGFN-6TD3 timed out at iter 260/400** (20 h; ~277 s/iter — ~2× the smoke estimate, from CPU
oversubscription with 4 docking jobs sharing balam004 + molecule size-drift), before the emit
phase. Resumed from its `last_gfn.pt` checkpoint (job **69868**) via a new
`scripts/fixed_reward.py --run-name/--resume-from` + `experiments/fixed_reward/6td3/submit_resume_6td3.sh`.
**Gotcha:** upstream `Trainer.__init__` does a STRICT `load_state_dict`, but the forward policy's
lazily-populated block-embedding cache (`forward_policy.b_action_embedding_fn._cache`) is absent
from a fresh model → strict load raises "Unexpected key(s)". The resume script strips `*_cache`
keys from the checkpoint (recomputable) before loading — validated (loads from iter 261). Note
for real 4000-iter chaining later: raise the docking walltime and/or use `--exclusive` to avoid
the oversubscription slowdown.

## Synthesizability (AiZynthFinder) — jobs 69870 (6TD3) / 69871 (ClpP) / 69872 (DRD2)

Ran `submit_aizynth_fourway.sh` per system on the completed candidate datasets (RGFN-6TD3 +
RGFN-DRD2 left as **blank placeholder rows** via the new `aggregate_synthesizability.py
--expect`). Tables in `validation/results/{6td3,clpp,drd2}_fixed_reward/comparison.md`.

| system | RGFN | FragGFN (foil) | RxnFlow | SCENT |
|---|---|---|---|---|
| ClpP  | 0.20 | **0.00** | **0.61** | 0.25 |
| 6TD3  | 0.16 | **0.00** | 0.25 | 0.21 |
| DRD2  | *(pending 69703→70034 resume; aizynth 70035)* | **0.00** | 0.24 | 0.28 |

(cells = AiZynth route-found success rate over top-100.) **Headline confirmed by an independent
retrosynthesis tool:** FragGFN (non-synth foil) = 0.00 everywhere; every reaction-based generator
gets real routes. Also: by-construction ≠ independently-verifiable — even self-route=1.0
generators clear AiZynth on only 0.2–0.6 (RxnFlow's small drug-like ClpP mols verify best, 0.61).

---

## ▶ PICK UP HERE — finish the AiZynth scoring (fill the 2 blank RGFN cells)

Two runs were still training when the tables were built; each leaves a blank RGFN row:
- **RGFN-6TD3** — resume job **69868** (`experiments/fixed_reward/6td3/<ts>/`). On finish emits
  `.../fixed_reward/candidates/candidates.csv`.
- **RGFN-DRD2 (SMALL)** — job **69703** (`experiments/fixed_reward/drd2_stdlib/<ts>/`). Same.

When a job shows `COMPLETED` (`sacct -j <id> -o State`) **and** its `candidates/candidates.csv`
exists, re-run that system's AiZynth aggregation — it scores only the newly-present RGFN dataset
(the other three are cached) and the RGFN row fills in automatically:

```bash
sbatch --export=ALL,SYSTEM=6td3 experiments/fixed_reward/submit_aizynth_fourway.sh   # after 69868
sbatch --export=ALL,SYSTEM=drd2 experiments/fixed_reward/submit_aizynth_fourway.sh   # after 69703
```

Then copy the refreshed tables from `$SCRATCH` into the repo (compute nodes can't write `$HOME`):

```bash
for S in 6td3 drd2; do
  cp $SCRATCH/rgfn_runs/results/${S}_fixed_reward/comparison.{csv,md} validation/results/${S}_fixed_reward/
done
```

Notes for whoever picks this up:
- If RGFN-6TD3 (69868) *also* times out before emitting, resume it again:
  `sbatch experiments/fixed_reward/6td3/submit_resume_6td3.sh` (auto-detects the latest checkpoint,
  strips `*_cache`, continues). RGFN-DRD2 is a proxy run (no docking) and should finish in one wall.
- The aizynth `SUBDIR` map already knows the DRD2-on-SMALL run-dir names
  (`drd2_stdlib` / `rxnflow_drd2_stdlib` / `scent_drd2` / `fraggfn_drd2`) — no edit needed.
- **Nothing is committed to git yet** (branch `GPU-Dock`); commit this session's code + results
  once the two blanks are filled.
