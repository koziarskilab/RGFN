# sEH — fixed-reward pipeline + matched four-way benchmark (RGFN / FragGFN / RxnFlow / SCENT)
**Date:** 2026-07-02, ~10am

## Question

Can we reproduce the RGFN paper's *single-shot* training setup — train the generator once
against a fixed reward, with no active-learning loop — and use it to compare RGFN against
three baseline generators on the same well-known sEH target and the same reward?

## Context & Summary

Everything in this project so far has trained RGFN (and the baselines) through an
**active-learning loop**: a cheap learned model gives the reward, an expensive docking
oracle is queried on small batches to keep that model honest, repeat (entries 011–017).
But the **RGFN paper itself uses no such loop** — it trains the generator *once* against a
**fixed reward generator** and reads off the result. To reproduce the paper and to have a
clean second pipeline for comparisons, we built a **fixed-reward pipeline** that sits
alongside the active-learning one.

We then use it for a **matched four-way comparison** on the classic sEH benchmark target:
RGFN, FragGFN (a non-synthesizable fragment generator), RxnFlow, and SCENT are each trained
**once** against the *same* frozen pretrained sEH scorer (the Bengio-2021 MPNN — the paper's
"sEH proxy"), with the same reward temperature and training-step budget. The only thing
that differs between the four is the **generator itself**. Every generator's output is then
scored for real-world makeability with AiZynthFinder (entry 018), producing one comparison
table. As a second RGFN-only reproduction we also run RGFN against **live GPU docking** as
the reward (the paper's "docking directly in the loop").

RGFN's fixed-reward run finished first and behaves exactly as the paper would predict: it
produces 1000 distinct, synthesizable-by-construction molecules that the sEH scorer rates
highly (mean score 7.0 out of a ~8.3 max), while showing the same drift toward large,
less drug-like molecules (mean weight ~535, QED ~0.26) we see in our own loop runs.

## Answer

The fixed-reward pipeline works and reproduces the paper's single-shot behaviour: RGFN
trained once against the frozen sEH scorer generates many diverse, high-scoring,
route-carrying molecules. The AiZynthFinder scoring path runs cleanly on this pipeline's
output (a 5-molecule spot check found routes for 4). The full four-way comparison is in
progress — RGFN is done; the three baseline runs are re-queued after a first attempt
surfaced (and we fixed) two operational issues. This entry establishes the pipeline and
the RGFN result; the comparison table lands once the baselines finish.

## Relevance to our Publication

NeurIPS reviewers will expect us to (a) reproduce the RGFN paper's own setup before trusting
our extensions, and (b) show RGFN's headline advantage — *synthesizability* — against
baselines on a common, well-known target under identical conditions. This fixed-reward
pipeline is the vehicle for both: it mirrors the paper exactly, and the matched four-way on
sEH isolates the generator's action space as the only variable, with AiZynthFinder giving an
independent makeability verdict on every entrant.

## Next Experiments

**Refining for publication**
- Complete the four-way table once RxnFlow and SCENT finish (~14–17 h runs): AiZynth
  success rate + SA + drug-likeness per generator.
- Resolve the RGFN-on-docking reproduction (test #2): the run died ~2 h in when the upstream
  GPU-docking backend returned no poses for a batch (a transient QuickVina2-GPU failure the
  upstream code does not guard). Retry on a fresh node, or gate on the docking health-check.
- Report the sEH scorer's own headline "number of diverse high-reward modes" for RGFN (the
  paper's figure of merit; metrics are already logged to wandb by the run).

**Next steps in project**
- Fold the fixed-reward pipeline into the standard entrant set so any future target can be
  run in either mode (loop vs single-shot) with the same candidate-dataset output.

# Re-creation

## Relevant Files

Root: `./` (repo root). Run outputs on `/scratch/markymoo/rgfn_runs/` (Balam; `$HOME` is
read-only on compute nodes).

**Scripts / pipeline (ours):**
- `./glue/fixed_reward/pipeline.py` — `FixedRewardPipeline`: the single-shot pipeline
  (train once → sample → score with the reward generator → write candidate dataset). The
  no-oracle, no-refit counterpart of `glue/active_learning/loop.py`.
- `./scripts/fixed_reward.py` — entry point (mirrors `scripts/active_learning.py`); routes
  outputs to `experiments/fixed_reward/<variant>/<timestamp>/`.
- `./scripts/ingest_candidates.py` — converts a baseline's pre-scored `pairs.csv`
  (+ optional `routes.jsonl`) into the standard candidate dataset under the `rgfn` env
  (baselines can't import `glue`). The fixed-reward analogue of `score_batch.py --finalize`.
- `./validation/generators/{fraggfn,rxnflow}/fixed_reward.py` — `SEHFrozenReward`: the frozen
  pretrained sEH MPNN (`gflownet.models.bengio2021flow`) as a drop-in reward, matching
  RGFN's `SehMoleculeProxy` (same checkpoint + featurization).
- `./validation/generators/{fraggfn,rxnflow}/run_*_fixed.py` — single-shot runners; reuse
  each AL adapter's `_train_steps`/`_sample_query_batch`. FragGFN samples in **chunks**
  (bounded memory).
- `./validation/generators/scent/{fixed_reward.py,run_scent_fixed.py}` —
  `ScentFixedRewardRun` (mirrors `FixedRewardPipeline`) using SCENT's own frozen
  `@SehMoleculeProxy`.
- `./validation/harness/aggregate_synthesizability.py` — assembles per-generator
  `synthesizability_summary.json` into the comparison table.

**Configs (ours):**
- `./configs/glue/fixed_reward_seh_proxy.gin` (β=8, `SehMoleculeProxy`) and
  `./configs/glue/fixed_reward_seh_docking.gin` (β=4, 400 iters, `DockingMoleculeProxy`) —
  each `include`s the upstream paper config unchanged.
- `./validation/configs/{fraggfn,rxnflow}_seh_fixed.yaml`, `./validation/configs/scent_seh_fixed.gin`.

**Submit scripts:** `./experiments/fixed_reward/{seh_proxy,seh_docking,fraggfn_seh,rxnflow_seh,scent_seh}/submit_*.sh`
and `./experiments/fixed_reward/submit_aizynth_fourway.sh` (AiZynth over all four + aggregate).

**Reward generator (upstream, pristine):**
- `./rgfn/gfns/reaction_gfn/proxies/seh_proxy.py::SehMoleculeProxy` — pretrained Bengio-2021
  MPNN, checkpoint `.../proxies/cache/bengio2021flow_proxy.pkl.gz`. The shared sEH reward.

**Results:**
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/seh_proxy/2026-07-01_13-54-58/fixed_reward/candidates/`
  — RGFN fixed-reward candidate dataset (1000 molecules; job 69563).
- `./validation/results/seh_fixed_reward/comparison.{csv,md}` — the four-way table (pending).

**Job Logs:** `/scratch/markymoo/rgfn_runs/fr_*-<jobid>.{out,err}`.

## Relevant Versions

Branch `GPU-Dock`. The fixed-reward pipeline + baseline runners + configs + submit scripts
are **not yet committed**. [TODO — add commit hash after pushing.] Files to commit: the
`glue/fixed_reward/` package, `scripts/fixed_reward.py`, `scripts/ingest_candidates.py`,
`configs/glue/fixed_reward_seh_*.gin`, `validation/generators/{fraggfn,rxnflow,scent}/*fixed*`,
`validation/generators/rgfn/README.md`, `validation/configs/*_seh_fixed.*`,
`validation/harness/aggregate_synthesizability.py`, `validation/results/`, and
`experiments/fixed_reward/`.

## Relevant Resources

**Sources**
- RGFN (`[koziarski2024rgfn]`, arXiv:2406.08506) §4.1 — the fixed-reward sEH setup (frozen
  proxy vs docking-in-the-loop).
- GFlowNet multi-round setup (`[bengio2021gflownet]`) — the active-learning loop this
  pipeline is the counterpart to.
- AiZynthFinder (`[genheden2020aizynth]`) — synthesizability metric (entry 018).

**Packages**
- `gflownet.models.bengio2021flow` (FragGFN/RxnFlow envs) — the frozen sEH MPNN reward,
  used by `validation/generators/{fraggfn,rxnflow}/fixed_reward.py`.
- QuickVina2-GPU / `DockingMoleculeProxy` (rgfn env) — docking reward for test #2.
- AiZynthFinder 4.x (`aizynth` env) — `validation/harness/synthesizability.py`.

## Method

1. Built `FixedRewardPipeline` + entry point + two RGFN configs; registered in
   `glue/registry.py`. Validated locally (login-node smoke: train few iters → sample →
   conformant candidate dataset).
2. Added single-shot runners + configs + `SEHFrozenReward` for FragGFN, RxnFlow, SCENT (all
   train once against the same frozen sEH proxy, β=8, 5000 steps, seed 42); candidate emission
   via `scripts/ingest_candidates.py`. Smoke-tested all three (3 steps → conformant datasets).
3. Submitted 5 Balam jobs (2026-07-01): `fr_seh_proxy` (69563), `fr_fraggfn_seh` (69564),
   `fr_rxnflow_seh` (69565), `fr_scent_seh` (69566), `fr_seh_docking` (69567).
4. First-attempt outcomes: 69563 **COMPLETED** (14 h 38 m); 69564 **OOM at sampling**
   (sampled all 4000 trajectories in one call); 69565 / 69566 **TIMEOUT** at 8 h (5000
   steps need ~14–17 h at ~10–12 s/step); 69567 **FAILED** ~2 h in (upstream GPU-docking
   returned `None` for a batch). Fixed: FragGFN now samples in chunks; RxnFlow/SCENT walltime
   raised to 24 h. Re-submitted baselines (69606/69607/69608).
5. Validated the AiZynthFinder path on the RGFN candidate dataset (top-5 spot check).
6. **Pending:** after the baselines finish, `submit_aizynth_fourway.sh` scores each top-100
   with AiZynth and writes `validation/results/seh_fixed_reward/comparison.{csv,md}`.

## Results

**RGFN fixed-reward on sEH proxy — job 69563, COMPLETED (14 h 38 m, 5002 iterations):**

| metric | value |
|---|---|
| candidates (all unique, all `has_route=1`) | 1000 |
| sEH reward score (higher = better) | mean 7.00, max 8.30, min 2.68 |
| molecular weight | mean 535.4 |
| QED | mean 0.255 |
| AiZynth success (top-5 spot check) | 0.80 (4/5), steps_mean 4.5, SA_mean 3.38 |

**Four-way synthesizability comparison** (AiZynthFinder over top-100; `success` = fraction
with a route to purchasable stock). Committed: `validation/results/{seh,drd2}_fixed_reward/comparison.{csv,md}`.

*sEH* (all four reach similar sEH reward ~6.4–7.5, so synthesizability is the differentiator):

| generator | synth-by-construction | AiZynth success | SA | MW | QED |
|---|---|---|---|---|---|
| SCENT | yes | **0.71** | 3.13 | 514 | 0.29 |
| RGFN | yes | 0.52 | 3.22 | 535 | 0.26 |
| RxnFlow | yes | 0.33 | 4.65 | 718 | 0.20 |
| FragGFN | no | **0.02** | 4.32 | 690 | 0.17 |

*DRD2*:

| generator | reward mean | synth | AiZynth success | SA | MW | QED |
|---|---|---|---|---|---|---|
| RGFN | 0.895 | yes | **0.47** | 3.10 | 486 | 0.34 |
| RxnFlow | 0.793 | yes | 0.30 | 3.33 | 534 | 0.29 |
| SCENT | 0.918 | yes | 0.25 | 3.29 | 509 | 0.35 |
| FragGFN | **0.035** | no | **0.00** | 4.67 | 680 | 0.19 |

Headline: the non-synthesizable foil (FragGFN) is ~0% independently synthesizable on both
targets (2% / 0%) vs 25–71% for the reaction/building-block generators — the paper's
synthesizability contrast. Honest nuance: the by-construction `has_route=1` generators are
only *partly* reproduced by AiZynth (25–71%), because AiZynth's ZINC stock + USPTO templates
are a different library than each generator's blocks — so it's an independent check, not a
tautology.

**FragGFN–DRD2 anomaly + diagnosis (diverges from the RGFN paper, which reports FragGFN
optimizes DRD2).** Unlike the consistent sEH result, FragGFN's DRD2 reward stayed flat at the
random base rate (mean 0.035, 918/1000 candidates < 0.05 DRD2 activity) — it never discovered
active chemistry, while RGFN/RxnFlow/SCENT (all at the same constant β=48) reached ~0.79–0.92.
Evidence it's an *exploration* failure, not a bug: (a) TB loss converged to ~0.5 (healthy);
(b) the training-DB flat-reward (`fr_0`) trajectory is dead flat across all 10 deciles for
DRD2 but climbs 472→1480 for sEH; (c) RxnFlow used the *identical* `DRD2FrozenReward` scorer
and succeeded, so the reward is wired correctly; (d) `ci_beta` was a constant 48, whereas
gflownet's stock FragGFN task (`seh_frag.py`) samples temperature **uniformly from [0,64]** —
i.e. our fixed-β regime (chosen to match RGFN's single fixed β) removes the temperature-
annealed exploration a fragment generator needs to find a *sparse* target like DRD2. The
building-block/reaction generators tolerate constant β=48 because their block prior starts
near drug-like/active chemistry. **Confirmatory run (job 69690, `fraggfn_drd2_uniform.yaml`):**
FragGFN-DRD2 with uniform β∈[0,64], final batch sampled at the exploitation β — a `temperature`
config block was added to the FragGFN runner (`constant` default, `uniform` for this
diagnostic; sEH/matched runs unchanged). [PENDING result: if reward climbs, the four-way
failure was the fixed-β regime; if it still flatlines, it's a genuine fragment-prior limit.]

**First-attempt job outcomes (2026-07-01):**

| job | generator | result | note |
|---|---|---|---|
| 69563 | RGFN (sEH proxy) | ✅ COMPLETED | 1000 candidates, 14 h 38 m |
| 69564 | FragGFN | ❌ OOM (sampling) | fixed: chunked sampling → re-run 69606 |
| 69565 | RxnFlow | ⏱ TIMEOUT @8h (step ~2300) | fixed: 24 h walltime → re-run 69607 |
| 69566 | SCENT | ⏱ TIMEOUT @8h | fixed: 24 h walltime → re-run 69608 |
| 69567 | RGFN (docking) | ❌ FAILED @2h | upstream QV2-GPU returned no poses; retry pending |

**Extension to the paper's other two surrogate proxies (2026-07-02).** The RGFN paper uses
three proxies: sEH (above), **DRD2**, and **senolytic**.
- **DRD2 — four-way built + submitted.** `DRD2Proxy` is the portable Therapeutics Data
  Commons oracle (`tdc.Oracle("DRD2")`, sklearn+FCFP6). Verified a standalone scorer (cached
  `oracle/drd2_current.pkl` + `GetMorganFingerprint(mol,3,useCounts=True,useFeatures=True)`→2048
  → `predict_proba`) reproduces `tdc.Oracle("DRD2")` **bit-for-bit**, so the baselines match
  RGFN without the heavy PyTDC install (which would break their pinned envs). Configs:
  `configs/glue/fixed_reward_drd2.gin` (RGFN, β=48; needed a `beta` macro bind — upstream
  `rgfn_drd2.gin` sets `Reward.beta` but not `%beta`), `validation/configs/{fraggfn,rxnflow}_drd2_fixed.yaml`,
  `validation/configs/scent_drd2_fixed.gin` (SCENT clone's `@TDCProxy`). Baseline runners are
  now reward-selectable via a `reward.type` config key. Jobs (2026-07-02): 69609 RGFN
  (running), 69610 FragGFN, 69611 RxnFlow, 69612 SCENT. AiZynth table via
  `sbatch --export=ALL,SYSTEM=drd2 experiments/fixed_reward/submit_aizynth_fourway.sh` →
  `validation/results/drd2_fixed_reward/`.
- **Senolytic — blocked.** `SenoProxy` subclasses `GNEpropProxy`, a custom GNN needing
  `external/gneprop/` + `models/seno.ckpt` (via `external/setup_gneprop.sh`), neither present;
  it also can't run in the baseline envs. Not "easy" — deferred until GNEprop is set up (and a
  four-way isn't possible without a portable senolytic scorer; RGFN-only would be possible once
  the checkpoint is obtained).
