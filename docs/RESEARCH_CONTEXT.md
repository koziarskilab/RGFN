# RGFN Molecular Glue Research — Project Context

This file exists so that AI agents working on this project can orient themselves quickly. Read it before writing any experiment log. Update it as the project evolves.

> **Ground your claims in the real work.** The method (what RGFN/GFlowNets are) and the
> biology (why these systems and metrics) come from a short shelf of papers in
> `Logs/references/` — read its `README.md` before asserting how the model or the oracle
> works. Cite by key (e.g. `[koziarski2024rgfn]`); don't reconstruct a citation from memory.

---

## What this project is

We're building an end-to-end computational pipeline for generating novel **molecular glue degraders** — small molecules that force two proteins to interact, triggering degradation of a disease-causing protein. The generative model is **RGFN** (`[koziarski2024rgfn]`), a **GFlowNet** (`[bengio2021gflownet]`). A GFlowNet is *not* a reward-maximizing RL agent: it learns to **sample** candidates with probability **proportional to their reward**, so it returns many diverse high-scoring molecules instead of collapsing onto a single optimum. RGFN specifically builds each molecule by composing **chemical reactions over a building-block library**, so everything it proposes is synthesizable by construction. RGFN needs a *reward signal* to know whether a molecule is a good glue candidate — we're building and validating that signal (the **oracle**).

The oracle uses **molecular docking**: simulate how well a candidate molecule sits in the complex and — crucially — whether its *arm* helps recruit the second protein the glue must bring in. A good glue should score well not just because its warhead is anchored in a pocket, but because its arm earns an **extra** binding bonus once the recruited partner is present. We capture that bonus as a two-tier differential on the **same pose**: Tier 1 scores the molecule against only the protein its warhead anchors into; Tier 2 adds the recruited partner; the **differential (Tier 2 − Tier 1)** isolates the **arm's contribution once the recruited partner is present** — our working proxy for glue-specific cooperativity. (This is the project's own methodological contribution, not something inherited from a paper.)

What the differential *does* capture is the ligand arm's direct contacts with the recruited partner in the docked pose. What it *cannot* capture is protein–protein interface stabilization that doesn't run through the ligand — which is the most likely explanation for the CRBN ceiling below, where neosubstrate recognition is heavily PPI-driven.

Which protein is *anchored into* versus *recruited* **flips between our two systems** — they are mirror images, which is exactly why the same differential works for both:

- **CRBN / 5HXB** (`[matyskiela2016cc885]`): warhead (glutarimide) anchors in the **E3** (CRBN); the arm recruits the **neosubstrate** (GSPT1). Differential = GSPT1 bonus.
- **CDK12–DDB1 / 6TD3** (`[slabicki2020cr8]`): warhead (purine) anchors in the **target kinase** (CDK12); the arm recruits the **E3 adaptor** (DDB1). Differential = DDB1 bonus.

We call this metric the **neosubstrate differential** for historical reasons — it was first defined on CRBN, where the recruited partner *is* the neosubstrate. For 6TD3 the recruited partner is the E3, so there the same metric is the DDB1 bonus. (For in-silico glue-design context and why 6TD3 is a sound testbed, see `[bengeoffrey2025molde]`.)

### How the model learns: the active-learning loop

RGFN never reads the expensive scoring function directly. We train it the way `[bengio2021gflownet]` runs its multi-round molecule experiments (§4.3; **Algorithm 1** in A.5): a fast **proxy** `M` is the in-loop reward, and the expensive **oracle** `O` is queried only on small batches to keep that proxy honest.

**Before the loop (initialization).** Build a seed dataset `D_0` of `(x, O(x))` pairs labeled by the *true* oracle, then **warm-start** the proxy `M` by fitting it on `D_0`. Skip this and round 1 trains RGFN against an untrained proxy (noise). Also fix `N` (rounds), `β` (inverse temperature), and `K` (how many top molecules you want out at the end). One more constraint: a GFlowNet needs a **positive** reward, so `O(x)` — and hence `M(x)` — must be scaled/shifted to be `> 0` before use (docking is negative-is-better, so this matters).

**One round** (higher β peaks the sampling distribution harder toward high reward):

1. Fit the proxy `M` on everything labeled so far, `D_{i-1}`.
2. Train RGFN to sample proportional to `M(x)^β`.
3. Sample a query batch `B = {x_1,…,x_b}`, each `x_j ~ π_θ`, from the trained policy.
4. Score `B` with the expensive oracle `O`, giving `D̂_i = {(x_j, O(x_j)) : x_j ∈ B}`.
5. Accumulate the **full history**: `D_i = D̂_i ∪ D_{i-1}` (the proxy is refit on everything, not just the latest batch).
6. Repeat for `N` rounds; the deliverable is the top-`K` molecules in `D_N`.

Verbatim algorithm (`[bengio2021gflownet]`, Alg. 1, A.5; their generic oracle `O` = our docking/MD scorer, their proxy `M` = our fast learned reward):

```
Input:  D_0 = {(x_i, y_i)}   (y_i = true oracle reward);  K;  N;  β
Result: TopK(D_N)
Init:   proxy M;  policy π_θ;  oracle O;  i ← 1
while i ≤ N:
    fit M on dataset D_{i-1}
    train π_θ with unnormalized target reward  r(x) = M(x)^β
    sample query batch  B = {x_1,…,x_b},  x_j ~ π_θ
    evaluate B with O:  D̂_i = {(x_1, O(x_1)), …, (x_b, O(x_b))}
    update dataset  D_i = D̂_i ∪ D_{i-1}
    i ← i + 1
return TopK(D_N)
```

**The one gotcha to respect:** oracle scores enter the loop *only* by retraining `M`, never as a direct RGFN reward — the chain is `oracle → improves proxy → proxy scores everything → RGFN learns`, not `oracle → RGFN`.

Why this needs a *diverse* sampler — i.e. why a GFlowNet rather than reward-maximizing RL — is the paper's own argument: the proxy is only trained on what the generator proposes, so a mode-collapsed generator gives the proxy no signal outside the modes it already found (`[bengio2021gflownet]`, A.5). RGFN's diversity keeps the proxy's training distribution broad, which is what makes the loop work at all.

**Mapping the paper's vocabulary onto our systems.** The expensive oracle `O` is our **docking** score today (and **MD stability** later — see open experiments); the **proxy** `M` is a fast learned model fit on those scores. In their molecule experiment they instantiate exactly this — an MPNN proxy trained to predict AutoDock scores, refit on ~200 freshly docked molecules per round (`[bengio2021gflownet]`, A.5.2). Where docking is cheap enough to call in-loop, it can stand in for `M` directly in early single-system runs; the proxy/refresh machinery is what keeps the full run tractable once the per-molecule score (MD, co-folding) is far too slow to call millions of times.

The core argument of the paper: RGFN + a properly designed docking oracle, trained through this active-learning loop, can generate novel, drug-like, **synthesizable** molecular glue candidates. Each experiment is a piece of evidence for this argument.

---

## Paper target

Realistic tiering (revised down from an earlier "NeurIPS primary / Nature stretch" framing to match what an in-silico-only result can actually clear; see the project's venue-strategy notes):

- **Fast / first** — an ML workshop: **NeurIPS AI4Science**, **ICLR MLDD**, or **GEM**. Low-risk, quick, citable; most are non-archival, so a fuller journal version can follow. Best way to leave the lab with a result locked in.
- **Primary journal** — **Digital Discovery (RSC)**: the best fit for a rigorous in-silico methods+application paper without wet validation. **Journal of Cheminformatics** is a strong alternative if we lean on the reusable-method/codebase framing.
- **Stretch** — **JCIM**: achievable *with* ≥2 validated systems, clean/transparent data splits, recovery of known glues, and strong baselines including a non-synthesizable generator.
- **Out of scope without wet-lab** — NeurIPS/ICLR **main track** and **Nature-family**. The competitive ML tracks and the Nature tier effectively require a synthesized, tested compound.

**What reviewers will ask:**
- Does the oracle generalize beyond one system? (Need 2+ validated systems.)
- Does RGFN actually generate better molecules than random sampling / a non-synthesizable baseline? (cf. `[bengio2021gflownet]` Fig. 7 — GFlowNet beats random acquisition in the multi-round setting; we need the analogous plot.)
- Can you show the neosubstrate differential specifically matters (ablation vs. Tier 2 absolute score only)?
- How do you position against existing **conditioned** glue generators, and is "synthesizable" doing real work here? (Our answer: reaction-grounded synthesis routes, not merely valid/drug-like molecules.)
- (Top-tier only) Did you synthesize and test any generated molecules?

---

## Current project status

A list of objectives for our project. Tiers: **MVP** (minimum publishable result), **Target** (the full glue story), **Stretch** (upside). Check items `[X]` as they land; blocked items carry a ⚠️.


### Objective 0 — Oracle validation *(Foundation)*
- [X] Validate the docking oracle on **6TD3**: 78-pp separation between known glues and warhead-matched decoys on the DDB1 neosubstrate differential — the validated testbed for RGFN. (exp `002`)
- [X] Confirm the **neosubstrate differential** (Tier2 − Tier1, same pose) as the discrimination metric. **The specific signal is the *Vina* Tier2 − Tier1 differential.** Did six-way ablation and validated again with MW matching. (exp `006`, `007`)
- [X] Stand up the batched **gnina** pipeline on Balam (16 workers, 4× A100; ~14 min / 400 molecules). (exp `004`)
- [ ] ⚠️ **CRBN / 5HXB docking oracle** — blocked at a ceiling (−3 pp; docking can't see CRBN's PPI-driven recognition). Not usable as-is; revisit via MD (Objective 3) rather than sinking more time into box-docking tweaks. (exp `001`, `003`)
- [ ] Validate the oracle on **≥1 additional system** from MolGlueDB (generalization evidence the journals require).
### Objective 1 — First end-to-end RGFN run *(MVP)*
- [X] Wire the glue oracle into RGFN's reward interface (`glue/` package → `[koziarski2024rgfn]` proxy/reward API). (exp `009`)
- [X] Build the seed dataset `D_0` and **warm-start the proxy** (`[bengio2021gflownet]` Alg. 1 init) — 408 labels, proxy val MSE 0.243. (exp `009`)
- [X] Run RGFN through the **active-learning loop** on the validated 6TD3 oracle. Multi-round loop (3 rounds) completed on a compute node with the query-batch docking intact (exp `011`, job 69445): generated molecules dock as real glues (median ΔT2−T1 ≈ −2.4, 61/96 ≤ −2.0). Inner-loop learning was first confirmed at 1 round (exp `009`); the login-node CPU cap that killed earlier query-batch docking is gone on compute nodes. A **fast GPU-docking oracle for sEH** also exists (`DockingSEHOracle`, exp `010`, 0.60 s/mol over 300 molecules) for driving the loop on the standard RGFN benchmark target via `experiments/active_learning/seh/submit_al_seh.sh`.
- [ ] **Instrument oracle-call counting from the very first run** (cannot be reconstructed later).
- [ ] Produce the **top-k-vs-oracle-calls** curve with a **random-acquisition baseline**.
### Objective 2 — Reward design & ablations *(Target)*
- [ ] Assemble the multi-objective reward: differential + QED (+ synthesizability comes free from RGFN).
- [ ] Ablation: **differential vs. Tier-2-absolute-only** — does the cooperativity term actually matter?
- [ ] Ablation: contribution of each reward component / building-block set.
- [x] Ablation: **pose selection — CNN vs. Vina** — **DONE** (exp `008`, Balam job 69443): CNN pose selection wins, AUROC **0.946 → 0.795** under Vina selection (Cohen's d 2.38 → 0.95; rules agree on only 41%/18% of known/decoy poses). Go/no-go on gnina resolved **in favour of keeping gnina** — the CNN has a load-bearing role as pose picker, so the 6TD3 oracle still needs a GPU-docking-**plus-CNN-rescore** path (not pure QuickVina2-GPU) to run multi-round.
### Objective 3 — Beat the CRBN ceiling with MD *(Target)*
- [ ] Test **MD stability** as oracle `O` for CRBN-type systems where docking plateaus.
- [ ] If MD discriminates, fold it in as the expensive `O` with a learned proxy `M` (multi-fidelity loop).
### Objective 4 — Generalization & baselines *(Target / journal bar)*
- [ ] Drive **≥2 systems** end-to-end through generation (not just oracle validation).
- [ ] Add a **non-synthesizable baseline generator** (e.g., graph GFlowNet / REINVENT) on the *same* oracle and budget. **FragGFN** (Recursion `gflownet` fragment env) implemented + locally validated through the AL loop on the same proxy/oracle/seed/budget as RGFN (exp `015`); GPU docking run pending.
- [ ] **≥3 seeds** per headline result, reported with error bars.
### Objective 5 — Evaluation suite *(Target)*
- [ ] **Synthesizability**: SA distribution + by-construction route advantage vs. the non-synthesizable baseline (the headline differentiator).
- [ ] **Diversity** (internal + scaffold) under a fixed oracle budget.
- [ ] **Recovery** of known glues / glutarimide chemistry (retrospective enrichment).
- [ ] **Anti-gaming**: in-loop-proxy vs. high-fidelity-oracle correlation on top-k; medchem sanity (PAINS, MW, logP).
- [ ] **High-fidelity validation** of top-k (co-folding / Boltz-2 / short MD).
### Objective 6 — Positioning & release *(always-on)*
- [ ] Write the **novelty paragraph** vs. the conditioned JT-VAE glue generator (reaction-grounded synthesizability + goal-directed sampling, not a conditioned VAE).
- [ ] Keep **code/data release-ready**: pinned env, fixed seeds, dataset provenance; archive every run's full output.


---

## Experiment log index

Chronological record; the objectives above cite these by number. Full entries in `../Logs/`.

| # | Date | Title | Verdict |
|---|------|-------|---------|
| [001](../Logs/001_5hxb-crbn-anchored-docking.md) | 2026-06-11 → 06-18 | 5HXB / CRBN — warhead-anchored docking oracle + decoy control | Works as a docker; **discrimination ceiling** — GSPT1 bonus is a geometric artifact, not a glue signal |
| [002](../Logs/002_6td3-cr8-validation-and-discrimination.md) | 2026-06-18 | 6TD3 / CDK12-DDB1 — oracle validation + discrimination run | **Decisive discrimination** — 85.6% vs 7.3% on DDB1 differential (+78 pts); validated oracle |
| [003](../Logs/003_crbn-vs-6td3-cross-system.md) | 2026-06-18 | 5HXB vs 6TD3 — head-to-head neosubstrate differential comparison | **6TD3 discriminates (+78 pts); 5HXB does not (−3 pts)** — failure is structural, not methodological |
| [004](../Logs/004_compute-benchmark.md) | 2026-06-18 | Compute benchmark — login node vs Balam debug_full_node (4× A100) | **CRBN 21× / 6TD3 6.6× faster; batching is the dominant lever** — conformer embedding is the new bottleneck |
| [005](../Logs/005_tier2-vina-roc-pr-curves.md) | 2026-06-23 | Tier 2 Vina — ROC and PR curves for 6TD3 and 5HXB | **6TD3 AUC=0.890 / AP=0.872; CRBN AUC=0.627** — absolute Tier 2 is a strong 6TD3 oracle; CRBN ceiling confirmed structural |
| [006](../Logs/006_6td3-violin-distributions.md) | 2026-06-25 | 6TD3 — which metric discriminates best (Tier 1 vs Tier 2 vs Δ, Vina vs CNN) | **Vina ΔT2−T1 wins (AUROC 0.946)**; Vina Tier 1 worst (0.691) — the signal lives in what DDB1 adds |
| [007](../Logs/007_6td3-molecular-weight-control.md) | 2026-06-25 | 6TD3 — controlling glue-vs-decoy discrimination for molecular weight | **Differential survives MW-matching (0.95→0.87); absolute scores collapse (Vina Tier 1 → 0.38)** — it isn't reading ligand size |
| [008](../Logs/008_pose-selection-cnn-vs-vina.md) | 2026-06-26 | 6TD3 — does pose selection (CNN vs Vina) change discrimination? | **CNN pose selection wins: AUROC 0.946 → 0.795 under Vina** (Cohen's d 2.38 → 0.95; rules agree 41%/18% known/decoy). gnina **stays** — CNN pose-picking is load-bearing; oracle needs GPU-dock + CNN-rescore, not pure Vina |
| [009](../Logs/009_first-rgfn-inner-loop-learning.md) | 2026-06-25 | 6TD3 / RGFN — first end-to-end active-learning run (1 round, inner-loop learning) | **Generator learns** — loss ↓24×, mean predicted ΔT2−T1 0→−0.62, stays diverse; but best plateaus short of real glues (no scaffold <−2.0) and QED drifts down 0.40→0.17. Query-batch docking killed by login-node CPU cap — no true labels yet |
| [010](../Logs/010_seh-gpu-docking-oracle.md) | 2026-06-26 | sEH — fast GPU-docking oracle (`DockingSEHOracle`) for the active-learning loop | **Works and is fast** — QuickVina2-GPU sEH dock at **0.60 s/mol** over 300 molecules (~6× faster than a tiny batch), reproduces aspirin −6.3; thin wrapper over RGFN's own docking proxy, wired into the loop with the MPNN proxy + a 250-molecule RGFN-generated seed. Unblocks the multi-round loop the CPU-bound glue oracle couldn't finish |
| [011](../Logs/011_6td3-multiround-al-true-labels.md) | 2026-06-26 → 06-27 | 6TD3 / RGFN — first multi-round AL loop with true oracle labels (3 rounds, compute node) | **Generated molecules dock as real glues** — median ΔT2−T1 ≈ −2.4 (in known-glue range), 61/96 ≤ −2.0, best −4.7; docking step intact (no CPU cap). Uses the **CPU two-tier gnina oracle** — the pre-GPU baseline that entries 012→013→014 build on. Cost: **GFN training 65% + docking 35%** of 5 h wall-clock (the 35% docking cost is what motivated the GPU-docking arc), proxy fit + sampling <0.5%. Fixed an inverted `top_k` deliverable. Size drift persists |
| [012](../Logs/012_pre-gpu-docking-substep-timing.md) | 2026-06-28 | 6TD3 / RGFN — pre-GPU-exploration: instrumented docking sub-step timing baseline (job 69451) | **Tier-2 search IS the docking cost: `tier2_dock` = 99.3%** of the docking phase (58 s/mol), embed 0.6% / rescore 0.2% / pose-pick ~0%; sub-steps sum to `oracle_score` exactly. Scales with molecule size (53→70 s/mol). ⇒ Tier-2 search ≈ 33% of whole loop = GPU-dock speed-up ceiling. Embed bottleneck worry (exp 004) refuted |
| [013](../Logs/013_gpu-pose-gen-vs-gnina-search.md) | 2026-06-29 | 6TD3 — GPU pose generation (QuickVina2-GPU) vs gnina CPU search, discrimination held fixed | **GPU search keeps most of the signal at ~50× speed** — swapping only the Tier-2 search to GPU (QV2-GPU `num_modes=9`, gnina CNN pose-pick + two-tier gnina `--score_only`) gives AUROC **0.88** (0.875 compute / 0.892 login) vs the **0.948** same-data gnina-search control (≈006's 0.946): −0.06 AUROC for **1.1 vs 58 s/mol**. New reusable `Docking6TD3GpuOracle`. Also: QV2 OpenCL is wedged on balam008 (`clCreateContext -5`) but fine on balam009 → submit script excludes it + gates on an OpenCL health check |
| [014](../Logs/014_gpu-oracle-al-loop-suggestion-logging.md) | 2026-07-01 | 6TD3 / RGFN — GPU-oracle active-learning loop + standard candidate-dataset logging (job 69517, **complete**) | **First full GPU-oracle AL loop; found+fixed a GPU-memory-contention bug.** 3 rounds, \|D\| 408→483, molecules dock in known-glue range (per-round median −2.0…−2.4, best −4.92, matching entry 011), maximally diverse (32/32 modes, novelty 1.0), heavy/non-drug-like (MW ~650, QED ~0.12 — size drift persists). Docking collapsed to ~1 min/round (<2% of loop) vs entry-012's ~31 min (33%). **Root cause of 3 failed attempts: torch's caching allocator starves the QV2-GPU docking subprocess (40 GB→1.1 GB free → 0 poses); `torch.cuda.empty_cache()` before docking fixes it.** New standard candidate format (`glue/datasets/candidates.py`) + `SuggestionLog` (96 candidates + routes + per-round metrics); + pre-flight dock gate, all-NaN abort guard, manifest provenance |
| [015](../Logs/015_fraggfn-6td3-baseline.md) | 2026-06-29 | 6TD3 / FragGFN — non-synthesizable fragment-GFlowNet baseline through the active-learning loop | **Ran end-to-end on the real GPU oracle (job 69482)** — non-synthesizable baseline docks as real glues, competitive with RGFN: best −4.86, median −2.06, 54% ≤ −2.0, fully diverse, same MW/QED drift; `has_route=0`. ⇒ synthesizability (not glue score) is RGFN's differentiator. Now a **clean same-oracle head-to-head**: the matched RGFN-GPU run has since completed (entry 014, job 69517), so FragGFN-GPU is compared against RGFN on the identical `Docking6TD3GpuOracle` (best −3.90 / median −2.14 / 44%), not the earlier CPU-oracle RGFN run (entry 011) |
| [016](../Logs/016_rxnflow-6td3-baseline.md) | 2026-06-30 | 6TD3 / RxnFlow — synthesizable reaction-template+block GFlowNet baseline through the active-learning loop | **Ran end-to-end on the real GPU oracle (job 69518, 51 min)** — the *synthesizable* peer to RGFN (vs FragGFN's non-synth foil): RxnFlow (`[seo2024rxnflow]`, ICLR 2025) through the same loop/oracle/seed/budget/β/proxy, isolating the generator's action space. 96 candidates, best −4.22, median −1.19, 31% ≤ −2.0, all synthesizable (`has_route=96/96`, 2-step). **Key finding:** the only entrant that is both synthesizable *and* drug-like (MW 489 / QED 0.36 vs RGFN's 665 / 0.12) — RGFN's synthesizability is real but doesn't buy drug-likeness; a block-constrained generator gets it free. Clean matched-oracle **three-way** with RGFN (69517) + FragGFN (69482). Adapter mirrors FragGFN (own `rxnflow` py3.12 env), bridge now **route-aware** (`--routes` → `has_route=1` + `routes.jsonl`) |
| [017](../Logs/017_scent-6td3-baseline.md) | 2026-06-30 | 6TD3 / SCENT — cost-aware template GFlowNet baseline through the active-learning loop | **Ran end-to-end on the real GPU oracle (job 69513, 2 h 03 m)** — SCENT's *cost-aware* generator (`[gainski2025scent]`, an RGFN fork w/ Recursive Cost Guidance + Exploitation Penalty + Dynamic Library) matches RGFN/FragGFN on glue quality while staying fully synthesizable: 96 candidates, median dvina **−2.12**, best **−5.81** (strongest of any entrant), 51% ≤ −2.0, **`has_route=96/96`** (tree routes via the dynamic library). Notable: cost-awareness appears to **counteract the size drift** — MW falls 694→579, QED rises 0.11→0.21 across rounds (opposite of RGFN `011`/FragGFN `015`). Same loop/oracle/seed/budget/β/proxy; own `scent` env (namespace clash), gin-driven, route-aware bridge. Bring-up needed 3 fixes (setuptools<81, explicit `Trainer` import, sign-safe `train_metrics`). RGFN-GPU anchor (entry `014`, job 69517) now complete → clean matched-oracle **four-way** (RGFN/FragGFN/RxnFlow/SCENT); SCENT + RxnFlow are the two drug-like entrants |
| [019](../Logs/019_fixed-reward-pipeline-seh-fourway.md) | 2026-07-02 | Fixed-reward (single-shot) pipeline mirroring the RGFN paper + matched four-way sEH benchmark (RGFN/FragGFN/RxnFlow/SCENT) | **Pipeline built + validated; RGFN done, four-way in flight.** New `glue/fixed_reward/FixedRewardPipeline` (no AL loop, no oracle — train once vs a fixed reward generator) + baseline single-shot runners, all trained vs the *same* frozen Bengio-2021 sEH proxy (β=8, 5000 steps). RGFN fixed-reward COMPLETE (job 69563, 14h38m): 1000 unique synthesizable candidates, sEH reward mean 7.0/max 8.3, MW 535/QED 0.26 (size drift persists); AiZynth path validated (top-5 = 4/5 routes). FragGFN OOM→chunked-sampling fix, RxnFlow/SCENT timeout→24h walltime, all re-queued (69606/07/08); RGFN-on-docking (test #2) hit an upstream QV2-GPU no-pose failure. Four-way AiZynth table pending baselines |
| [020](../Logs/020_stdlib-fourway-seh-fixed-reward.md) | 2026-07-02 → 07-06 | sEH — same-library four-way fixed-reward (RGFN/SCENT/RxnFlow share `glue_standard_v1`; FragGFN native foil) | **COMPLETE — the library was not entry `019`'s confound.** New `glue.chemistry.ChemLibrary` + `GlueReactionDataFactory` + canonical `data/libraries/glue_standard_v1` (SCENT SMALL: 418 frags/112 rxns, priced) + retroactive cost evaluator (`validation/harness/cost.py`). All 4 ran on the shared lib (RGFN 69616 / RxnFlow 69614 / SCENT 69608 / FragGFN 69606 foil); AiZynth+cost table job 69867. On identical chemistry the three synthesizable generators separate, not converge: **SCENT** best sEH (7.63) + cheapest routes (2.9, w/ 74% dynamic-lib fallback caveat) + best AiZynth (0.71); **RxnFlow** most drug-like (QED 0.62, MW 413) but lowest reward (6.22) and lowest AiZynth (0.29 despite has_route=1); **RGFN** scores well, drifts largest/priciest (34). FragGFN foil: reward parity but AiZynth 0.02. ⇒ has_route=1 is not uniform; RGFN ~2×/iter slower on the larger library (metric audit: metrics only ~11% of per-iter, size-drift dominates). Results: `validation/results/seh_fixed_reward_stdlib/`. |
| [018](../Logs/018_synthesizability-metric-aizynthfinder.md) | 2026-06-30 | Synthesizability metric — AiZynthFinder + SA over the standard candidate dataset (Objective 5) | **Works end-to-end on Balam.** Post-hoc evaluator (`validation/harness/synthesizability.py`) that runs the same way on every entrant: AiZynthFinder route-found "success rate" (`[genheden2020aizynth]`; the metric RGFN/RxnFlow/SCENT all report) + RDKit SA distribution over the standard candidate-dataset format, with a by-construction-vs-independent route cross-check. AiZynth 4.4.1 in its own `aizynth` env (`external/setup_aizynthfinder.sh`); real API matched the code with no changes. Test set of 8 known glues + 3 anchors: **0.82 success (9/11)**, SA mean 2.50, 0 errors; ibuprofen's miss reproduces AiZynth's known conservatism. Ready to run across all generators' outputs → committed comparison table (next) |
| [021](../Logs/021_fixed-reward-docking-matrix.md) | 2026-07-03 → 07-06 | Fixed-reward 4-reward × 4-generator matrix — per-step GPU docking for EVERY generator (ClpP + 6TD3 differential) on the SMALL library | **LAUNCHED (400 iters) + AiZynth partial; 2 RGFN cells pending.** Benchmark (69691): cross-env per-step docking amortizes to ~10-13% at 400 iters → per-step `score_batch.py` bridge, no server. New reusable `glue.proxies.OracleRewardProxy` (any `GlueOracle`→GFN reward, `clip(-raw/norm)` + per-step `empty_cache`; the guard job 69567 lacked), `DockingClpPOracle`, baseline `DockingBridgeReward`/`DockingBridgeProxy`. Both mechanisms validated (smoke 69692/69693), then full matrix fired `submit_matrix.sh` (jobs 69695-69704). **8/10 emitted candidates; baseline per-step docking worked end-to-end.** Docking quality (dvina/Vina): SCENT strongest 6TD3 glue (med −2.51, best −5.81) > FragGFN (−2.17) > RxnFlow (drug-like, MW 379/QED .45 on ClpP); ClpP all ≥196/200 ≤−2. **AiZynth (69870/71/72):** FragGFN foil = 0.00 route-success everywhere; synth gens 0.2–0.61 (RxnFlow ClpP 0.61) — independent-route ≠ by-construction. **PENDING:** RGFN-6TD3 (timed out iter 260/400 → resumed 69868) + RGFN-DRD2 (69703) still training → blank rows in `validation/results/{6td3,drd2}_fixed_reward/comparison.md`; re-run aizynth per Logs/021 "▶ PICK UP HERE" to fill. Not committed (branch GPU-Dock). |
| [022](../Logs/022_hub-diversification-pareto-pipeline.md) | 2026-07-03 | sEH / RGFN — hub-based late-stage-diversification analysis pipeline (`glue/analysis/`) + diversity-vs-concurrency/-cost Pareto fronts | **Infrastructure built + validated end-to-end on a real trained GFN.** New `glue/analysis/` package turns a trained reaction-GFN into a parallel-batch synthesis plan: sample → **hub graph** (pre-terminal `ReactionStateA` intermediates; flow = visit count, since Trajectory Balance `[malkin2022trajectorybalance]` trains no `F(s)`) → pluggable hub selectors (`highest_flow`/`highest_tb_flow`/`most_modes`/`highest_expected_reward`/`highest_child_reward`) → observed **or** enumerative child expansion → molecule selectors (`top_k_reward`/`top_k_reward_diverse`/`scaffold_diverse_k`/`random_k`) → `BatchPlan`→standard `CandidateDataset` → diversity/concurrency/cost/reward metrics → trials sweep → **`pareto_front` + auto-plots**. Validated on the sEH-proxy stdlib checkpoint (entry 020): observed sweep 48 rows, enumerative 72 rows; single hubs enumerate to 29–106 products; `top_k_reward_diverse` yields all-distinct products; **hub-batching saves 33–50%** vs independent synthesis, and the highest-concurrency config (m8·k30) is Pareto-optimal on both diversity (0.816) and cost/mol (7.25). Also adds a **second, balance-based flow estimator** — `F(h)=R(x)P_B(h|x)/P_F(x|h)` read off the TB condition (`glue/analysis/tb_flow.py`) — plus a sampling-vs-TB `flow_agreement` training-quality diagnostic. Methods/tooling (Objective 5/6); core pipeline committed in `9dd7b37` (balance-flow follow-up pending commit) |
| [023](../Logs/023_oracle-call-curve-pilot.md) | 2026-07-06 | 6TD3 / RGFN — single-seed pilot of the oracle-call curve (RGFN vs random acquisition) | **Instrumentation works on the real oracle; RGFN already beats random.** New oracle-call counting + best-so-far trace (`glue/active_learning/acquisition_trace.py`) + a **random-acquisition baseline arm** (uniform-policy over the same reaction blocks, proxy/training off) for the Objective-1 top-k-vs-oracle-calls curve (`[bengio2021gflownet]` Fig. 7). Both arms COMPLETED on the 6TD3 GPU oracle (jobs 69945/69946): at 96 oracle calls RGFN improves the top-16 mean differential **−0.68** vs random's **−0.14** (≈5× faster/call), from a shared −3.209 seed baseline; both best stuck at −4.915 (D_0's CR8 purine — budget too small to beat it). Docking 1.6% of the loop (reproduces entry 014). Found+fixed a run-dir collision (concurrent same-config jobs shared a timestamped dir → now keyed by arm+seed). Curve validated end-to-end; ready for the 3-seed × 10-round campaign (`launch_curve_6td3.sh`). Not committed (branch GPU-Dock). |
| [024](../Logs/024_scent-backward-policy-recovery.md) | 2026-07-07 | SCENT — persist the trained backward policy (P_B) so it survives checkpointing / is recoverable for flow analysis | **Fixed + verified bit-exact.** Acid test found SCENT's `last_gfn.pt` saves only forward policy + logZ (131 keys, 0 backward) — the cost-guided P_B's two learned MLPs (`SimpleCostModel`/`BinaryDecomposableModel`, ~263k params each) live in a plain Python list on `JointlyGuidedBackwardPolicy`, which `state_dict()` skips, so reload lands on random weights (differs on 146/195 multi-parent steps). Fix: `guidance_io.py` sidecar (`guidance_models.pt`) wired into our `fixed_reward.py`/`al_loop.py` adapters (SCENT clone untouched). Login-safe round-trip verifier: with the sidecar, reloaded P_B is **bit-exact** (max \|ΔlogP_B\|=0.0 over 690 steps); without it, 146/195 C-phase steps wrong. All 4 SCENT fixed-reward runs re-launched with the patched adapter (70066/67/68/69); **seh + drd2 COMPLETE and confirmed end-to-end** (`check_trained_sidecar.py`: sidecar clean, trained P_B differs from a random reload by ~8 nats max / carries real info, reload deterministic), 6td3 + clpp (docking) still training. All 5 pre-fix SCENT checkpoints flagged on disk (`BACKWARD_POLICY_NOT_SAVED.txt`). Not committed (branch GPU-Dock). |

---

## The systems

| Name | PDB | E3 | Neosubstrate | Glue | Warhead | Oracle status |
|---|---|---|---|---|---|---|
| 5HXB | 5HXB | CRBN | GSPT1 | CC-885 | glutarimide | ⚠️ Ceiling — even with warhead-anchored docking, the neosubstrate differential doesn't discriminate real glues from random ones |
| 6TD3 | 6TD3 | DDB1 (adaptor) | cyclin K (CDK12-bound) | CR8 | purine (binds CDK12, *not* the E3) | ✅ Validated oracle |

> Note on 6TD3: the protein *recruited* in the differential is the **E3 adaptor DDB1**, not the degraded neosubstrate (cyclin K). This is the mirror-image flip described above and the reason "neosubstrate differential" is a slight misnomer for this system.

---

## Key terminology

- **RGFN**: Our generative model — a GFlowNet that builds molecules from reactions and **samples proportional to reward** (not reward-maximizing). Needs an oracle/proxy reward to learn what "good" looks like. See `[koziarski2024rgfn]`, `[bengio2021gflownet]`.
- **Oracle (`O`)**: The expensive, trusted scoring function. Currently docking on the ternary complex; MD stability is a candidate complementary/replacement oracle. In the active-learning loop it is queried only on the per-round query batch.
- **Proxy (`M`)**: The fast, learned in-loop reward, fit on oracle-labeled data and refit each round. RGFN trains against `M(x)^β`, never against `O` directly. See `[bengio2021gflownet]` §4.3 / Alg. 1.
- **Active-learning loop / multi-round**: The training procedure — fit proxy, train RGFN against it, sample a batch, label the batch with the oracle, refit the proxy, repeat for `N` rounds. `[bengio2021gflownet]` Alg. 1 (A.5).
- **β (inverse temperature)**: Controls how peaked the target reward is. Higher β concentrates sampling on high-reward modes.
- **Molecular glue**: Small molecule that induces proximity of two proteins (E3 ligase + neosubstrate), leading to ubiquitination and degradation of the neosubstrate.
- **E3 ligase**: The degradation machinery. We study CRBN (part of the CRL4 complex) and DDB1 (adaptor for CDK12).
- **Neosubstrate**: The protein being degraded. GSPT1 (CRBN system); cyclin K, presented by CDK12 (6TD3 system).
- **Warhead**: The part of the glue that anchors into a fixed pocket. That pocket is on the **E3** for CRBN (glutarimide → CRBN tri-Trp cage) but on the **target kinase** for 6TD3 (purine → CDK12 ATP pocket) — *not* the E3. This flip is the mirror-image point above.
- **Neosubstrate differential**: Tier2 − Tier1 score for the same docked pose. Isolates the arm's contribution to recruiting the second protein (the neosubstrate GSPT1 for CRBN; the E3 adaptor DDB1 for 6TD3). Our primary discrimination metric and the project's novel contribution. The validated signal uses the **Vina** score (lower = better binding, so the differential is lower-is-better; entries 006/007); gnina's CNN score is used only to pick the docked pose, not as the differential.
- **Tier 1 / Tier 2**: Tier 1 = the **warhead-anchoring protein only**. Tier 2 = that protein **plus the recruited partner**. Same pose scored against both. (CRBN: Tier 1 = CRBN, Tier 2 = CRBN+GSPT1. 6TD3: Tier 1 = CDK12, Tier 2 = CDK12+DDB1.)
- **Decoys**: Realistic fake glues — correct warhead, random drug-like arm. If decoys score like known glues, the oracle only reads warhead binding and is useless for RGFN.
- **gnina**: Our docking engine (v1.3.2, CNN-rescored). Launched from `/scratch/markymoo/gnina/run_gnina.sh`.
- **Balam**: SciNet GPU cluster (4× A100 per debug_full_node, 64 cores, 1 h max). Outputs go to `$SCRATCH` (`/scratch/markymoo/`), not `$HOME`.

---

## Where things live

- **Experiment logs**: `Logs/` (indexed above).
- **Key publications** (method + biology): `Logs/references/` — orientation sheet + PDFs; ground method/biology claims here.
- **Repo layout, code, datasets, and result/scratch locations**: see `docs/ARCHITECTURE.md` (the single source for where things live in the repo) — kept there so locations aren't duplicated across docs.
