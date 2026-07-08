# 6TD3 / RGFN — first end-to-end active-learning run: does the generator actually learn?

**Date:** 2026-06-25, ~2pm

## Question

When we train our molecule generator against the fast learned glue-scoring model, does it actually shift toward producing molecules that model rates as better glues?

## Context & Summary

**Context** — Everything so far (entries 001–008) has been about *validating the oracle*: proving our two-tier docking score can tell real molecular glues from fakes on the 6TD3 system (entry 002 found a 78-point separation; entries 006/007 pinned down the exact winning signal and showed it isn't just reading molecular size). But validating a scoring function is not the same as generating molecules. This is the first run where the actual generative model — RGFN — is wired to that signal and trained. It is the opening step of the project's central deliverable (Objective 1): an end-to-end run of the active-learning loop on the validated 6TD3 oracle.

**Summary** — We ran one round of the active-learning loop. First we fit the fast in-loop scorer (the "proxy") on our 408 labeled seed molecules, then trained RGFN for 150 steps to generate molecules the proxy rates highly, watching whether the generated molecules' predicted glue scores improved over training. This was deliberately a small run — on a shared machine while the cluster's compute nodes are down — sized to the smallest budget where genuine learning should still be visible. It is a *plumbing-and-learning* check, not a finished result: a single round means there is no across-rounds comparison yet, and the expensive docking step that would score the generated molecules did not complete (see Answer).

## Answer

Yes — the generator demonstrably learns. Over 150 training steps the model's training loss fell ~24-fold and the average predicted glue score of the molecules it generates moved from roughly neutral into clearly favorable territory, all while it kept exploring tens of thousands of distinct molecules rather than collapsing onto one. So the full chain — seed labels → fit proxy → train RGFN against it — works end-to-end on Balam. Two honest caveats temper the win: the *best* molecules the generator found plateaued well short of the scores our known real glues achieve (and never crossed even the loosest "good glue" threshold), and their drug-likeness drifted *down* as training progressed — an early hint of the score-gaming we will need to guard against. Separately, the run did not produce any *true* oracle scores on the generated molecules: the docking step that scores the sampled batch was killed by a CPU-time limit on the shared login node, so confirming whether the proxy's optimism survives real docking is still open.

## Relevance to our Publication

This is the first evidence for the paper's core claim — that RGFN, trained through the active-learning loop on our validated oracle, *generates* (not just scores) glue candidates. Digital Discovery / J. Cheminformatics reviewers will want the headline "top-k vs. oracle-calls, with a random-acquisition baseline" curve (cf. `[bengio2021gflownet]` Fig. 7); this run is the prerequisite that proves the loop trains at all before we spend the compute to build that curve. The two caveats map directly onto reviewer concerns we already anticipated: the best-molecule plateau speaks to *recovery of known glue chemistry*, and the falling drug-likeness is exactly the *anti-gaming / medchem-sanity* check listed under Objective 5. Catching the drug-likeness drift on the very first run is useful — it tells us the reward likely needs a QED term before the headline run.

## Next Experiments

**Refining for publication** — Re-run on a Balam compute node (via `sbatch`, no login-node CPU cap) so the query batch actually docks, then compare the generated molecules' *true* oracle scores against the proxy's predictions — the first real anti-gaming datapoint. Add molecular-weight and QED tracking to every run so we can see whether the generator earns its scores or drifts to big, undruglike molecules (the entry-007 concern, now observed as a falling-QED signal). Scale to the multi-round loop (`active_learning_6td3_mini.gin`, then the full `active_learning_6td3.gin`) and produce the top-k-vs-oracle-calls curve with a random-acquisition baseline.

**Next steps in project** — Decide whether to fold a QED (drug-likeness) term into the reward now, given the observed drift. Instrument oracle-call counting end-to-end (Objective 1 requires it and it can't be reconstructed later). Harden the docking oracle so one slow/huge generated molecule or a CPU-time limit can't kill an entire batch (sub-batch the gnina call) before committing to long unattended runs.

# Re-creation

## Relevant Files

Root: `configs/glue/`, `glue/`, `experiments/active_learning_6td3_inner/`

Scripts / entry points:
- `./scripts/active_learning.py` — driver for the outer active-learning loop (`[bengio2021gflownet]` Alg. 1); imports `glue` so gin can resolve our oracle/proxy/dataset/loop, parses the config, builds `ActiveLearningLoop`, runs it.
- `./glue/active_learning/loop.py` — the loop itself: fit proxy `M` → train RGFN against `M` → sample query batch `B` → score `B` with oracle `O` → accumulate. The oracle call is `loop.py:132`.
- `./glue/proxies/` (`LearnedGlueProxy`) — the fast learned in-loop reward (atom-graph MPNN), fit on the seed labels and shared as one singleton between the GFN reward and the loop.
- `./glue/oracles/docking_6td3_oracle.py` — the expensive two-tier gnina oracle `O` (Tier 2 dock → CNN pose pick → Tier 1 rescore → Vina ΔT2−T1).

Config:
- `./configs/glue/active_learning_6td3_inner.gin` — **this run.** One round, 150 inner GFN steps, 16-molecule query batch; sized for the login node while compute is down. Includes the metric fix below.
- `./configs/glue/active_learning_6td3_mini.gin` — downsized-but-multi-round (3 rounds) variant, staged for the next `sbatch` run.

New code (the one fix this run forced):
- `./glue/metrics/safe_num_scaffolds.py` — `SafeNumScaffoldsFound`, a crash-safe subclass of upstream `NumScaffoldsFound`. Upstream reads `state.molecule` on every terminal state, but early-terminated trajectories end in a `ReactionStateEarlyTerminal` (no `molecule`), so it raises `AttributeError` the first time an early-terminated trajectory's proxy value crosses a threshold. Every *other* metric in `rgfn/trainer/metrics/reaction_metrics.py` already guards this with `isinstance(state, ReactionStateTerminal)`; `NumScaffoldsFound` is the lone exception. Per the repo rule (don't edit `rgfn/`), fixed by subclassing in `glue/` and wiring `@SafeNumScaffoldsFound` into the config. **This bug also sits in the unmodified `active_learning_6td3.gin` and would crash the full run.**
- `./glue/registry.py` — imports `glue.metrics` so gin can resolve `@SafeNumScaffoldsFound`.

Datasets:
- `./experiments/active_learning_6td3/seed_6td3.csv` — seed dataset `D_0`: 408 validated docking labels (160 known glues + 248 decoys), `label` = Vina ΔT2−T1 (from entry 002, job 69271). Warm-starts the proxy.

Receptors (gitignored binaries; staged into the oracle's expected path this session):
- `./research/preprocessing/docking_6td3/6TD3_tier2.pdbqt` — CDK12+DDB1 (Tier 2).
- `./research/preprocessing/docking_6td3/6TD3_tier1.pdbqt` — CDK12 only (Tier 1).
- `./research/preprocessing/docking_6td3/crystal_RC8.pdb` — native CR8, used as the autobox reference.

Results:
- `./experiments/active_learning_6td3_inner/2026-06-25_14-03-06/` — run dir (logs/config.txt, operative_config.txt, modes/, checkpoints).
- `/scratch/markymoo/wandb/wandb/offline-run-20260625_140307-2asa2sjy/` — offline wandb history (source of the Results numbers below; no `wandb-summary.json` because the run aborted at the docking step).

Job Logs:
- `/scratch/markymoo/al_inner_20260625_140301.log` — full stdout/stderr, including the GFN loss trace and the `SIGXCPU` traceback.

## Relevant Versions

```
ca6253d [DOCS] Merge redundant README into RESEARCH_CONTEXT.md
57efb27 [DOCS] Update research context w/ objectives
13cf8b6 [CODE] MW Matched Decoys and corrected active learning oracle
```

The files this run introduced/changed were committed in `492b2d1` ("[CODE] Active Learning & prepare cnn re-pose experiment"): `glue/metrics/` (new package), `glue/registry.py` (import line), `configs/glue/active_learning_6td3_inner.gin`, `configs/glue/active_learning_6td3_mini.gin`. The receptor `.pdbqt`/`.pdb` under `research/preprocessing/docking_6td3/` are intentionally gitignored.

## Relevant Resources

**Sources**
- Active-learning loop / multi-round protocol and the GFlowNet-vs-random argument: Bengio et al., *GFlowNet Foundations* — `[bengio2021gflownet]`, Alg. 1 / A.5.2 (their MPNN-proxy-on-AutoDock molecule experiment is the template for this run).
- RGFN generative model: Koziarski et al. — `[koziarski2024rgfn]`.
- 6TD3 / CR8 system and the seed labels: entry 002 (Balam job 69271); CR8 structure Słabicki et al., *Nature* 2020 (doi:10.1038/s41586-020-2133-z), `[slabicki2020cr8]`.

**Packages**
- torch 2.3.0+cu118, dgl 2.2.1+cu118, torch-geometric 2.5.3 — RGFN policy/proxy. dgl's Graphbolt requires `module load cuda/11.8.0` (CUDA-11.8 runtime libs) or the import fails.
- gnina v1.3.2 (`/scratch/markymoo/gnina/run_gnina.sh`) — the docking oracle.
- RDKit 2023.09.5 — embedding + QED. gin-config — configuration. wandb 0.26.1 (offline) — metric logging.
- Run inside the `rgfn` conda env on Balam (`balam-login01`, one A100-PCIE-40GB).

## Method

1. **Stage receptors** — copied the gitignored `6TD3_tier{1,2}.pdbqt` and `crystal_RC8.pdb` into `research/preprocessing/docking_6td3/` (the path `Docking6TD3Oracle` expects; they previously existed only under the old untracked `pre-processing/`).
2. **Fix the metric crash** — added `glue/metrics/safe_num_scaffolds.py::SafeNumScaffoldsFound`, registered it via `glue/registry.py`, and wired `@SafeNumScaffoldsFound` into the config with the lower-is-better orientation (`proxy_higher_better=False`, thresholds `[-2.0,-2.5,-3.0,-3.5]`).
3. **Run one round** — `module load cuda/11.8.0`, then
   `python scripts/active_learning.py --cfg configs/glue/active_learning_6td3_inner.gin --seed 42`
   (WANDB offline). The loop fit `M` on the 408 seed labels, trained RGFN 150 steps against `M`, sampled a 16-molecule query batch, and called the oracle on it.
4. **Recover metrics** — parsed the offline wandb binary history (`run-*.wandb`, leveldb-framed protobuf `Record`s; metrics stored under `nested_key`) to extract the per-step trajectories below.

## Results

One round; proxy `M` = `LearnedGlueProxy` (MPNN) fit on n=408 seed labels (160 known + 248 decoy), `β=8`. Inner GFN: 150 steps, 100 forward + 20 replay trajectories/step. Hardware: one A100 on `balam-login01`. Per-step costs measured this session: inner GFN ~10–12 s/step; oracle docking ~12 s/molecule (both tiers).

**Proxy warm-start.** Fitting `M` on the 408 seed labels: best validation MSE **0.243** (RMSE ≈ 0.49 against a label standard deviation of 0.90) — the proxy carries real predictive signal, not noise.

**The generator learns (offline wandb history, 151 logged steps).** Lower predicted differential = better glue.

| metric | step 0 | step 149 | best over run | reading |
|---|---|---|---|---|
| train/loss | 390.1 | 16.6 | 8.2 | converged (~24× drop) |
| train/logZ | 37.5 | 31.6 | — | partition-function estimate stabilized |
| train/reward_mean | 2442 | 12 670 | 17 890 | mean reward of generated batch ↑ ~5× |
| **train/proxy_mean** | **+0.052** | **−0.616** | −0.748 | mean predicted ΔT2−T1 of generated molecules moved neutral → favorable |
| train/min_proxy | −1.364 | −1.385 | −1.422 | **best single molecule essentially flat** |
| train/num_unique_molecules | 100 | 14 940 | — | broad exploration, no mode collapse |
| train/fraction_early_terminate | 0 | 0.008 | ≤0.025 | early termination negligible |
| train/qed | 0.403 | 0.167 | min 0.121 | **drug-likeness fell ~2.4×** over training |
| train/num_scaffolds_{−2.0,−2.5,−3.0,−3.5} | 0 | 0 | 0 | **no generated molecule predicted below −2.0** |

Three takeaways. (1) **Learning is real and the loop works end-to-end**: loss converges, mean predicted glue score of generated molecules improves from ~0 to −0.62, and the model stays diverse (≈15k unique molecules). (2) **But the generator never reaches known-glue territory**: the best generated molecule plateaus at a predicted ΔT2−T1 of ≈ −1.4 and *no* generated molecule crosses even the −2.0 threshold, whereas real known glues in the seed reach −2.2 to −2.6 (Vina ΔT2−T1; entry 002). So in 150 steps RGFN shifts the *bulk* of its distribution toward better scores without discovering anything the proxy rates as strongly as a real glue. (3) **Drug-likeness drifts down** (QED 0.40 → 0.17) as the proxy score improves — a flag for the Objective-5 anti-gaming checks, suggesting the reward may need a QED term.

**The true-oracle labels are still missing.** After sampling 16 unique candidates, the oracle docking call (`run_gnina.sh -r 6TD3_tier2.pdbqt -l batch.sdf … --exhaustiveness 16 --cpu 8`) was killed by `SIGXCPU` (signal 24, CPU-time limit) — the login node enforces `ulimit -t = 3600 s` per process, and docking 16 freshly generated molecules at exhaustiveness 16 across 8 threads exceeded it. This is a shared-login-node guardrail, not a code or method failure: a Balam compute node under SLURM is bounded by job walltime, not this rlimit. Consequently `D_0` was not extended and no generated molecule has a *true* docking score yet — that is the first item for the compute-node re-run.

---

## Amendment (2026-06-25, ~4pm) — is the falling QED a problem? What the source papers actually do

The QED drop above (0.40 → 0.15) prompted the question: do the publications we build on address drug-likeness, and how? Read the actual PDFs (`Logs/references/pdfs/`). Verdict: **our low/falling QED is expected, documented behavior — not a bug — and the fix is one our own plan already names.**

**Both foundational papers deliberately leave drug-likeness *out* of the reward; QED is an evaluation metric only.**

- `[bengio2021gflownet]` (sEH task, §4.2): *"The reward is computed with a pretrained proxy model that predicts the binding energy of a molecule to a particular protein target (soluble epoxide hydrolase, sEH)."* They explicitly punt: *"for realistic drug design, we would need to consider many more quantities such as drug-likeness (Bickerton et al., 2012), toxicity, or synthesizability. Our goal here is not [to] solve this problem."* They tried QED/logP *as* rewards and found them *"very easy to maximize"* (so they used docking as the hard benchmark), and note they *"experimented with combining different scores multiplicatively (e.g. multiplying docking score by a renormalized QED and synthesizability), with some success"* — left to future work.
- `[koziarski2024rgfn]` (the paper we fork): reward is `exp(β·score(x))`, a **single** oracle, no QED/SA term (their sEH β = 8 — the same β our config uses). QED appears only in **Table 1** as an eval metric, framed explicitly *"to gauge the size of generated compounds."*

**The calibration that matters: RGFN's *own* published QED is low.** Table 1, sEH task, top-500 modes: **RGFN QED = 0.29 ± 0.10** (MW 495), vs FGFN 0.39, casVAE 0.52, SyntheMol 0.57, GraphGA 0.21. So the reaction-grounded methods (RGFN, FGFN) sit *low* on QED by construction. **Our batch-mean 0.40 → 0.15 is in the same family as RGFN's headline result — not an anomaly of our setup.**

**They name the mechanism, and it is exactly the entry-007 concern.** `[koziarski2024rgfn]` Limitations (§5): *"docking scores correlate strongly with molecular weight (MW) … molecule size was constrained only by the number of reaction steps, encouraging RGFN to generate large molecules within the building block limit. This can be somewhat rectified by augmenting reward with a drug-likeness or ligand efficiency term."* So the prescribed remedy — augment (RGFN) or multiply (GFlowNet) the oracle with a QED / ligand-efficiency term — **is precisely Objective 2's "multi-objective reward: differential + QED."** This run is the first empirical motivation for building it, and GFlowNet's "QED is easy to maximize" finding suggests adding it won't destabilize learning (the docking differential stays the hard part).

(The glue-design reference `[bengeoffrey2025molde]` does not address drug-likeness/QED/Lipinski/ADMET at all.) Confirmatory test still worth running: measure MW on a sampled batch from the saved checkpoint to verify MW is the property driving the QED drop, as the RGFN authors' MW argument predicts.
