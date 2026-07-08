# RxnFlow / 6TD3 — synthesizable baseline head-to-head with RGFN
**Date:** 2026-06-30, ~4pm

## Question

When the generated molecules are *all* guaranteed to be synthesizable, does our RGFN
glue generator still match or beat another synthesis-aware generator (RxnFlow) at
producing good 6TD3 glue candidates, given the same scoring oracle and the same budget?

## Context & Summary

**Context.** Entry `015` ran our first baseline comparison: FragGFN, a generator that
builds molecules by gluing fragments together with **no synthesis route** — the
"non-synthesizable foil." It scored competitively with RGFN on glue quality (best
docking differential −4.86, median −2.06), which sharpens the story: if a
non-synthesizable model scores about as well, then RGFN's real selling point is that
its molecules come with a synthesis recipe. But reviewers will immediately ask the
harder question — how does RGFN compare against a generator that is *also*
synthesizable? RxnFlow (`[seo2024rxnflow]`, ICLR 2025) is exactly that peer: like RGFN
it assembles molecules from real building blocks and reaction templates, so every
molecule it proposes has a route too.

**Summary.** We run RxnFlow through the **same** active-learning loop, the **same**
6TD3 docking oracle, the **same** seed set of 408 labelled molecules, the **same**
budget (3 rounds × 32 molecules/round), the **same** inverse-temperature β=8, and the
**same** learned proxy reward as the RGFN run — so the *only* thing that changes is the
generator's internal machinery. RxnFlow runs in its own software environment and reaches
our docking oracle through the shared scoring bridge (the same cross-environment setup
FragGFN uses), now extended to record each molecule's synthesis route so RxnFlow's
output sits directly alongside RGFN's for comparison.

## Answer

**RxnFlow, the synthesizable peer, generates glue candidates in the same league as our
generators — a little below on raw glue score, but with markedly more drug-like
molecules and a real synthesis route for every one.** Over its 96 generated molecules
(job 69518, GPU docking oracle): best `dvina` **−4.22**, median −1.19, mean −1.41, and
**30/96 (31%) beat the −2.0 "good glue" cutoff**; internal diversity 0.86–0.90 with all
scaffolds distinct. Every candidate is synthesizable (`has_route=1`, 96/96), with a
short 2-step route (mean 1.97 reactions). Crucially its molecules are far more
reasonable than the non-synthesizable foil's: **MW ≈489 vs FragGFN's ≈720, QED ≈0.36 vs
≈0.15** — the synthesis constraint keeps RxnFlow inside a buildable, more drug-like
region of chemical space.

**With the matched-oracle RGFN GPU run now complete (job 69517), the clean three-way
verdict has two parts.** (1) *On glue score, at the same GPU oracle, all three are in the
same league* — RGFN best on central tendency (median/mean −2.14), FragGFN best on the
single top hit (−4.86) and fraction past the cutoff (54%), RxnFlow modestly behind
(median −1.19, 31%); no generator dominates and the spread is within round-to-round
noise. So RGFN does **not** out-dock a non-synthesizable foil — entry 015's conclusion
survives the clean matched oracle. (2) *The real separation is drug-likeness, and it
splits the two synthesizable generators:* RGFN and FragGFN share a bloated, low-QED
failure mode (MW 665–720, QED 0.11–0.15; RGFN passes Lipinski just 2/96), while
**RxnFlow is the only entrant that is both synthesizable and physically reasonable**
(MW 489, QED 0.36). RxnFlow's building-block library confines it to buildable, drug-like
chemistry; RGFN's reaction-DAG action space, though synthesizable by construction, drifts
into the same oversized region FragGFN does. The headline for the paper: **RGFN's
synthesizability is real, but does not by itself buy drug-likeness — a
block-constrained generator gets that for free**, which is a concrete lead for pairing
RGFN's DAG with a property term or tighter block library (Objective 5). **Caveat:**
single-seed, small 3×300-step budget, and RxnFlow's short ≤2-step synthesis depth — the
deltas need ≥3 seeds + longer training before they're load-bearing.

## Relevance to our Publication

This is the **synthesizable-vs-synthesizable** arm of the baseline comparison reviewers
will demand (`docs/RESEARCH_CONTEXT.md`, Objectives 4–5). FragGFN (entry `015`) shows
RGFN beats a non-synthesizable generator on the synthesizability axis; RxnFlow controls
for synthesizability so we can isolate what RGFN's specific action space buys. A clean
result here — RGFN competitive with a state-of-the-art synthesis-aware generator on the
same oracle and budget — is the kind of strong baseline a methods venue (Digital
Discovery / JCIM) expects before believing the headline claim.

## Next Experiments

**Refining for publication.**
- Run ≥3 seeds for RGFN, RxnFlow, and FragGFN so the head-to-head carries error bars
  (all three now have one clean matched-oracle GPU seed — job 69517/69518/69482).
- **Add a property term (QED / MW) to RGFN** — the sharpest lead from this run: RGFN is
  synthesizable but as bloated/low-QED as FragGFN (MW 665, QED 0.12, Lipinski 2/96),
  whereas block-constrained RxnFlow is drug-like for free (MW 489, QED 0.36). Test
  whether a property-shaped reward or a tighter block library closes that gap without
  hurting glue score.
- Compare *route quality* (synthesis length, route diversity), not just glue score —
  the dimension where two synthesizable generators most plausibly differ (RxnFlow's
  routes are short here, ≤2 steps).

**Next steps in project.**
- Fold RxnFlow into the validation harness's top-k-vs-oracle-calls curve (the
  `[bengio2021gflownet]` Fig. 7 analogue) alongside RGFN and a random-acquisition
  baseline.
- Extend the three-way comparison to a second protein system once a second oracle is
  validated (Objective 0 / 4).

# Re-creation

> **STATUS: COMPLETE — both runs landed.** RxnFlow (job 69518, 2026-07-01 00:40, 51 min)
> and the **matched-oracle RGFN GPU run** (job 69517, elapsed 3h14m, exit 0:0) both
> finished; the head-to-head below is now a clean **same-oracle GPU three-way** (RGFN vs
> RxnFlow vs FragGFN-015), no CPU placeholder. RxnFlow: 96/96 scored, **all 96 with
> synthesis routes**, conformant dataset. Adapter/stability fixes are in
> `docs/REFACTOR_LOG.md` (esp. the `hb_edited.txt`→`real.txt` env fix). (RGFN's in-loop
> GPU training is ~4× slower per round than the bridge baselines — 1h/round vs ~15min —
> which is why 69517 took 3h while 69518 took 51m.)
>
> **Balam validation before the run (login A100):**
> - RxnFlow API confirmed: `algo.graph_sampler.sample_inference` + `ctx.read_traj`/
>   `object_to_log_repr` yield SMILES + faithful synthesis routes (8/8 samples, real
>   `FirstBlock`/`UniRxn`/`BiRxn` steps) → `extract_route` rewritten to that real format.
> - **Training stability:** `hb_edited.txt` (71 templates) on the 10k debug library →
>   non-finite TB loss (all-masked action states); reproduced with RxnFlow's own QED
>   task (not our code). Fixed by the default **`real.txt`** (109 templates): QED control
>   0/25 non-finite, our constant-β=8 loop 0/25. Kept RxnFlow's native
>   `subsample=0.02` + `train_random_action_prob=0.1`.
> - Full cross-env mock smoke: 2 rounds → standard dataset, `has_route=1`,
>   `routes.jsonl` 12/12 with real routes; conformant.

## Relevant Files

Root: repository root unless noted.

**Scripts / adapter** (`./validation/generators/rxnflow/`):
- `run_rxnflow_al.py` — entry point: loads the YAML, builds the proxy + RxnFlow trainer
  + loop, runs it (runs in the `rxnflow` conda env).
- `al_loop.py` — `RxnFlowActiveLearningLoop`: the `[bengio2021gflownet]` Alg.1 loop
  (fit proxy → train RxnFlow → sample batch + routes → label via bridge → accumulate →
  Top-K), with the all-NaN oracle-failure abort guard; `extract_route` reconstructs each
  molecule's synthesis route from its trajectory.
- `task.py` — `RxnFlowTask` (reward = our proxy `M`) + `RxnFlowGlueTrainer` (RxnFlow's
  synthesis GFlowNet with our proxy, `num_workers=0`, constant β).
- `proxy.py` — re-exports FragGFN's `AtomMPNNProxy` so the proxy `M` is identical across
  all entrants.

**Shared oracle bridge:**
- `./scripts/score_batch.py` — scores a batch of SMILES with a named glue oracle and
  writes the standard candidate-dataset format; extended here to be **route-aware**
  (`--routes` per-round JSONL → joined into `routes.jsonl` on `--finalize`,
  `has_route=1`). Runs in the `rgfn` env.

**Install / run scaffolding:**
- `./external/setup_rxnflow.sh` — builds the `rxnflow` conda env (py3.12, torch
  2.5.1+cu121), installs RxnFlow + its bundled gflownet, prepares the env directory
  (building blocks + reaction templates).
- `./experiments/active_learning/rxnflow_6td3/submit_rxnflow_6td3.sh` — Balam SLURM
  submit (OpenCL health gate, `--exclude=balam008`, `$SCRATCH` outputs).
- `./experiments/active_learning/rxnflow_6td3/README.md` — run card + the RGFN/RxnFlow
  comparison table.

**Config / seed:**
- `./validation/configs/rxnflow_6td3.yaml` — budget/oracle/proxy matched field-for-field
  to `configs/glue/active_learning_6td3_gpu.gin`; `rxnflow:` block carries the generator
  knobs. `./validation/configs/rxnflow_smoke.yaml` — tiny CPU mock-oracle dry run.
- `./experiments/active_learning/6td3/seed_6td3.csv` — the shared seed `D_0` (408
  validated docking labels), reused identically by RGFN, FragGFN, and RxnFlow.

**Results** (git-ignored, on `$SCRATCH`):
- `/scratch/markymoo/rgfn_runs/experiments/active_learning/rxnflow_6td3/2026-06-30_23-49-14/`
  (SLURM job **69518**, balam004) — per-round `dataset_round_NNN.csv`, `top_k.csv`, and
  `suggestions/` (standard `candidates.csv` (96 rows) + `manifest.json`
  (`has_routes=true`) + `routes.jsonl` (96 routes) + `batch_metrics.csv`). Stdout:
  `/scratch/markymoo/rgfn_runs/al_rxnflow_6td3-69518.out`.

## Relevant Versions

```
08da97c Working GPU Oracle RGFN, FGFN, RxnFlow, SCENT, AiZynthFinder
ded1c0d checkpoint for FGFN, RxnFlow, SCENT, AiZynthFinder, sEH
cdf3f78 GPU loop + FGFN loop
```

The route-aware `scripts/score_batch.py` (the `--routes` → `has_route=1` + `routes.jsonl`
extension) landed first in **`ded1c0d`** ("checkpoint …"). The RxnFlow adapter
(`validation/generators/rxnflow/*`), `external/setup_rxnflow.sh`,
`validation/configs/rxnflow_{6td3,smoke}.yaml`, `experiments/active_learning/rxnflow_6td3/*`,
and the reference/doc updates were finalised in **`08da97c`** ("Working GPU Oracle …"), the
commit that carries the job-69518 run and the matched-oracle RGFN GPU run (job 69517) it is
compared against.

## Relevant Resources

**Sources.**
- `[seo2024rxnflow]` — RxnFlow: *Generative Flows on Synthetic Pathway for Drug Design*,
  arXiv:2410.04542 (ICLR 2025). Code: https://github.com/SeonghwanSeo/RxnFlow.
- `[bengio2021gflownet]` — the active-learning loop (Alg. 1) every entrant follows.
- `[koziarski2024rgfn]` — RGFN, the generator RxnFlow is being compared against.
- `[slabicki2020cr8]` — the 6TD3 system (DDB1·CDK12–cyclinK·CR8) the oracle scores.

**Packages.**
- RxnFlow (+ bundled Recursion `gflownet` v0.2.0) — the generator; used by
  `validation/generators/rxnflow/task.py`, `al_loop.py`.
- gnina + QuickVina2-GPU — the docking oracle backend, reached via
  `scripts/score_batch.py` (`Docking6TD3GpuOracle`).
- RDKit — SMILES canonicalization + descriptors, throughout.

## Method

> Commands below are the ones actually run for job 69518.

1. `bash external/setup_rxnflow.sh` — build the `rxnflow` env, install RxnFlow, prepare
   the env directory (10k debug building blocks + **109 `real.txt`** reaction templates;
   `hb_edited.txt` was too sparse on the debug library → non-finite loss, see status box).
2. CPU smoke: `conda run -n rxnflow python validation/generators/rxnflow/run_rxnflow_al.py
   --cfg validation/configs/rxnflow_smoke.yaml --seed-csv
   experiments/active_learning/6td3/seed_6td3.csv --device cpu --root-dir /tmp/rxnflow_smoke`
   (mock oracle; validates the full loop + route logging).
3. Real run: `sbatch experiments/active_learning/rxnflow_6td3/submit_rxnflow_6td3.sh` —
   3 rounds, GPU docking oracle, on a healthy node.

## Results

**Run:** job 69518, balam004, 51 min wall-clock. Per round the loop fit the proxy `M`
on the accumulated set, trained the synthesis GFlowNet 300 steps, sampled 32 unique
molecules, and docked them through the shared GPU oracle. Training stayed finite
throughout (TB loss spikes transiently when the policy hits high-reward molecules — e.g.
step 750 = 2251 — but never non-finite; the `real.txt` env fix held).

| Round | \|D\| (+added) | oracle mean | median | best | frac ≤ −2.0 | int. div. | novelty | MW (mean) | QED |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 439 (+31) | −1.834 | −1.806 | **−4.223** | 15/32 (47%) | 0.864 | 1.0 | 539 | 0.27 |
| 2 | 471 (+32) | −1.106 | −0.862 | −3.094 | 6/32 (19%) | 0.896 | 1.0 | 465 | 0.41 |
| 3 | 503 (+32) | −1.289 | −1.163 | −2.897 | 9/32 (28%) | 0.900 | 1.0 | 463 | 0.39 |

**Pooled over all 96 generated molecules:** best **−4.223**, median −1.187, mean −1.405,
**30/96 (31%) ≤ −2.0**, 5/96 (5%) ≤ −3.0. Every candidate synthesizable
(**`has_route=1`, 96/96**), synthesis length mean **1.97 reactions** (3× 1-step, 93×
2-step; `max_reactions=3` never bound). Mean MW 489, heavy atoms 34, QED 0.36, Lipinski
48/96. Top-5 generated (all 2-step routes):

| `dvina` | num_rxn | MW | QED | SMILES |
|---|---|---|---|---|
| −4.223 | 2 | 429 | 0.43 | `COc1ccc(N)c(Nc2ccc(OC)nc2OCCNc2cc(C)ccc2Cl)n1` |
| −3.694 | 2 | 558 | 0.27 | `CCOc1cc(F)cc(CN(CC(=O)Nc2cccc3nn[nH]c23)…` |
| −3.605 | 2 | 475 | 0.23 | `COc1ccc(Nc2cc(N)c(-c3noc(-c4nc(C)[nH]c4C(=O)O)n3)c(Cl)c2)…` |
| −3.094 | 2 | 422 | 0.36 | `Cc1cc(NC(=O)Nc2cccc(Sc3ccc4ocnc4c3)c2F)ccc1N=O` |
| −3.028 | 2 | 601 | 0.26 | `COc1ccc(Nc2ncnc3c2ncn3-c2cc(CO)ccc2I)c(Sc2ncccc2F)n1` |

> Note: `top_k.csv` (rank-1 −4.915, `CC[C@@H](CO)Nc1nc(NC(=O)CCc2ccccc2)c2ncn(C)c2n1`)
> pools the shared seed `D_0` with the generated set, so its top rows are seed molecules
> (this exact rank-1 also tops FragGFN's Top-16 — it's a seed glue, not a generated one).
> The generator comparison therefore uses `candidates.csv` (generated-only), matching how
> entry `015` reported FragGFN's "96 suggested" numbers.

### Head-to-head — clean matched-oracle GPU three-way

All three entrants share the **identical** GPU docking oracle (`Docking6TD3GpuOracle`),
seed `D_0` (408 mol), budget (3 rounds × 32), β=8, and proxy `M`; only the generator
differs. Numbers are **generated-only** (the 96 suggested molecules; `top_k.csv` pools
the shared seed and is not a generator metric). RGFN = job **69517**, RxnFlow = job
**69518**, FragGFN = job 69482 (entry 015) — all on the GPU oracle on healthy nodes.

| Metric | RGFN (69517) | **RxnFlow** (69518) | FragGFN (69482) |
|---|---|---|---|
| best `dvina` | −3.90 | −4.22 | **−4.86** |
| median `dvina` | **−2.14** | −1.19 | −2.06 |
| mean `dvina` | **−2.14** | −1.41 | −1.84 |
| frac ≤ −2.0 | 42/96 (44%) | 30/96 (31%) | **52/96 (54%)** |
| frac ≤ −3.0 | 8/96 (8%) | 5/96 (5%) | 20/96 (21%) |
| int. diversity | 0.85 | **0.89** | 0.87 |
| **synthesizable route** | yes (by constr.) | **yes, 96/96 (2-step)** | **none** (`has_route=0`) |
| mean MW | 665 | **489** | ≈720 |
| mean QED | 0.12 | **0.36** | ≈0.15 |
| Lipinski pass | **2/96** | 48/96 | (low) |

**Reading it — two findings.**

*(1) On glue score, at matched oracle, the three are in the same league.* RGFN has the
best central tendency (median/mean −2.14), FragGFN the best single hit (−4.86) and the
most molecules past the cutoff (54%), RxnFlow trails modestly (median −1.19, 31%). No
generator dominates; the spread is small relative to round-to-round noise (all three show
non-monotonic per-round scores). So on the docking axis alone, RGFN does **not** beat a
non-synthesizable foil — the entry-015 conclusion holds under the clean matched oracle.

*(2) The real separation is on drug-likeness, and it splits the two synthesizable
generators.* RGFN and FragGFN share a failure mode — bloated, low-QED molecules (MW
665–720, QED 0.11–0.15, RGFN passing Lipinski just **2/96**). RxnFlow is the only entrant
that is **both** synthesizable **and** physically reasonable: ~180 Da lighter than RGFN
and 3× the QED (0.36 vs 0.12). That is a genuine methodological signal — RxnFlow's
Enamine building-block library confines the search to buildable, drug-like chemistry,
whereas RGFN's reaction-DAG action space, though synthesizable by construction, drifts
into the same oversized region FragGFN does. **The synthesizability RGFN sells is real
(routes), but it does not by itself buy drug-likeness — a building-block-constrained
generator gets that for free.** This reframes RGFN's selling point and is a concrete
lead for the discussion (Objective 5): pair RGFN's DAG with a QED/property term or a
tighter block library.

**Caveats.** RxnFlow's synthesis depth is short (≤2 steps observed; `max_reactions=3`
never bound) and all three runs are single-seed with a small 3×300-step budget — the
next refinement (Next Experiments) is ≥3 seeds + longer training for error bars before
any of these deltas is load-bearing.

### Local validation done this session (no heavy stack)
- `py_compile` of all new RxnFlow Python + the edited `scripts/score_batch.py` — passed.
- `bash -n` of `setup_rxnflow.sh` + `submit_rxnflow_6td3.sh` — passed.
- `rxnflow_6td3.yaml` `loop`/`oracle`/`proxy` blocks match `fraggfn_6td3.yaml` (budget +
  oracle identical) — confirmed by a parity check.
- Route-aware bridge round-trip (Trillium login, mock oracle): with `--routes`, the
  standard dataset emits `has_route=1`, `num_reactions` populated, `routes.jsonl` with one
  entry per candidate, `manifest.has_routes=True`, and `validate_candidate_dataset` →
  conformant; **without** `--routes` (FragGFN path) → `has_route=0`, no `routes.jsonl`,
  also conformant (no regression).
