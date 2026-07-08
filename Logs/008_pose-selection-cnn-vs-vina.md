# 6TD3 — does pose selection (CNN vs. Vina) change the glue/decoy discrimination?

**Date:** 2026-06-25, ~4pm (drafted); run completed 2026-06-26, ~5:25pm (Balam job 69443)

## Question

When we dock a candidate glue and keep one pose to score, does it matter whether we keep the pose our neural-network ranker likes best or simply the one the docking program scores best?

## Context & Summary

Our oracle docks a molecule into the CDK12+DDB1 complex, keeps a single pose, and scores how much extra binding the recruited partner adds (the "Vina Tier 2 − Tier 1 differential"). Entry 006 ranked six candidate scoring signals and found this differential is the best discriminator between real glues and decoys (AUROC 0.946), but every one of those six was measured on the *same* pose — the one with the highest gnina CNN score. We have never tested whether the rule used to *pick* that pose matters. The natural cheaper alternative is to just keep the docking program's own top-scored pose (best Vina energy) and skip the CNN entirely. We added CNN pose-picking in the first place because plain Vina ranking buried the correct pose in earlier work (the CC-885/5HXB redock, entry 001), so this entry asks whether that still matters for 6TD3.

**Why this question is bigger than pose ranking — it decides whether we keep gnina at all.** The validated oracle's *discrimination signal is pure Vina* (the ΔT2−T1 differential uses the Vina energy, lower = better; entries 006/007). The gnina CNN never enters the score — its **only** remaining job in the whole oracle is to choose which of the `num_modes` docked poses we keep. That single job is also what pins us to the slow engine: gnina is CPU-bound (Vina Monte-Carlo search) and reloads its CNN model per call, so even fully batched on 4× A100 it runs ~1.57 s/mol for 6TD3 global docking (entry 004). RGFN's own in-loop engine, QuickVina2-GPU-2.1, does the same Vina search on the GPU at **~0.60 s/mol on a single A100** (entry 010, sEH). The CNN is therefore the *only* thing standing between us and that faster, simpler path. So this ablation is really a go/no-go on gnina: if Vina-selected poses discriminate as well as CNN-selected ones, the CNN has no remaining role and we should drop gnina entirely in favour of GPU Vina docking — the same speed lever entry 010 already proved out for sEH, brought back to the glue oracle.

This experiment re-docks the same known-glue and decoy molecules from entry 002 but keeps **every** pose and scores all of them against both receptors. With every pose's scores in hand we can, without any further docking, pick a pose two ways — by CNN score (what we do now) and by Vina energy (the "just take the top pose" alternative) — form the validated differential each way, and compare how well each separates real glues from decoys. The CNN-picked version doubles as an in-run sanity check that should land near the published 0.946.

## Answer

**Pose selection matters, and the CNN earns its keep.** Selecting each molecule's
scored pose by Vina energy instead of CNN score drops the validated differential's
AUROC from **0.946 → 0.795** (ΔAUROC −0.151) and more than halves the effect size
(Cohen's d 2.38 → 0.95). The two rules pick the **same** pose only 40.6% of the
time on known glues and 17.7% on decoys — so they disagree on the majority of
molecules, and that disagreement is exactly what degrades discrimination. The
in-run CNN control reproduced the entry-006 published baseline to the digit
(AUROC 0.946, Δ +0.000 vs. published — well within docking stochasticity),
confirming the all-poses pipeline is faithful to production and the gap is a real
effect, not a setup artifact.

The mechanism is visible in the violins (`pose_selection_violins.png`): under Vina
selection the **decoy** differential grows a long low-energy tail — plain Vina
ranking happily promotes spuriously tight-scoring poses for decoys too, so the
decoy and known distributions overlap far more. The CNN ranker suppresses those
junk poses, keeping decoys near zero. This is the 6TD3 echo of the 5HXB finding
(entry 001) that Vina ranking buries the correct pose: it still holds.

**Decision: CNN selection IS necessary — gnina stays.** This is the positive
branch below: the CNN has a demonstrated, load-bearing role as pose picker, so the
slower CPU gnina oracle is now justified by evidence rather than inherited from the
5HXB work. The go/no-go went *against* retiring gnina; we therefore owe a
GPU-docking-**plus-CNN-rescore** path for the glue loop to ever run multi-round
(see Next Experiments), rather than the pure GPU-Vina path a null result would have
unlocked.

**The decision this answer drives:**
- **If CNN selection is NOT necessary** (Vina-selected poses discriminate as well — AUROC within noise of the CNN arm, high pose agreement): the gnina CNN has no surviving role in the oracle, since the differential is already pure Vina. **We should retire gnina and dock with QuickVina2-GPU instead** — the engine RGFN uses in-loop — for the ~2.6× speedup measured in entry 010 (1.57 → 0.60 s/mol) and a much simpler, single-engine stack. This is the concrete payoff of a null result here, and the route to giving the 6TD3 glue loop the GPU oracle it still lacks.
- **If CNN selection IS necessary** (Vina selection meaningfully degrades the differential): the CNN earns its keep as a pose picker, gnina stays, and the slower CPU oracle is justified by evidence rather than inherited from the 5HXB work — but we then owe a GPU-docking-plus-CNN-rescore path for the glue loop to ever run multi-round.

## Relevance to our Publication

This is a design-justification ablation that Digital Discovery / J. Cheminformatics reviewers will ask for: we claim the gnina CNN is worth the extra machinery for pose selection, and a reviewer can reasonably ask "would the plain docking pose have done just as well?" This entry answers it head-on. A null result (CNN and Vina selection discriminate equally) does more than simplify — it lets us **drop gnina for GPU Vina docking**, turning a per-molecule cost of ~1.57 s into ~0.60 s (entry 010) and collapsing the oracle to a single engine shared with RGFN's in-loop docking, which is exactly the scalability story reviewers want for an active-learning oracle. A positive result (CNN selection wins) instead justifies the current design with evidence rather than assertion, and complements the entry-006 scoring-signal ablation by closing the remaining pose-*selection* gap. Either way the ablation converts an inherited assumption (CNN pose-picking, carried over from the 5HXB work) into a documented, defensible choice.

## Next Experiments

**Refining for publication**
- If the two rules diverge, add per-molecule pose RMSD between the CNN-picked and Vina-picked poses (the all-poses SDF is retained) to quantify *how different* the chosen geometries are, not just how often they differ.
- Repeat on the MW-matched subsets from entry 007 to confirm the conclusion is not an artifact of ligand size.

**Next steps in project**
- Lock the pose-selection rule justified here into the production `Docking6TD3Oracle`, then run RGFN against the validated oracle and show generated molecules populate the high-scoring region the known glues occupy.
- **If CNN selection is unnecessary, port the 6TD3 oracle to QuickVina2-GPU** the way `DockingSEHOracle` already docks sEH (entry 010): two-tier (T1/T2) GPU Vina docks driven by SMILES, differential `= vina_t2 − vina_t1`, no gnina/CNN. This is the missing GPU/compute-node story flagged in entry 010's "bring the speed lesson back to the glue oracles" — and the prerequisite for running the *glue* active-learning loop multi-round (entry 009 was killed by the login-node CPU cap on the gnina path).

# Re-creation

## Relevant Files

Root: `experiments/ablations/pose_selection/` (was `research/preprocessing/pose_selection_ablation/` when this entry was drafted; the repo was reorganized to the `experiments/<type>/<run>/` layout — entry-002 inputs it references now live under `experiments/oracle_validation/docking_6td3/`).

**Scripts**
- `dock_allposes.py` — variant of `docking_6td3/dock_cluster.py` that retains **all** `num_modes` docked poses and `--score_only`s every pose against Tier 1, emitting a per-pose CSV. Identical gnina flags / embedding / autobox / exhaustiveness / num_modes / seed / sharding to `dock_cluster.py` so the CNN-selected arm reproduces production as an in-run control.
- `analyze_pose_selection.py` — offline (no docking): for each molecule picks one pose by CNNscore (`max cnnsc_t2`, production) and by Vina (`min vina_t2`, ablation), forms `ddb1_dvina = vina_t2 − vina_t1` on each, and reports Cohen's d / AUROC / Mann-Whitney U per rule plus the pose-agreement fraction. Writes `pose_selection_stats.csv` + `pose_selection_violins.png`. Reuses the discrimination math from `full_comparison/discrimination_stats.py`. Reads `$DATA_DIR`, writes `$OUT_DIR` (defaults to `$DATA_DIR`, since `$HOME` is read-only on Balam compute nodes).
- `submit_pose_ablation.sh` — SLURM submit (debug_full_node, 4× A100, no explicit cpu/mem per Balam rules); runs the redock then the analysis, writing to `$SCRATCH`.
- `collect_results.sh` — run from a **login node** after the job: copies the result CSVs + figure from `$OUTDIR` (scratch) back into the repo dir, since the compute-node job can't write to read-only `$HOME`.
- `README.md` — standalone run + interpretation guide for a future agent (what each file is, how to run on Balam, how to read `pose_selection_stats.csv`).
- `smoke_test.py` — off-Balam gate (no gnina/GPU): exercises `load_all`/`embed`/`_poses` and the full analysis on synthetic per-pose CSVs; exits non-zero on failure. Run before spending a Balam allocation.

**Datasets** (inputs; same molecule set as entry 002)
- `../docking_6td3/../test-data/DDB1_CDK12_Glues.csv` — known CDK12-DDB1 glues (positives).
- `../docking_6td3/decoys_cdk.smiles` — warhead-matched decoys (negative control).
- `../docking_6td3/6TD3_tier{1,2}.pdbqt`, `../docking_6td3/crystal_RC8.pdb` — Tier-1 (CDK12 alone) / Tier-2 (CDK12+DDB1) receptors and the CR8 autobox reference (Balam-only; git-ignored `.pdbqt`).

**Results** (produced on Balam, into `$OUTDIR` on scratch; copy `*_allposes.csv` into the repo dir to commit)
- `known_allposes.csv` / `decoy_allposes.csv` — one row per (molecule, pose): `vina_t2`, `cnnsc_t2`, `cnnaff_t2`, `vina_t1`, `cnnaff_t1`, `pose_rank`.
- `pose_selection_stats.csv` — per-rule discrimination summary (Cohen's d, AUROC, AUROC vs. CNN).
- `pose_selection_violins.png` — known-vs-decoy `ddb1_dvina` violins, one panel per selection rule.

## Relevant Versions

Committed in `c447490` ("Refactor & timing"). The run completed (Balam job 69443) and
results were pulled into the repo with `collect_results.sh 69443` from balam-login01.
All under `experiments/ablations/pose_selection/`:
the scripts (`dock_allposes.py`, `analyze_pose_selection.py`, `submit_pose_ablation.sh`,
`collect_results.sh`, `smoke_test.py`, `README.md`), the collected results
(`known_allposes.csv`, `decoy_allposes.csv`, `pose_selection_stats.csv`,
`pose_selection_violins.png`), and this log.

## Relevant Resources

**Sources**
- Molecule set + 6TD3 docking pipeline: entry 002 (`002_6td3-cr8-validation-and-discrimination.md`), Balam job 69271.
- Scoring-signal ablation this extends (held pose selection fixed at max-CNNscore): entry 006 (`006_6td3-violin-distributions.md`); MW control: entry 007.
- Motivation for CNN pose selection (Vina ranking buried the native pose): entry 001 (`001_5hxb-crbn-anchored-docking.md`).
- CR8/6TD3 structure: Słabicki et al., *Nature* 2020 (doi:10.1038/s41586-020-2133-z).

**Packages**
- gnina v1.3.2 (CNN-rescored docking) — `dock_allposes.py`.
- RDKit (embedding/MMFF, SDF I/O) — `dock_allposes.py`.
- pandas / numpy / scipy (`mannwhitneyu`) / matplotlib — `analyze_pose_selection.py`.

## Method

*(Executed 2026-06-26 as Balam job 69443, debug_full_node, 4× A100 on balam007, 13.9 min.)*

1. **Redock retaining all poses** — `sbatch experiments/ablations/pose_selection/submit_pose_ablation.sh`. Embeds the entry-002 known+decoy molecules, docks each into Tier 2 (CDK12+DDB1, autoboxed on crystal CR8, exhaustiveness 16, num_modes 9, seed 42), keeps every pose, and `--score_only`s all poses against Tier 1 (CDK12 alone) → `known_allposes.csv` / `decoy_allposes.csv` in `$OUTDIR`.
2. **Pose-selection ablation** — `DATA_DIR=$OUTDIR python analyze_pose_selection.py` (run automatically by the submit script): re-pick each molecule's pose by `max cnnsc_t2` vs. `min vina_t2`, compute `ddb1_dvina` per rule, and report discrimination + pose agreement.
3. **Sanity check** — confirm the CNN-selected arm's AUROC lands near the entry-006 baseline (0.946); large divergence flags a pipeline mismatch, not a real effect.

## Results

Balam job **69443** (debug_full_node, 4× A100, balam007), 13.9 min wall: phase-1
embedding 2.8 min (408/409 embedded — 1 molecule failed RDKit embedding), phase-2
docking 11.0 min across 16 shards. Retained **160 known / 248 decoy** molecules =
1440 / 2232 per-pose rows (9 poses each).

| selection rule | select by | known median dVina | decoy median dVina | Cohen's d | AUROC | ΔAUROC vs CNN |
|---|---|---|---|---|---|---|
| cnn (production) | max(cnnsc_t2) | −2.20 | −0.60 | 2.38 | **0.946** | 0.000 |
| vina (ablation) | min(vina_t2) | −2.42 | −0.89 | 0.95 | **0.795** | **−0.151** |

(Mann-Whitney U: CNN p = 1.3e−52, Vina p = 4.3e−24 — both arms separate the
classes, but the CNN arm far more strongly.)

Pose agreement (same pose chosen by both rules): known **40.6%**, decoy **17.7%**.

In-run CNN control vs. entry-006 production baseline (AUROC 0.946): reproduces to
**0.946**, Δ +0.000 — pipeline faithful, the Vina gap is a real effect.
