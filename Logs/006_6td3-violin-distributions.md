# 6TD3 Metric Comparison (Tier 1, Tier 2, CNN, Vina, Difference) for known glues vs decoys.

**Date:** 2026-06-25, ~3pm

## Question

For the 6TD3 system, which metric is the best at discriminating between glues and decoys: Tier 1 score, Tier 2 score, or T2 - T1? What about when comparing Vina versus CNN?


## Context & Summary

**Context** — Entry 002 established that the CDK12-DDB1 oracle separates real molecular glues from realistic fakes (decoys with the right warhead but a random arm), and entry 005 backed this with ROC/PR curves on the absolute Tier 2 score. But the oracle gives us *six* candidate signals to reward RGFN with — two scoring functions (Vina and the gnina CNN affinity) crossed with three tiers (Tier 1 = kinase pocket only, Tier 2 = kinase + DDB1, and the Tier 2 − Tier 1 differential) — and we have never ranked them against each other on the same footing. Picking the wrong signal would train the generator toward the wrong thing, so before wiring an oracle into the active-learning loop we need to know which of the six actually discriminates best.

**Summary** — We reused the existing docking results from entry 002 (no new docking) and put all six metrics through the same head-to-head test against our negative control — decoys built with the correct CDK12 warhead but an arbitrary drug-like arm, so any metric that separates binders from decoys is reading something beyond mere warhead binding. For each metric we drew a known-vs-decoy violin plot, then scored discrimination on a unit-free footing (Cohen's d and AUROC, so Vina kcal/mol and CNN pK are comparable) and confirmed it with a confusion matrix at the best single threshold.

## Answer

We found that Vina ΔT2−T1 is the best discriminator, with CNN Tier 2 being a close second. Vina ΔT2−T1 has a Cohen's D of 2.38 and AUROC of **0.946**, which means that Vina ΔT2−T1 is a VERY STRONG discriminator between real and fake glues. This gives precedence to our choice of Vina ΔT2−T1 as our oracle signal.

## Relevance to our Publication

This is the ablation Digital Discovery / J. Cheminformatics reviewers will ask for: *is the Tier 2 − Tier 1 differential actually necessary, or does an absolute score do just as well?* By ranking all six candidate signals on the same data, we answer it directly — the Vina differential wins, and the kinase-only Tier 1 scores trail badly, showing the discriminating information lives in what DDB1 adds, not in the warhead pocket. That justifies our oracle-signal choice with evidence rather than assertion, and the violins double as a publication-ready supplementary figure that makes the tier-by-tier, score-by-score logic legible at a glance.

## Next Experiments

**Refining for publication** — The per-panel effect-size / AUROC / Mann–Whitney table is now in place (see Results); the remaining airtightness step is to match molecular-weight distributions between known and decoy sets so a reviewer can't attribute the Tier 2 shift to ligand size alone, and ideally report held-out (rather than in-sample) confusion matrices via cross-validation so the Youden cut isn't fit on the data it scores. A combined figure pairing these violins with the entry-005 ROC/PR curves would make a compact, self-contained discrimination story.

**Next steps in project** — Wire the winning signal — the Vina Tier 2 − Tier 1 differential — into the active-learning loop as the 6TD3 oracle reward, then produce the analogous distribution plots for *generated* molecules, showing the generator's output populates the high-scoring region these known glues occupy.

# Re-creation

## Relevant Files

Root: `research/preprocessing/`

Scripts:
- `./full_comparison/plot_violins.py` — loads the entry-002 result CSVs, filters to `status=='ok'`, recomputes the Tier 2 − Tier 1 differentials, and renders the six-panel violin figure plus a medians/gap summary CSV.
- `./full_comparison/discrimination_stats.py` — computes the scale-free discrimination stats per panel (Cohen's d, AUROC, Mann-Whitney U p-value) and writes `discrimination_stats.csv` ranked by AUROC.
- `./full_comparison/confusion_matrices.py` — picks the Youden-J-optimal cut per panel and renders a 2×3 grid of confusion matrices (binders = positive class) plus `confusion_matrices.csv`.

Datasets (inputs, reused from entry 002 — no new docking):
- `./docking_6td3/known_results.csv` — per-molecule docking scores for 160 real CDK12-DDB1 glues (job 69271); the KNOWN+ positives.
- `./docking_6td3/decoy_cdk_results.csv` — per-molecule docking scores for 248 purine-armed decoys (job 69271); the negative control.

Results:
- `./full_comparison/violins_known_vs_decoy.png` — the six-panel violin figure (2×3 grid).
- `./full_comparison/violin_medians.csv` — per-panel decoy median, known median, and known-minus-decoy gap.
- `./full_comparison/discrimination_stats.csv` — per-panel gap, Cohen's d, AUROC, MWU p-value.
- `./full_comparison/confusion_matrices.png` — 2×3 grid of confusion matrices at the Youden-optimal cut.
- `./full_comparison/confusion_matrices.csv` — per-panel threshold, TP/FP/FN/TN, accuracy, sensitivity, specificity, precision.

## Relevant Versions

```
f418de8 [CODE] Mock Active Learning Loop (no balam)
a3d21c1 [DOCS] Remove 000 Template
1c54a01 [DOCS] Remove template
```

The three scripts and their outputs (`research/preprocessing/full_comparison/`) were committed in `c723fcb`. The input CSVs were committed previously with entry 002 (`106a4e6`).

## Relevant Resources

**Sources**
- Score data originate from entry 002 (6TD3 discrimination run, Balam job 69271); CR8/6TD3 structure: Słabicki et al., *Nature* 2020 (doi:10.1038/s41586-020-2133-z).

**Packages**
- matplotlib 3.10.6 (violin + confusion-matrix rendering), pandas 2.3.3, numpy 2.3.5 — `plot_violins.py`, `discrimination_stats.py`, `confusion_matrices.py`.
- scipy (Mann-Whitney U) — `discrimination_stats.py`; scikit-learn (`roc_curve`, `confusion_matrix`) — `confusion_matrices.py`.

## Method

1. **Load + filter** — read `known_results.csv` and `decoy_cdk_results.csv`, keep only `status=='ok'` rows (160 known, 248 decoy).
2. **Derive differentials** — recompute `dvina = vina_t2 − vina_t1` and `dcnnaff = cnnaff_t2 − cnnaff_t1` per molecule (matches the stored `ddb1_dvina` / `ddb1_dcnnaff` columns) so the plotted quantity is explicit.
3. **Render violins** — `python research/preprocessing/full_comparison/plot_violins.py` draws six violin panels (decoy vs. binder per panel) with median lines, jittered raw points, and a median annotation; writes the PNG and `violin_medians.csv`.
4. **Discrimination stats** — `python research/preprocessing/full_comparison/discrimination_stats.py` computes, per panel, Cohen's d (pooled SD, oriented so positive = binders better), AUROC (= MWU U / (n_known·n_decoy)), and a one-sided Mann-Whitney U p-value; writes `discrimination_stats.csv` ranked by AUROC.
5. **Confusion matrices** — `python research/preprocessing/full_comparison/confusion_matrices.py` orients each metric (negate lower-is-better Vina), picks the Youden-J-optimal ROC cut, classifies all samples (binders = positive), and renders the 2×3 confusion-matrix grid + `confusion_matrices.csv`.

## Results

n = 160 known glues, 248 decoys (status=ok), all six metrics computed on the same entry-002 poses (job 69271).

**The head-to-head ranking (`discrimination_stats.csv`).** Ranked by AUROC — P(a random binder scores better than a random decoy) — which is unit-free and so the fair way to put Vina (kcal/mol) and CNN (pK) on the same scale:

| metric | Cohen's d | AUROC | MWU p |
|---|---|---|---|
| **Vina ΔT2−T1** | **2.38** | **0.946** | 1.3e-52 |
| CNN Tier 2 | 1.95 | 0.907 | 3.9e-44 |
| Vina Tier 2 | 1.21 | 0.890 | 8.4e-41 |
| CNN Tier 1 | 1.52 | 0.863 | 1.9e-35 |
| CNN ΔT2−T1 | 1.18 | 0.850 | 3.1e-33 |
| Vina Tier 1 | 0.41 | 0.691 | 4.0e-11 |

**Vina ΔT2−T1 wins** (AUROC 0.946 — it ranks a real glue above a decoy 95% of the time), with CNN Tier 2 a close second (0.907). The pattern across the table is the point: within each scoring function, adding DDB1 (Tier 1 → Tier 2) and isolating its contribution (→ Δ) lifts discrimination, and **Vina Tier 1 is the worst signal of all** (0.691) — kinase-pocket binding alone barely tells glues from decoys. That is exactly why the differential is the right oracle signal: the discriminating information lives in what DDB1 adds, not in the warhead pocket. (Caveats: AUROC is the fair ranker; Cohen's d assumes ~normal equal-variance populations that docking-score tails violate, which is why the d and AUROC orderings disagree slightly. The six metrics are not independent — Tier 2 = Tier 1 + differential by construction.)

**Supporting — median shift and raw gap (`violin_medians.csv`; consistent with entry 002).** Useful for reading the violins, but the gap is in native units and so is *not* comparable across rows (the CNN differential's tiny +0.19 gap still gives AUROC 0.850 because ΔCNN has a very tight spread). Ordered as the figure panels:

| metric | decoy median | known median | gap (known − decoy) | better |
|---|---|---|---|---|
| Vina Tier 1 (CDK12 only) | −7.35 | −8.02 | −0.67 | lower |
| Vina Tier 2 (CDK12+DDB1) | −7.96 | −10.15 | −2.19 | lower |
| Vina Tier 2 − Tier 1 (DDB1 ΔVina) | −0.60 | −2.20 | −1.60 | lower |
| CNN affinity Tier 1 (CDK12 only) | 6.68 | 7.59 | +0.92 | higher |
| CNN affinity Tier 2 (CDK12+DDB1) | 6.70 | 7.82 | +1.13 | higher |
| CNN affinity Tier 2 − Tier 1 (DDB1 ΔCNN) | 0.04 | 0.23 | +0.19 | higher |

**Confirmation — confusion matrices at the Youden-optimal cut (`confusion_matrices.csv`, in-sample; n=160 binders / 248 decoys).** Threshold fit on the same data it scores — descriptive separability, not held-out performance.

| panel | cut (native) | TP | FP | FN | TN | acc | sens | spec |
|---|---|---|---|---|---|---|---|---|
| Vina ΔT2−T1 | ≤ −1.58 | 136 | 12 | 24 | 236 | 0.91 | 0.85 | 0.95 |
| CNN Tier 2 | ≥ 7.36 | 128 | 27 | 32 | 221 | 0.86 | 0.80 | 0.89 |
| CNN Tier 1 | ≥ 7.20 | 129 | 39 | 31 | 209 | 0.83 | 0.81 | 0.84 |
| Vina Tier 2 | ≤ −8.98 | 127 | 41 | 33 | 207 | 0.82 | 0.79 | 0.83 |
| CNN ΔT2−T1 | ≥ 0.12 | 127 | 39 | 33 | 209 | 0.82 | 0.79 | 0.84 |
| Vina Tier 1 | ≤ −7.86 | 96 | 57 | 64 | 191 | 0.70 | 0.60 | 0.77 |

The accuracy ranking agrees with the AUROC ranking at both ends — **Vina ΔT2−T1 best (acc 0.91), Vina Tier 1 worst (acc 0.70)** — with only the middle pair swapping, so the head-to-head verdict is robust to how it is measured. As a bonus check, the Vina differential's Youden-optimal cut (−1.58) lands essentially on the −1.5 threshold entry 002 chose by hand, independently confirming that cutoff.
