# 6TD3 — controlling the glue-vs-decoy comparison for molecular weight

**Date:** 2026-06-25, ~5pm

## Question

When our oracle scores a real glue higher than a fake one, is it reading genuine glue cooperativity — or just the fact that real glues happen to be bigger molecules?

## Context & Summary

**Context** — Entry 006 ranked the six candidate oracle signals on the 6TD3 system and crowned the Vina Tier 2 − Tier 1 differential the best discriminator between real glues and decoys (it ranks a real glue above a fake one ~95% of the time). But it flagged one loose end that a reviewer would seize on: real glues and our decoys might differ in *size*, and bigger molecules generally dock better simply by making more contacts. If that were the whole story, our "glue signal" would be a molecular-weight artifact in disguise. This entry closes that gap.

**Summary** — We computed the molecular weight of every known glue and every decoy, confirmed the two groups really do differ in size, then built a size-balanced comparison: each real glue was paired with a decoy of nearly identical molecular weight (1:1 nearest-neighbor matching), and we re-scored discrimination on those matched pairs. To make sure the result wasn't an accident of which pairs got matched, we repeated the matching 500 times and report the average with a confidence interval.

## Answer

The size confound is real and large — known glues are ~93 Da heavier than decoys on average — so the concern was justified. But the winning signal survives it: once glues and decoys are matched on molecular weight, the Vina Tier 2 − Tier 1 differential stays a strong discriminator (it drops only from 0.95 to 0.87). In sharp contrast, the *absolute* scores were mostly reading size — Vina Tier 2 falls from 0.89 to 0.65, and kinase-pocket-only Vina Tier 1 collapses from 0.69 to **below chance (0.38)**. The differential wins precisely because it cancels out ligand size and leaves the DDB1-recruitment signal we actually care about.

## Relevance to our Publication

This is the exact robustness check a Digital Discovery or J. Cheminformatics reviewer demands of any "our metric separates actives from decoys" claim: *prove it isn't a trivial property like size*. We now answer it head-on. The result is also stronger than a mere defense — it turns molecular weight from a liability into supporting evidence for our central design choice. The differential isn't just the best discriminator (entry 006); it's the best *because* it is the one signal that doesn't reduce to ligand size, which is exactly the argument for using a Tier 2 − Tier 1 differential rather than an absolute docking score. The before/after AUROC figure is a clean supplementary panel that makes the ablation case visually.

## Next Experiments

**Refining for publication** — The matched comparison necessarily restricts to the molecular-weight range the two sets share; 51 of the heaviest known glues sit above the decoy ceiling and can't be matched at all. Reporting a held-out (cross-validated) version of the entry-006 confusion matrices, and generating decoys that span the full glue weight range, would make the control airtight. Pairing this figure with the entry-005 ROC/PR curves and the entry-006 violins would give a single self-contained discrimination story.

**Next steps in project** — Wire the molecular-weight-robust signal (Vina Tier 2 − Tier 1) into the active-learning loop as the 6TD3 oracle reward, and confirm RGFN's *generated* molecules earn their high differential scores through arm-recruitment rather than by simply drifting to high molecular weight — i.e. watch generated-molecule size alongside the reward so the generator can't game the oracle the way the absolute scores could be gamed.

# Re-creation

## Relevant Files

Root: `research/preprocessing/full_comparison_mw/`

Scripts:
- `./mw_common.py` — shared spine: loads the entry-002 docking CSVs (status='ok'), adds an RDKit molecular-weight column and the two Tier 2 − Tier 1 differentials, and provides the AUROC helper and the reproducible greedy MW-matching routine. The other three scripts import it.
- `./mw_distributions.py` — draws the known-vs-decoy molecular-weight violin and reports the size gap (Cohen's d, AUROC, Mann-Whitney U, Kolmogorov-Smirnov). Establishes that the confound exists.
- `./mw_score_correlation.py` — regresses each of the six oracle signals on molecular weight (Spearman/Pearson, pooled and within-group), showing which signals are size-driven.
- `./mw_matched_discrimination.py` — the decisive test: 1:1 nearest-neighbor MW matching (15 Da caliper), recomputes AUROC on the matched subset, bootstraps 500 randomized matchings for a 95% CI, and compares to the full-set AUROC.

Datasets (inputs, reused from entry 002 — no new docking):
- `../docking_6td3/known_results.csv` — per-molecule docking scores for 160 real CDK12-DDB1 glues (job 69271); the KNOWN+ positives. SMILES column is the molecular-weight source.
- `../docking_6td3/decoy_cdk_results.csv` — per-molecule docking scores for 248 purine-armed decoys (job 69271); the negative control.

Results:
- `./mw_distribution.png` / `./mw_distribution.csv` — molecular-weight violin and the size-gap summary stats.
- `./mw_score_correlation.png` / `./mw_score_correlation.csv` — 2×3 scatter grid (MW vs each metric) and the per-metric correlation table.
- `./mw_matched_auroc.png` / `./mw_matched_auroc.csv` — full vs MW-matched AUROC (bar pairs with bootstrap CI) and the per-metric numeric table.
- `./mw_match_balance.csv` — molecular-weight balance before vs after matching (medians, gap, MWU p).

## Relevant Versions

```
5633939 [docs] update commit in logs
c723fcb [CODE] Little determination of CNN vs Vina scoring on Tier 1 vs Tier 2 vs Tier 2 - Tier 1.
f418de8 [CODE] Mock Active Learning Loop (no balam)
```

The four scripts and their outputs (now under `experiments/ablations/mw/` after the repo restructure) were committed in `13cf8b6` ("[CODE] MW Matched Decoys and corrected active learning oracle").

## Relevant Resources

**Sources**
- Score data originate from entry 002 (6TD3 discrimination run, Balam job 69271); CR8/6TD3 structure: Słabicki et al., *Nature* 2020 (doi:10.1038/s41586-020-2133-z), `[slabicki2020cr8]`.
- Builds directly on entry 006 (which metric discriminates best) and entry 005 (Tier 2 ROC/PR curves).

**Packages**
- RDKit 2023.09.5 (`rdkit.Chem.Descriptors.MolWt`, average molecular weight from SMILES) — `mw_common.py`.
- pandas 2.x, numpy — all scripts.
- scipy (`mannwhitneyu`, `ks_2samp`, `pearsonr`, `spearmanr`) — distribution tests and correlations.
- matplotlib 3.10.x — all figures.
- Run inside the `rgfn` conda env (only env with RDKit locally): `conda run -n rgfn python <script>`.

## Method

1. **Compute molecular weight** — `mw_common.load()` reads each entry-002 CSV, keeps `status=='ok'` rows (160 known, 248 decoy), and adds `mw = Descriptors.MolWt(MolFromSmiles(smiles))` plus the differentials `dvina`, `dcnnaff`.
2. **Establish the confound** — `python mw_distributions.py` draws the MW violin (decoy vs binder) and writes the size-gap stats (Cohen's d, AUROC, MWU, KS).
3. **Map size-dependence per signal** — `python mw_score_correlation.py` regresses each of the six signals on MW (pooled + within-group Spearman/Pearson) and writes the 2×3 scatter grid.
4. **Match and re-score** — `python mw_matched_discrimination.py` greedily pairs each known glue to a unique decoy with the closest MW (≤15 Da caliper, randomized visiting order), recomputes AUROC on the matched subset, repeats over 500 seeded matchings for a 95% percentile CI, and compares against the full-set AUROC; writes the before/after bar figure and the balance/AUROC tables.

## Results

n = 160 known glues, 248 decoys (status=ok), molecular weight from SMILES; same entry-002 poses (job 69271), no new docking.

**The confound is real (`mw_distribution.csv`).** Known glues are substantially heavier than decoys:

| set | n | median MW (Da) | mean MW (Da) | sd | range |
|---|---|---|---|---|---|
| known | 160 | 435.6 | 442.0 | 66.2 | 254–722 |
| decoy | 248 | 342.9 | 346.9 | 48.1 | 260–462 |

Median gap **+92.7 Da**; Cohen's d **1.70**; AUROC(MW alone) **0.885**; MWU p = 1.9e-39, KS p = 3.9e-32. Note the decoy MW ceiling is 462 Da — **51 known glues are heavier than any decoy**, so size-matching is only possible in the shared MW region.

**Absolute scores track size; differentials track it less (`mw_score_correlation.csv`).** Spearman ρ of each signal with MW:

| signal | ρ (pooled) | ρ (known only) | ρ (decoy only) |
|---|---|---|---|
| Vina Tier 1 | −0.66 | −0.70 | −0.54 |
| Vina Tier 2 | −0.83 | −0.79 | −0.65 |
| Vina ΔT2−T1 | −0.72 | −0.53 | −0.43 |
| CNN Tier 1 | +0.75 | +0.76 | +0.50 |
| CNN Tier 2 | +0.79 | +0.80 | +0.53 |
| CNN ΔT2−T1 | +0.50 | +0.29 | +0.12 |

Within each group, the differentials are the *least* MW-correlated signals of their scoring function (Vina Δ: −0.53/−0.43 vs Tier 2's −0.79/−0.65; CNN Δ: +0.29/+0.12 vs Tier 2's +0.80/+0.53) — the size term largely cancels between the two tiers.

**The decisive test — MW-matched discrimination (`mw_matched_auroc.csv`, `mw_match_balance.csv`).** Matching yields ~82 size-balanced pairs (canonical seeded match: 81 pairs, known median 398.5 vs decoy 398.5 Da, MWU p = 0.83 — balanced). AUROC before vs after matching (mean of 500 randomized matchings, 95% CI):

| signal | AUROC (full) | AUROC (MW-matched) | 95% CI | drop |
|---|---|---|---|---|
| **Vina ΔT2−T1** | 0.946 | **0.866** | [0.85, 0.88] | −0.080 |
| CNN ΔT2−T1 | 0.850 | 0.766 | [0.74, 0.79] | −0.084 |
| CNN Tier 2 | 0.907 | 0.761 | [0.75, 0.77] | −0.146 |
| CNN Tier 1 | 0.863 | 0.703 | [0.68, 0.72] | −0.159 |
| Vina Tier 2 | 0.890 | 0.648 | [0.63, 0.66] | −0.243 |
| Vina Tier 1 | 0.691 | **0.380** | [0.36, 0.40] | −0.311 |

Two clean conclusions. (1) **The two differentials lose the least to MW-matching** (−0.08 each) and the Vina differential remains the top discriminator overall (0.866) — its edge is not a size artifact. (2) **The absolute scores were largely reading size**: Vina Tier 2 sheds a quarter of its AUROC, and kinase-pocket-only Vina Tier 1 falls *below chance* (0.38), meaning once size is held fixed, raw warhead-pocket binding is if anything mildly anti-discriminative. This reorders the entry-006 ranking to put the differentials on top within each scoring function, and confirms the differential is the right oracle signal for reasons beyond raw AUROC. (Caveats: matching is restricted to the shared MW window, so the heaviest 51 glues are excluded; the matched n≈82 is roughly half the known set, reflected in the bootstrap CI; the six signals remain non-independent since Tier 2 = Tier 1 + differential.)
