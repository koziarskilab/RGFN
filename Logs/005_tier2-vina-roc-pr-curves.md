# Tier 2 Vina discrimination curves — ROC and Precision-Recall for 6TD3 and 5HXB
**Date:** 2026-06-23

## Question

How well does raw Tier 2 docking score alone separate real molecular glues from decoys across both our tested protein systems?

## Context & Summary

**Context** — Entries 001–003 characterized discrimination using threshold-based metrics (percentage of molecules above a score cutoff, and the neosubstrate differential gap). Entry 003 showed a +78-percentage-point separation for 6TD3 vs. −3 points for CRBN on the neosubstrate differential. Those summary statistics collapse the full score distribution into a single threshold. ROC and PR curves show the full picture across every possible threshold and are the standard figures reviewers will expect when evaluating a scoring oracle. This entry produces those curves for both systems using their Tier 2 Vina score — the absolute ternary-complex docking score before any differential is applied. It also answers a specific ablation question flagged in docs/RESEARCH_CONTEXT.md: does Tier 2 absolute score alone work, or is the neosubstrate differential truly necessary?

**Summary** — We compute ROC-AUC and average precision (area under the PR curve) for Tier 2 Vina score as a binary classifier (real glue vs. decoy) for 6TD3 and 5HXB, using the same result CSVs produced in entries 002 and 003. No new docking is run. The output is a two-panel figure (ROC left, PR right) with both systems overlaid.

## Answer

Tier 2 Vina alone is a strong discriminator for 6TD3 (AUC = 0.890, AP = 0.872) and only a weak one for CRBN (AUC = 0.627, AP = 0.726). This confirms that 6TD3 is a high-quality oracle even on absolute binding score, and makes the ablation case that Tier 2 score alone is useful — but the neosubstrate differential (entry 003) further sharpens discrimination by removing the warhead-binding component. For CRBN, the modest absolute-binding signal (AUC above chance) is consistent with the fit-rate numbers from entry 003 (77% vs. 45%), but the differential adds nothing, confirming the system ceiling is structural.

## Relevance to our Publication

NeurIPS reviewers will ask for the ablation: does the neosubstrate differential add anything over just using the raw docking score? This entry provides half of that answer — Tier 2 Vina alone is already strong for 6TD3, which means any further gain from the differential is a bonus, not a necessity. It also provides publication-ready discrimination figures for both systems in a format that directly supports the oracle validation argument. The PR curve is especially important at NeurIPS because the known/decoy class ratio differs between systems (39% known for 6TD3 vs. 54% for CRBN), and PR curves are interpretable at different baselines.

## Next Experiments

**Refining for publication** — Add the neosubstrate differential (`ddb1_dvina` / `gspt1_dvina`) as a second pair of curves on the same figure, so the ROC/PR plot directly shows differential vs. absolute-score discrimination for both systems in one panel. This is the ablation figure.

**Next steps in project** — Run RGFN with the validated 6TD3 oracle and evaluate generated molecules against these baseline curves to show the generative model produces candidates with better oracle scores than random warhead-bearing molecules.

# Re-creation

## Relevant Files

Root: `./research/preprocessing/docking_gnina/analysis/`

**Scripts**
- `plot_discrimination_curves.py` — loads Tier 2 Vina scores for both systems, builds ROC and PR curves via sklearn, outputs `discrimination_curves.png`; paths are resolved relative to the script via `pathlib` so it runs from any directory

**Datasets** (inputs; no new docking run)
- `./research/preprocessing/docking_6td3/known_results.csv` — per-molecule 6TD3 docking results for 160 known CDK12-DDB1 glues (from entry 002, job 69271)
- `./research/preprocessing/docking_6td3/decoy_cdk_results.csv` — per-molecule 6TD3 docking results for 248 decoys (from entry 002, job 69271)
- `./research/preprocessing/docking_gnina/known_crbn_results.csv` — per-molecule CRBN docking results for 136 anchorable known glues (from entry 003, job 69272)
- `./research/preprocessing/docking_gnina/decoy_crbn_results.csv` — per-molecule CRBN docking results for 117 decoys (from entry 003, job 69272)

**Results**
- `./research/preprocessing/docking_gnina/analysis/discrimination_curves.png` — two-panel ROC + PR figure; both systems overlaid; dashed baseline lines on PR panel

## Relevant Versions

`0c3f154` — Add Tier 2 Vina ROC/PR discrimination curves (entry 005)

## Relevant Resources

**Sources**
- Entry 002 (`002_6td3-cr8-validation-and-discrimination.md`) — source of 6TD3 result CSVs
- Entry 003 (`003_crbn-vs-6td3-cross-system.md`) — source of CRBN result CSVs

**Packages**
- scikit-learn — `roc_curve`, `auc`, `precision_recall_curve`, `average_precision_score`; used in `plot_discrimination_curves.py`
- matplotlib — figure rendering; used in `plot_discrimination_curves.py`
- pandas / numpy — data loading and label array construction

## Method

1. Load `known_results.csv` and `decoy_cdk_results.csv` for 6TD3; `known_crbn_results.csv` and `decoy_crbn_results.csv` for CRBN. Filter to `status == 'ok'` rows only.
2. Concatenate Tier 2 Vina scores; assign binary labels (1 = known glue, 0 = decoy). Negate scores before passing to sklearn (more negative Vina = stronger predicted binding = more positive class).
3. Compute `roc_curve` + `auc` and `precision_recall_curve` + `average_precision_score` for each system.
4. Render two-panel figure: ROC left, PR right; dashed diagonal reference on ROC; dashed class-baseline references on PR. Save at 200 dpi.

```
~/anaconda3/bin/python research/preprocessing/docking_gnina/analysis/plot_discrimination_curves.py
```

## Results

| System | ROC-AUC | Avg Precision | n known | n decoy | PR baseline |
|---|---|---|---|---|---|
| 6TD3 (CDK12–DDB1) | 0.890 | 0.872 | 160 | 248 | 0.392 |
| 5HXB (CRBN–GSPT1) | 0.627 | 0.726 | 136 | 117 | 0.538 |

Note average precision is also PR-AUC

Only molecules with `status='ok'` included; CRBN n=136 known reflects the 177-anchorable set with 41 no-valid-pose failures (see entry 003). 6TD3 n=248 decoys reflects same decoy set scored in entry 002; all 248 returned `status='ok'`. Input score CSVs are from entries 002 and 003 — no new docking run.
