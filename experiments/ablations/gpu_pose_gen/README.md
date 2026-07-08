# GPU pose-generation ablation — QuickVina2-GPU search vs gnina CPU search

**Question.** The 6TD3 oracle's cost is ~99% a *CPU* gnina/AutoDock-Vina
conformational *search* (Logs/012). Can we replace **only that search** with
GPU docking (QuickVina2-GPU) and keep the validated discrimination — i.e. recover
entry-006's **Vina ΔT2−T1 AUROC = 0.946** (Cohen's d 2.38)?

**What is swapped (and what is not).** Everything except the search is held fixed,
so this is a one-variable ablation against 006:

| step | 006 (baseline) | this experiment |
|---|---|---|
| Tier-2 pose search | gnina/Vina MCMC, **CPU**, exh 16, 9 modes | **QuickVina2-GPU**, 9 modes, one search |
| pose selection | max **CNNscore** (gnina) | max **CNNscore** (gnina `--score_only`) |
| Tier-2 / Tier-1 score | gnina Vina / gnina `--score_only` | gnina `--score_only` / gnina `--score_only` |
| differential | `Vina(T2) − Vina(T1)` | `Vina(T2) − Vina(T1)` |

Both tier scores come from gnina `--score_only` (same scoring function as 006), so
the AUROC is directly comparable. QuickVina2-GPU is used **only to generate poses**
— it exposes no `--score_only` (commented out of its source), and mixing a QV2
score with a gnina score would not cancel the two engines' calibration offset.
The search box reproduces gnina's `--autobox_ligand crystal_RC8.pdb --autobox_add 4`.

**Code under test.** `glue/oracles/docking_gpu_differential_oracle.py`
(`Docking6TD3GpuOracle`, a thin subclass of the reusable
`GpuDifferentialDockingOracle`). Smoke-validated on 3 molecules: known_22
dvina −2.51 (006 −2.20), known_36 −1.16 (−1.26), decoy_0195 −1.75 (−1.82).

## Files
- `dock_gpu.py` — re-dock the 408 entry-006 molecules through the GPU oracle;
  chunked, incremental, resumable. Output rows carry both the GPU differential and
  the 006 reference (for discrimination + per-molecule correlation).
- `analyze_gpu_pose_gen.py` — discrimination (AUROC / Cohen's d / MWU, GPU vs the
  recomputed 006 reference control) and agreement (Pearson/Spearman, bias, RMSE)
  + a GPU-vs-gnina scatter.
- `submit_gpu_pose_gen.sh` — Balam compute-node job (login CPU-time limit).

## Inputs (reused, identical molecules)
`../../oracle_validation/docking_6td3/{known_results.csv, decoy_cdk_results.csv}`
(160 known + 248 decoy, `status==ok`), and the receptors
`6TD3_tier{1,2}.pdbqt` + `crystal_RC8.pdb`.

## Run
```bash
sbatch experiments/ablations/gpu_pose_gen/submit_gpu_pose_gen.sh
```
Outputs land on `$SCRATCH/rgfn_runs/experiments/gpu_pose_gen/`
(`gpu_pose_gen_results.csv`, `discrimination_stats.csv`, `agreement_stats.csv`,
`gpu_vs_gnina_scatter.png`, `gpu_pose_gen_timing.csv`); copy the small CSVs/PNG
back here to commit.

## Read-out
- **AUROC ≈ 0.946** → GPU search preserves discrimination; the speedup is free.
- **AUROC drops** → QV2 poses are worse than gnina's search (the CNN can't find a
  native-like pose among them); the CPU search is load-bearing.
- High AUROC + modest per-molecule correlation → "different poses, same
  separation" (acceptable); low both → the swap breaks the signal.
