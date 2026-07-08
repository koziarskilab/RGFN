# 6TD3 — GPU pose generation (QuickVina2-GPU) vs gnina CPU search

**Date:** 2026-06-29, ~3pm

## Question

Will GPU generated docking poses (faster) still work as a good discriminator instead of CPU generated docking poses (slower but accurate)?

## Context & Summary

Our 6TD3 glue oracle scores a molecule by the **DDB1 differential** — how much extra binding a molecule gains once the recruited partner protein (DDB1) is present (entry 006: this signal separates real glues from decoys 95% of the time, AUROC 0.946). The catch is speed: entry 012 showed that **99.3% of the docking time is one step** — a *conformational search* on the CPU that tries thousands of poses (~58 s per molecule). Everything else (picking the best pose with the gnina CNN, and the two energy evaluations that form the differential) is nearly free. So the obvious lever is to move just that one expensive search onto the GPU.

This experiment swaps **only** the pose search: QuickVina2-GPU generates the candidate poses on the GPU, and then everything downstream is left exactly as entry 006 had it — the gnina CNN still picks the most native-like pose, and both halves of the differential are still scored by gnina (so the numbers are directly comparable to the 0.946 baseline). We re-dock the same 408 molecules entry 006 used (160 known glues + 248 decoys) and check two things: does the GPU version reproduce the 0.946 discrimination, and does it agree with the CPU version molecule-by-molecule.

## Answer

GPU pose generation keeps **most** of the glue/decoy discrimination but does not fully match the CPU search: the GPU differential scores **AUROC ≈ 0.88** (0.875 on the compute node, 0.892 on the login node — the spread is QuickVina2-GPU's stochastic search, which has no fixed seed) versus **0.948** for the gnina CPU search recomputed on the very same 408 molecules (which reproduces entry 006's 0.946 exactly, confirming the comparison is fair). So moving only the pose search to the GPU costs about **0.06–0.07 AUROC**. Known glues land in the same place (median −2.20, identical to gnina), but decoys drift slightly more glue-like (−0.78 vs −0.60), which narrows the gap — the same "a worse pose can score spuriously well" effect entry 008 saw, now coming from the pose *source* rather than the pose *picker*. The payoff is speed: the GPU search runs at **~1.1 s/molecule vs the CPU search's 58 s/mol (entry 012)** — roughly **50× faster** on the step that was 99% of the docking cost. Verdict: the GPU oracle is a viable fast screen when a ~0.06 AUROC trade is acceptable (e.g. driving the active-learning loop at scale); for maximum discrimination the CPU gnina search still wins.

A 3-molecule smoke test first confirmed the pipeline reproduces entry-006 values closely (known_22 −2.51 vs −2.20; known_36 −1.16 vs −1.26; decoy_0195 −1.75 vs −1.82), with the gnina CNN selecting high-confidence poses (CNNscore 0.80–0.95) from the GPU-generated set.

## Relevance to our Publication

This is the throughput result that makes the oracle practical at scale. Entry 008 established that the gnina CNN pose-picker is load-bearing and concluded the oracle "needs a GPU-dock **plus** CNN-rescore path, not pure QuickVina2-GPU." This entry builds exactly that path and tests whether it holds the 0.946 discrimination. A reviewer (Digital Discovery / J. Cheminformatics) asking "your oracle is expensive — can it run inside an active-learning loop at realistic scale?" gets a direct answer: a GPU pose-generation oracle that preserves the validated signal, with a measured speed-up over the CPU search that entry 012 pinned as the bottleneck.

## Next Experiments

**Refining for publication.** If discrimination holds, add the wall-clock comparison (GPU s/molecule vs the entry-012 CPU baseline of 58 s/mol on the Tier-2 search) using the substep-timing CSV this run records, so the speed-up is quantified, not asserted. QuickVina2-GPU pose sampling is stochastic (no fixed seed), so a repeat run would show the differential's run-to-run noise. If discrimination drops, dock a sweep of QV2 search effort (`thread`/`exhaustiveness`) and pose count (`num_modes`) to find where the GPU poses become good enough for the CNN.

**Next steps in project.** Wire `Docking6TD3GpuOracle` into the multi-round active-learning loop (replacing the CPU `Docking6TD3Oracle`) and re-run the entry-011 loop to confirm the generator still produces real-glue-range molecules, now at GPU throughput.

# Re-creation

## Relevant Files

Root: `experiments/ablations/gpu_pose_gen/`

**Scripts:**
- `./dock_gpu.py` — re-docks the 408 entry-006 molecules through `Docking6TD3GpuOracle`; chunked, incremental, resumable. Each row carries both the GPU differential and the entry-006 reference (for discrimination + per-molecule correlation).
- `./analyze_gpu_pose_gen.py` — discrimination (AUROC / Cohen's d / Mann-Whitney for the GPU `dvina` and, as an internal control, the recomputed `ref_dvina`) and agreement (Pearson/Spearman, bias, RMSE) + a GPU-vs-gnina scatter.
- `./opencl_healthcheck.c` — 20-line raw-OpenCL context probe; the submit script runs a prebuilt copy as a per-node gate (catches wedged nodes like balam008) and it doubles as the minimal SciNet repro.
- `./run_login_loop.sh` — login-node fallback driver (bounded resumable `--max-chunks` processes under the login CPU rlimit); used for the cross-check run.
- `./submit_gpu_pose_gen.sh` — Balam compute-node SLURM job (1 GPU); `--exclude=balam008` + OpenCL health gate, then dock + analyse. `--output`/`--error` are absolute `$SCRATCH` paths (read-only `$HOME` on compute nodes — Logs/012).

**Oracle under test:**
- `../../../glue/oracles/docking_gpu_differential_oracle.py` — `Docking6TD3GpuOracle` (thin 6TD3 subclass) and the reusable `GpuDifferentialDockingOracle`. QuickVina2-GPU generates `num_modes=9` Tier-2 poses from one GPU search (box = crystal-ligand bbox + 4 Å, reproducing gnina `--autobox_ligand crystal_RC8.pdb --autobox_add 4`); Open Babel splits the multi-MODEL pdbqt to per-pose SDF; one batched gnina `--score_only` gives per-pose CNNscore + Affinity; the max-CNNscore pose is scored against Tier 1 with a second gnina `--score_only`; `dvina = Vina(T2) − Vina(T1)`. Composes upstream `DockingMoleculeProxy` (Meeko + QuickVina2-GPU) for the search rather than re-implementing it.

**Inputs (reused from entry 002/006, identical molecules):**
- `../../oracle_validation/docking_6td3/known_results.csv` — 160 known CDK12-DDB1 glues (`status==ok`): SMILES + reference `ddb1_dvina`, `vina_t2`, `vina_t1`, `cnnsc_t2`.
- `../../oracle_validation/docking_6td3/decoy_cdk_results.csv` — 248 purine-armed decoys (negative control), same columns.
- `../../oracle_validation/docking_6td3/6TD3_tier2.pdbqt` — CDK12 + DDB1 (Tier 2): the ternary complex the pose is searched against.
- `../../oracle_validation/docking_6td3/6TD3_tier1.pdbqt` — CDK12 alone (Tier 1): isolates the kinase-pocket contribution so the differential reads the DDB1 bonus.
- `../../oracle_validation/docking_6td3/crystal_RC8.pdb` — native CR8 ligand, used to autobox the QV2 search region.

**Results (on `$SCRATCH`, copy small CSVs/PNG back to commit):**
- `/scratch/markymoo/rgfn_runs/experiments/gpu_pose_gen/gpu_pose_gen_results.csv` — per-molecule GPU vs reference differentials.
- `/scratch/markymoo/rgfn_runs/experiments/gpu_pose_gen/discrimination_stats.csv`, `agreement_stats.csv`, `gpu_vs_gnina_scatter.png`.
- `/scratch/markymoo/rgfn_runs/experiments/gpu_pose_gen/gpu_pose_gen_timing.csv` — substep wall-clock (qv2_dock / pose_convert / tier2_score / pose_select / tier1_score).

**Job Logs:**
- `/scratch/markymoo/rgfn_runs/experiments/gpu_pose_gen/gpu_pose_gen-69478.{out,err}` — headline compute run (balam009). Job 69468 was the all-`no_pose` failure on wedged balam008; 69477 a buggy health-gate false negative (in-job gcc compile failed). Login-node cross-check via `run_login_loop.sh` → `login_loop.log` + `*_login.csv`.

## Relevant Versions

Committed in `d1d77f9` ("Active learning & GPU dock"): `glue/oracles/docking_gpu_differential_oracle.py`, `glue/oracles/__init__.py`, and `experiments/ablations/gpu_pose_gen/` (`dock_gpu.py`, `analyze_gpu_pose_gen.py`, `submit_gpu_pose_gen.sh`, `run_login_loop.sh`, `opencl_healthcheck.c`, `balam008_opencl_report.md`, `README.md`), plus this log and the `docs/RESEARCH_CONTEXT.md` index row.

Most recent commit before this run: `e07d953 small AL loop logs`.

## Relevant Resources

**Sources**
- Pose data origin and CR8/6TD3 structure: Słabicki et al., *Nature* 2020 (doi:10.1038/s41586-020-2133-z). Reference differentials from entry 002 (Balam job 69271) / entry 006.
- QuickVina2-GPU / Vina-GPU-2.1: Tang et al. (Vina-GPU+ single-receptor-multi-ligand GPU docking). gnina CNN scoring: McNutt et al., *J. Cheminformatics* 2021 (GNINA 1.0).

**Packages**
- QuickVina2-GPU-2.1 (`$SCRATCH/vina_gpu/Vina-GPU-2.1`) — GPU pose search, via upstream `DockingMoleculeProxy` / `VinaDocking`.
- gnina (`$GNINA=/scratch/markymoo/gnina/run_gnina.sh`) — `--score_only` CNN selection + two-tier Vina scoring.
- Open Babel 3.1.0 — multi-MODEL pdbqt → SDF (`docking_gpu_differential_oracle._pdbqt_to_sdf_blocks`).
- meeko 0.5.0 (Meeko ligand prep), RDKit; numpy/pandas/scipy/scikit-learn/matplotlib — `analyze_gpu_pose_gen.py`.

## Method

1. **Submit** the compute-node job: `sbatch experiments/ablations/gpu_pose_gen/submit_gpu_pose_gen.sh` (headline job 69478 on balam009; gate excludes wedged balam008). Env: `module load cuda/11.8.0`; `conda activate rgfn`; `LD_LIBRARY_PATH += $SCRATCH/vina_gpu/boost/lib`; `GNINA` set.
2. **Dock** all 408 molecules (`dock_gpu.py --num-modes 9 --exhaustiveness 8000 --chunk 32`): for each molecule, QuickVina2-GPU runs one Tier-2 search → 9 poses; gnina `--score_only` scores all poses (CNNscore + Affinity); the max-CNNscore pose is scored against Tier 1; `dvina = Vina(T2) − Vina(T1)`. Writes `gpu_pose_gen_results.csv` incrementally; substep times to `gpu_pose_gen_timing.csv`.
3. **Analyse** (`analyze_gpu_pose_gen.py`): AUROC / Cohen's d / Mann-Whitney on the GPU `dvina` (known vs decoy) and on the recomputed `ref_dvina` control; Pearson/Spearman + bias/RMSE of GPU `dvina` vs `ref_dvina`; scatter PNG.

Validation before the full run: 3-molecule oracle smoke (reproduces entry-006 values), `dock_gpu.py --limit 4` (CSV schema + resume), `analyze` on a 6-row mixed CSV (stats + plot path).

## Results

Two full runs: a compute-node job (**balam009, job 69478**, the headline) and a login-node loop (cross-check). 407/408 molecules docked (`status==ok`); 1 `no_pose`.

**Discrimination (n = 159 known / 248 decoy, `status==ok`), known-vs-decoy on `dvina`:**

| run / metric | known median | decoy median | Cohen's d | AUROC |
|---|---|---|---|---|
| GPU dvina — compute (job 69478) | −2.204 | −0.782 | 1.36 | **0.875** |
| GPU dvina — login loop | −2.208 | −0.786 | 1.59 | **0.892** |
| ref dvina (gnina search, recomputed, same mols) | −2.201 | −0.600 | 2.40 | **0.948** |
| entry-006 baseline (from Log 006) | −2.20 | −0.60 | 2.38 | 0.946 |

The reference control reproduces the 006 baseline exactly (0.948 ≈ 0.946), confirming an apples-to-apples comparison. GPU pose-gen loses ≈0.06–0.07 AUROC; the two GPU runs differ by 0.017 (QV2 stochastic search, no seed). The loss is on the decoy side (median −0.78 vs −0.60); known-glue medians match gnina.

**Agreement (GPU vs gnina-search `dvina`, compute run, n=407):** Pearson r = 0.58, Spearman r = 0.67, mean bias (GPU − ref) = −0.17 kcal/mol, RMSE = 0.88 kcal/mol. (Login run: Pearson 0.66, RMSE 0.77.) Moderate per-molecule agreement — "different poses, similar separation".

**Cost (compute run substep totals, 408 mols):** `qv2_dock` 456.6 s = **1.12 s/mol** (the GPU search); `tier2_score` 60.2 s; `tier1_score` 41.3 s; `pose_convert` 36.9 s (openbabel); `pose_select` ~0 s. Whole oracle ≈ 1.46 s/mol. Compare entry-012's CPU `tier2_dock` = **58 s/mol** → ≈**52× faster** on the search step.

**Compute-node OpenCL health (operational finding).** The first compute submit (job 69468) returned all `no_pose`: QuickVina2-GPU uses NVIDIA OpenCL, and on **balam008** `clCreateContext` fails with `CL_OUT_OF_RESOURCES (-5)` while CUDA is unaffected. A 20-line raw-OpenCL repro (`opencl_healthcheck.c`) **fails on balam008 but succeeds on balam009** (same A100-SXM4 / driver 580) and on the login node — so it is a wedged-node state, not a driver/partition/kernel-cache issue (ruled out: OpenCL loader version, `CUDA_VISIBLE_DEVICES`, MPS, compute-mode, MIG). The submit script now `--exclude=balam008` and gates on a live OpenCL context check before docking (exit 42 on a bad node). balam008 reported to SciNet.

**Smoke-test (3 molecules, pre-run validation):**

| id | GPU dvina | 006 ref dvina | GPU CNNscore | 006 CNNscore |
|---|---|---|---|---|
| known_22 | −2.51 | −2.20 | 0.88 | 0.79 |
| known_36 | −1.16 | −1.26 | 0.95 | 0.98 |
| decoy_0195 | −1.75 | −1.82 | 0.80 | 0.79 |
