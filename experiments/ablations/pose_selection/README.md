# Pose-selection ablation — CNN vs. Vina (experiment 008)

**Test dataset + code for one ablation.** Full write-up: [`Logs/008_pose-selection-cnn-vs-vina.md`](../../../Logs/008_pose-selection-cnn-vs-vina.md).

## The question

Our 6TD3 oracle docks a glue, keeps **one** pose, and scores the DDB1 cooperativity
differential (`ddb1_dvina = Vina(Tier2) − Vina(Tier1)`; lower = better glue). Production
keeps the pose with the highest gnina **CNN score**. This experiment asks: would keeping
the **top-Vina pose** instead (skip the CNN) discriminate real glues from decoys just as
well? Entry 006 ranked the *scoring* signals but always on the CNN pose — this closes the
pose-*selection* gap.

The trick: re-dock the entry-002 molecules keeping **every** pose with both receptors'
scores, so both selection rules can be applied to the *same* poses offline. The comparison
is therefore paired and immune to docking stochasticity.

## Files

**Scripts**
- `dock_allposes.py` — re-docks the entry-002 known+decoy sets on 6TD3/CR8, retains **all**
  `num_modes` poses, and `--score_only`s every pose against Tier 1. Faithful copy of
  `../docking_6td3/dock_cluster.py` (same gnina flags) except it keeps all poses instead of
  collapsing to the CNN-best one. Writes per-pose CSVs. **Balam/GPU only.**
- `analyze_pose_selection.py` — no docking. Re-picks each molecule's pose by `max(cnnsc_t2)`
  (production) vs `min(vina_t2)` (ablation), forms `ddb1_dvina` each way, and reports
  Cohen's d / AUROC / Mann-Whitney + pose-agreement. **Runs anywhere with pandas/scipy/matplotlib.**
- `submit_pose_ablation.sh` — SLURM wrapper (debug_full_node, 4× A100). Runs the redock then
  the analysis, all to `$SCRATCH`.
- `collect_results.sh` — copies results from scratch back into this dir. Run from a **login
  node** (see "Why two machines" below).

**Datasets** (committed test results — present only after a run + collect)
- `known_allposes.csv` / `decoy_allposes.csv` — one row per (molecule, pose):
  `idx,id,set,smiles,status,n_poses,pose_rank,vina_t2,cnnsc_t2,cnnaff_t2,vina_t1,cnnaff_t1`.
  `pose_rank` 1 = gnina's top-ranked pose. These are the raw ablation data.

**Results** (committed)
- `pose_selection_stats.csv` — per-rule discrimination summary.
- `pose_selection_violins.png` — known-vs-decoy `ddb1_dvina` violins, one panel per rule.

## How to run

### Why two machines
Heavy docking runs on a Balam **compute node**, where `$HOME` (this repo) is **read-only** —
the job can only write to `$SCRATCH`. So results are written to scratch during docking, then
pulled back from a **login node** (writable `$HOME`). Don't try to make the job write here.

### Steps
```bash
# 0. SMOKE TEST first (no Balam/GPU needed) — confirms the code is sound before you
#    spend a compute allocation. Exits 0 on success.
python experiments/ablations/pose_selection/smoke_test.py

# 0b. (on Balam) optional cheap dry-run of the FULL docking path on a few molecules:
#     MAX=2 caps to 2 known + 2 decoy so you confirm gnina wiring before the real run.
#     sbatch ... or, in an interactive GPU session: MAX=2 N_GPU=1 PROCS_PER_GPU=1 \
#       python experiments/ablations/pose_selection/dock_allposes.py

# 1. Dock on Balam (writes per-pose CSVs + analysis to $SCRATCH/rgfn_runs/pose_abl_<jobid>)
sbatch experiments/ablations/pose_selection/submit_pose_ablation.sh

# 2. After it finishes, from a LOGIN node, pull results into this dir:
bash experiments/ablations/pose_selection/collect_results.sh <jobid>
#    (or pass the full $OUTDIR path instead of the job id)

# 3. (optional) re-generate stats/figure locally from the committed CSVs:
#    DATA_DIR + OUT_DIR both default to this dir, which is writable off-cluster.
python experiments/ablations/pose_selection/analyze_pose_selection.py
```

`analyze_pose_selection.py` env knobs: `DATA_DIR` (where the `*_allposes.csv` live; default =
this dir), `OUT_DIR` (where stats/figure go; default = `DATA_DIR`). On-cluster the submit
script sets `DATA_DIR=$OUTDIR` so outputs stay on writable scratch.

### Smoke test without Balam
`analyze_pose_selection.py` needs only per-pose CSVs. Point `DATA_DIR` at any dir holding
`known_allposes.csv` / `decoy_allposes.csv` with the header above to exercise the analysis
path (used during development with synthetic data).

## How to read the results

`pose_selection_stats.csv` has one row per selection rule (`cnn`, `vina`):

| column | meaning |
|---|---|
| `auroc` | P(random glue scores better than random decoy) on `ddb1_dvina` — the headline discriminator |
| `auroc_vs_cnn` | this rule's AUROC minus the CNN rule's — **the ablation answer** |
| `cohens_d` | standardized effect size (oriented so positive = glues better) |
| `known/decoy_median_dvina` | median differential per set (native kcal/mol units) |

What to look for:
- **In-run sanity:** the `cnn` row's AUROC should land near the entry-006 baseline **0.946**
  (`BASELINE_CNN_AUROC` in the script). A big gap means a pipeline mismatch, not a real effect.
- **The verdict:** `auroc_vs_cnn` for the `vina` row. ≈0 ⇒ pose selection doesn't matter, the
  cheaper top-Vina pose is fine. Clearly negative ⇒ CNN pose-picking earns its keep.
- **Pose agreement** (printed to stdout): fraction of molecules where both rules pick the same
  pose. High ⇒ the choice rarely bites; low ⇒ selection is doing real work even if AUROC is similar.
