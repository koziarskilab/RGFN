# experiments/ — one directory per run/experiment

Every individual run / experiment lives in its **own self-contained directory**
here: its code (seed-gen, submit, analysis scripts), its committed inputs (seeds,
small result CSVs), its README, and its run outputs. Reusable, pipeline-wide code
does **not** live here — it graduates into `glue/` (the shipped pipeline) or
`scripts/` (the generic entry points); this tree holds only what is specific to a
particular experiment.

## Grouped by type

```
experiments/
├── active_learning/        # active-learning runs ([bengio2021gflownet] Alg. 1)
│   ├── seh/                #   sEH GPU-docking oracle run (exp 010)
│   ├── 6td3/               #   CDK12-DDB1 gnina-oracle run (exp 009)
│   ├── mock/               #   smoke test — mock oracle, CPU-only
│   └── probe/              #   smoke test — quick inner-loop probes
├── oracle_validation/      # does the docking oracle separate glues from decoys?
│   ├── docking_6td3/       #   6TD3/CR8 box docking (the validated testbed)
│   └── docking_crbn/       #   5HXB/CRBN warhead-anchored docking
└── ablations/              # studies characterizing the oracle/metric
    ├── pose_selection/     #   CNN- vs Vina-selected pose (exp 008)
    ├── sixway/             #   six-signal comparison (exp 006)
    └── mw/                 #   molecular-weight control (exp 007)
```

The **group tells you the kind of experiment** (an active-learning run vs an
oracle-validation study vs an ablation); the leaf dir names the system/variant.
Full write-ups for each are in [`../Logs/`](../Logs) (indexed in
`docs/RESEARCH_CONTEXT.md`).

## Committed vs. generated

Each run dir mixes **committed** material (code, seeds, small result CSVs, README)
with **generated run outputs**. Run outputs are timestamped subdirs
(`<YYYY-MM-DD_HH-MM-SS>/` holding `train/`, `logs/`, `modes/`, per-round
`active_learning/*.csv`, checkpoints) and are **git-ignored by pattern** — so a
run drops its outputs right next to its code without cluttering git. (The AL entry
point `scripts/active_learning.py` routes outputs here: e.g.
`active_learning_seh.gin` → `experiments/active_learning/seh/<timestamp>/`.)

Inputs (structures, building blocks, curated molecule sets, seeds' upstream
sources) live under [`../data/`](../data), not here.
