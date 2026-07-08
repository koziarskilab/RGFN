# Active-learning run — 6TD3 / CDK12–DDB1 (gnina oracle)

The runnable **active-learning loop** (`[bengio2021gflownet]` Alg. 1) on the
validated CDK12–DDB1 / 6TD3 glue system. Write-up:
[`Logs/009`](../../../Logs/009_first-rgfn-inner-loop-learning.md) (first inner-loop run).

## The loop (one round)

1. **Fit `M`** — train the proxy on everything labelled so far, `D_{i-1}`.
2. **Train RGFN** — `Trainer.train()` against the reward `M(x)^β`.
3. **Sample `B`** — draw a query batch from the trained policy.
4. **Score `B` with `O`** — the expensive 6TD3 two-tier gnina docking differential.
5. **Accumulate** — `D_i = D̂_i ∪ D_{i-1}`; repeat for `N` rounds.

The expensive oracle enters training **only** by retraining the proxy — never as
a direct RGFN reward.

## Components (reusable pieces ship under `glue/` + `configs/glue/`)

| Role | Paper | Class / file |
|---|---|---|
| Oracle `O` | expensive scorer | `glue.oracles.Docking6TD3Oracle` (DDB1 `ddb1_dvina` differential) |
| Proxy `M` | MPNN over atom graph (A.4) | `glue.proxies.LearnedGlueProxy` — reuses RGFN's `MPNNet` |
| Dataset `D` | `D_i = D̂_i ∪ D_{i-1}` | `glue.datasets.OracleLabeledDataset` |
| Loop | Alg. 1 | `glue.active_learning.ActiveLearningLoop` |
| Configs | — | `configs/glue/active_learning_6td3{,_inner,_mini}.gin` |
| Entry point | — | `scripts/active_learning.py` (generic; shared with sEH) |

## This directory (6TD3-specific run material)

- `seed_6td3.csv` — `D_0`: 408 real validated-docking labels (160 known glues +
  248 decoys), `label` = the **`ddb1_dvina`** differential (`Vina(Tier2) −
  Vina(Tier1)`, more-negative = better), built from
  `../../oracle_validation/docking_6td3/{known,decoy_cdk}_results.csv`. This is the
  signal the evidence selected — **not** the CNNaffinity differential:
    - Log 002: +78pt discrimination (known median −2.20 vs decoy −0.60);
    - Log 006: best of six candidate signals, **AUROC 0.946** (CNN ΔT2−T1 only
      0.850; absolute Vina Tier 1 just 0.69);
    - Log 007: that edge is **molecular-weight-robust** (0.946 → 0.866 after
      size-matching, while absolute scores collapse).

  Smaller than the paper's `|D_0| = 2000` (documented divergence).
- `build_seeds.py` — rebuilds `seed_6td3.csv` (and `../mock/seed_mock.csv`) from
  the oracle-validation result CSVs. Run from the repo root.
- `submit_al_6td3_mini.sh` — Balam **compute-node** SLURM script for the 3-round
  mini run (Logs/009). Compute node, not login: the per-round gnina docking is
  CPU-bound and the login node's CPU-time cap would SIGXCPU-kill it.
  `sbatch experiments/active_learning/6td3/submit_al_6td3_mini.sh`.
- Timestamped run outputs (`<ts>/active_learning/dataset_round_*.csv`, `top_k.csv`,
  checkpoints, logs) land here and stay git-ignored.

## Running

```bash
# local smoke test (CPU, no gnina — uses the mock oracle, ../mock/)
python scripts/active_learning.py --cfg configs/glue/active_learning_mock.gin
# real run (Balam — needs gnina, a GPU, and the prepared 6TD3 receptors)
python scripts/active_learning.py --cfg configs/glue/active_learning_6td3.gin --seed 42
```

## Where the oracle was validated

The 6TD3 docking-oracle *validation* (the experiments that established the
differential metric) lives separately under
[`../../oracle_validation/docking_6td3/`](../../oracle_validation/docking_6td3/) —
that's oracle validation; this dir is the AL *run*.
`glue/tests/test_oracle_discrimination.py` guards the metric on a laptop (no gnina),
reproducing Log 002's +78pt gap.
