# sEH active-learning loop — GPU-docking oracle (experiment 010)

**Tooling for the sEH active-learning run.** Full write-up:
[`Logs/010_seh-gpu-docking-oracle.md`](../../../Logs/010_seh-gpu-docking-oracle.md).

## What this is

The active-learning restructuring of upstream's `configs/rgfn_seh_docking.gin`
(`[bengio2021gflownet]` Alg. 1): a cheap learned MPNN proxy `M` is the in-loop
reward RGFN trains against, and a **real QuickVina2-GPU sEH docking oracle** `O`
is queried only on each round's query batch. So docking never enters the inner
loop — the GFN trains on the plain `reaction` env.

The oracle itself is production code and lives under `glue/` (it ships):
- `glue/oracles/docking_seh_oracle.py` — `DockingSEHOracle`, a thin SMILES-facing
  wrapper over upstream `DockingMoleculeProxy` (the docking is byte-for-byte
  RGFN's own QuickVina2-GPU path).
- `configs/glue/active_learning_seh.gin` — wires proxy + oracle + dataset + loop.
- `scripts/active_learning.py` — the generic loop entry point (shared with 6TD3).

This directory holds the **sEH-specific run tooling** around that oracle: seed
generation, a quick live validation, and the HPC run script. Timestamped run
outputs land here too (git-ignored).

## Files

**Scripts**
- `make_seh_seed.py` — builds the seed dataset `D_0`: samples unique valid
  molecules from the *untrained* RGFN forward policy (the same sampler the loop
  uses each round, so `D_0` is in-distribution), docks them with `DockingSEHOracle`,
  and writes `smiles,label` to `experiments/active_learning/seh/seed_seh.csv`.
  Reports average docking time per molecule.
- `validate_seh_oracle.py` — quick correctness/speed smoke: docks a few known
  molecules + one invalid SMILES, asserts finite negative Vina energies and a
  `nan` for the invalid input. Cross-checks aspirin against the known −6.3.
- `submit_al_seh.sh` — Balam **compute-node** SLURM script: regenerates a larger
  `D_0` on `$SCRATCH`, then runs the multi-round loop via a gin overlay that
  repoints `seed_csv` at that seed. (Compute node, not login: the login node's
  per-process CPU-time cap would SIGXCPU-kill long conformer prep.)

## How to run

All commands from the **repo root**, in the `rgfn` conda env, after:

```bash
module load cuda/11.8.0                                      # dgl graphbolt + vina OpenCL
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:$LD_LIBRARY_PATH   # QuickVina2-GPU boost libs
```

```bash
# quick validation (a few molecules)
python experiments/active_learning/seh/validate_seh_oracle.py

# (re)generate the seed D_0
python experiments/active_learning/seh/make_seh_seed.py \
    --cfg configs/glue/active_learning_seh.gin --n 300

# full multi-round loop on a compute node
sbatch experiments/active_learning/seh/submit_al_seh.sh
```

## Status

Oracle created and **validated live** — first on the balam-login01 A100 (Logs/010)
and **re-validated on the Trillium login H100** (2026-06-30): QuickVina2-GPU sEH
docking reproduces aspirin −6.3 (the known value), ibuprofen −6.9, caffeine −6.5,
at ~0.9 s/mol; invalid SMILES → `nan`.

The loop config `configs/glue/active_learning_seh.gin` is now at **GPU-pipeline
parity** with `active_learning_6td3_gpu.gin`: a `system='seh'` provenance tag and a
corrected `@SafeNumScaffoldsFound` metric block (the inherited `@NumScaffoldsFound`
used positive thresholds `[5,6,7,8]`, inert against our proxy's negative Vina
predictions — now `proxy_higher_better=False` with Vina-scale cutoffs grounded in
the seed `D_0`).

The **validation baselines can now target sEH too**: `validation/configs/{fraggfn,
rxnflow,scent}_seh.*` point each generator at the shared `docking_seh` oracle via
the bridge `scripts/score_batch.py`; submit scripts live under
`experiments/active_learning/{fraggfn,rxnflow,scent}_seh/`. The bridge path was
validated live (`--oracle docking_seh` scores aspirin −6.3 and writes the standard
candidate dataset).

The committed `seed_seh.csv` holds **250** sEH Vina labels (range −12.5 to +1.7,
median −7.0). The **full multi-round RGFN loop has not yet been run end-to-end** —
pending a Balam compute node (`submit_al_seh.sh` is ready).
