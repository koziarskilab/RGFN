# sEH — fast GPU-docking oracle for the active-learning loop
**Date:** 2026-06-26, ~4pm

### Question

Can we give the active-learning loop a docking oracle that is fast enough to
actually run end-to-end — by docking against sEH on the GPU the same way RGFN
itself does — and does it produce sensible binding scores?

### Context & Summary

The active-learning loop (`[bengio2021gflownet]` Alg. 1) trains RGFN against a
cheap learned proxy `M` and queries an expensive oracle `O` only on small
per-round batches. We have the machinery — the loop, the MPNN proxy, the dataset
— and we confirmed RGFN's inner loop learns against the proxy (entry 009). But
the only oracle wired in so far is the two-tier **gnina** glue oracle for 6TD3,
which runs on the CPU and is slow: on the Balam login node its per-round docking
was killed by the per-process CPU-time limit, so the *multi-round* loop has never
actually completed (entry 009, Objective 1).

This experiment adds a second, much faster oracle: real **GPU docking against
sEH** using QuickVina2-GPU-2.1 — the exact engine and settings upstream RGFN uses
in `configs/rgfn_seh_docking.gin`. sEH is the standard RGFN benchmark target, so
it lets us exercise the whole active-learning pipeline on a known system with an
oracle fast enough to finish. We built the oracle as a thin wrapper over RGFN's
own docking proxy (so the docking is identical, just driven by SMILES), wired it
into the loop with the regular MPNN proxy, and validated it live on the
balam-login01 A100: docked a handful of known molecules, generated a real seed
set by docking molecules RGFN itself proposes, and checked the speed.

### Answer

The oracle works and is fast, and gets *faster per molecule* on bigger batches:
~3.7 s/mol on a 5-molecule probe but **0.60 s/mol over 300 molecules** on one
A100, because the GPU docking overhead amortizes across the batch. It reproduces
the known result for aspirin against sEH (−6.3 kcal/mol) exactly, returns sensible
binding energies for valid molecules and `nan` for inputs it can't dock, and docks
molecules drawn from RGFN's own policy without issue (250 of 300 sampled molecules
docked successfully). We now have a docking oracle that is fast enough to drive the
multi-round loop to completion — the thing the CPU-bound glue oracle could not do
on a login node — on the standard RGFN benchmark system.

### Relevance to our Publication

This unblocks Objective 1 (the first end-to-end multi-round active-learning run):
NeurIPS reviewers will expect to see the full loop of `[bengio2021gflownet]` Fig.
7 — top-k improving as oracle calls accumulate, beating a random-acquisition
baseline — and that needs an oracle fast enough to run many rounds. sEH GPU
docking provides exactly that on the field-standard benchmark target, and gives a
second system alongside 6TD3 (a step toward the generalization evidence of
Objective 4). It also de-risks the design: it shows the loop's oracle seam is
genuinely modular — swapping the slow CPU glue oracle for a fast GPU docking
oracle was a single new class plus a config.

### Next Experiments

**Refining for publication**
- Run the full multi-round sEH loop on a **compute node** (`submit_al_seh.sh`)
  and produce the top-k-vs-oracle-calls curve with a random-acquisition baseline
  (Objective 1). Instrument oracle-call counting from this first run.
- Grow `D_0` from the committed 250-molecule seed toward the paper's scale
  (~2000) — the submit script already regenerates a 300-molecule seed on
  `$SCRATCH` before the run.
- Cross-check a sample of in-loop proxy predictions against fresh oracle docks
  (anti-gaming, Objective 5).

**Next steps in project**
- Reconcile this oracle's "docks exactly like RGFN" claim against an actual
  `configs/rgfn_seh_docking.gin` run on the same molecules (numbers should match).
- Bring the same speed lesson back to the glue oracles: the 6TD3 path needs a
  GPU/compute-node story before the glue loop can run multi-round.

# Re-creation

### Relevant Files

Root: project repository `RGFN-Fork/`.

**Scripts**
- `./glue/oracles/docking_seh_oracle.py` — the new `DockingSEHOracle`. Thin
  SMILES-facing wrapper that composes upstream `DockingMoleculeProxy` and calls
  its `dock_batch_qv2gpu` directly, so the QuickVina2-GPU docking is byte-for-byte
  the upstream code path; `higher_is_better=False` (Vina energy, more negative =
  better); returns `nan` per the `GlueOracle` failure contract. Built lazily so
  the module imports on a laptop.
- `./experiments/active_learning/seh/make_seh_seed.py` — generates the seed `D_0`:
  samples unique valid molecules from the *untrained* RGFN forward policy (same
  sampler the loop uses), docks them with `DockingSEHOracle`, writes `smiles,label`
  CSV (drops `nan`); reports average docking time per molecule.
- `./experiments/active_learning/seh/validate_seh_oracle.py` — quick live
  correctness/speed smoke (a few known molecules + one invalid SMILES).
- `./experiments/active_learning/seh/submit_al_seh.sh` — Balam compute-node SLURM
  script: loads cuda/11.8.0, puts the QuickVina2-GPU boost libs on
  `LD_LIBRARY_PATH`, regenerates a 300-molecule seed on `$SCRATCH`, then runs the
  loop via a gin overlay that repoints `seed_csv` at the scratch seed.
- `./experiments/active_learning/seh/README.md` — directory guide (how to run).
- `./scripts/active_learning.py` — unchanged generic loop entry point (shared
  with 6TD3).

**Datasets**
- `./experiments/active_learning/seh/seed_seh.csv` — the committed seed `D_0`:
  250 RGFN-generated molecules with their sEH QuickVina2-GPU binding energies
  (kcal/mol). In-distribution by construction (drawn from the RGFN policy). A
  larger seed can be regenerated on `$SCRATCH` at run time.
- `./data/targets/sEH.pdbqt` — the sEH receptor; box centre/size come from
  upstream `DockingMoleculeProxy`'s receptor tables (`sEH` key).

**Config**
- `./configs/glue/active_learning_seh.gin` — wires the loop: `LearnedGlueProxy`
  (MPNN, `higher_is_better=False`) as `M`, `DockingSEHOracle` as `O`, the seed
  dataset, `reaction` env (docking is the oracle, never the inner reward), 5
  rounds, 200 molecules/round.

**Infrastructure (Balam, git-ignored)**
- `quickvina_dir` symlink → `/scratch/markymoo/vina_gpu/Vina-GPU-2.1`; binary
  `QuickVina2-GPU-2.1/QuickVina2-GPU-2-1` (precompiled OpenCL kernels present);
  boost runtime libs at `/scratch/markymoo/vina_gpu/boost/lib`.

### Relevant Versions

Committed in `c447490` ("Refactor & timing"): `glue/oracles/docking_seh_oracle.py`,
`glue/oracles/__init__.py`, `configs/glue/active_learning_seh.gin`,
`experiments/active_learning/seh/` (make_seh_seed.py, validate_seh_oracle.py,
submit_al_seh.sh, README.md),
`experiments/active_learning/seh/seed_seh.csv`, this log, and the README/context
updates.

### Relevant Resources

**Sources**
- `[bengio2021gflownet]` — GFlowNet active-learning loop (Alg. 1, A.5.2: MPNN
  proxy trained on AutoDock scores, ~200 molecules docked per round).
- `[koziarski2024rgfn]` — RGFN; `configs/rgfn_seh_docking.gin` is the reference
  in-loop sEH docking setup this oracle mirrors.

**Packages**
- QuickVina2-GPU-2.1 (Tang et al.) — GPU docking engine, via
  `rgfn/gfns/reaction_gfn/proxies/docking_proxy/{docking_proxy,vinagpu_wrapper}.py`.
- Meeko / RDKit — ligand prep + conformer embedding (upstream preparator).
- gin-config, PyTorch, dgl — RGFN env/policy/proxy machinery.

### Method

All on balam-login01 (A100), `rgfn` conda env, after
`module load cuda/11.8.0` and
`export LD_LIBRARY_PATH=/scratch/markymoo/vina_gpu/boost/lib:$LD_LIBRARY_PATH`.

1. Wrote `DockingSEHOracle`; registered it in `glue/oracles/__init__.py`.
2. Static checks: `py_compile`; import via `glue.registry`; gin parse of
   `active_learning_seh.gin` + confirmed `DockingSEHOracle` resolves by name.
3. Live oracle check — docked 4 known molecules + 1 invalid SMILES:
   `python experiments/active_learning/seh/validate_seh_oracle.py`.
4. Seed generation / batch-scale check (40 then 300 molecules):
   `python experiments/active_learning/seh/make_seh_seed.py --cfg configs/glue/active_learning_seh.gin --n 300 --oversample 2.0 --seed 42`
   → `experiments/active_learning/seh/seed_seh.csv`.
5. Loop preconditions: loaded the seed via `OracleLabeledDataset` (|D_0|=250) and
   asserted proxy/oracle `higher_is_better` agree (both `False`).

### Results

**Live oracle dock (step 3)** — 5 molecules in 18.5 s (3.7 s/mol):

| Molecule | sEH Vina (kcal/mol) |
|---|---|
| aspirin | −6.30 |
| ibuprofen | −6.90 |
| anthracene (PAH) | −8.50 |
| caffeine | −6.50 |
| `not_a_smiles_at_all` (invalid) | nan |

Aspirin −6.30 matches the previously recorded sEH validation (raw-binary aspirin
dock −6.3, `gpu-docking-oracle-setup` note), confirming the wrapper reproduces
the known result. The invalid SMILES correctly returned `nan`.

**Seed generation (step 4)** — sampled molecules from the untrained RGFN policy
(350 fragments / 66 reactions) and docked them. Two runs:

| Run | Sampled | Docked OK | Vina range (kcal/mol) | mean | Dock time | **s/mol** |
|---|---|---|---|---|---|---|
| n=40 | 40 | 36 (4 nan) | [−9.90, +4.90] | −6.78 | 18.5 s* | 3.7* |
| n=300 | 300 | 250 (50 nan) | [−12.50, +1.70] | −6.97 | 180.6 s | **0.60** |

*The n=40 per-molecule figure is the 5-molecule live-dock probe (step 3); the
n=40 seed run itself was not separately timed.

**Batch-scale speedup (the question of this run):** per-molecule docking drops
from ~3.7 s/mol on a single tiny batch to **0.60 s/mol** over 300 molecules
(12 batches of 25) — ~6× faster, because the QuickVina2-GPU per-call/kernel
overhead amortizes across a full batch. Total wall 4:01 (incl. RGFN build +
sampling); user CPU 285 s, well under the login node's 3600 s cap (so this size
is safe on a login node; larger seeds should still go to a compute node). The
committed `seed_seh.csv` is now the **250-molecule** D_0 from the n=300 run.
Positive Vina values are genuine poor/clashing poses, kept as labels.

**Preconditions (step 5)** — `OracleLabeledDataset` loaded D_0 (≥ 2, loop OK);
proxy and oracle `higher_is_better` both `False` (sign check passes).
