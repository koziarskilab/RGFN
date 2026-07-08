# Oracle validation — docking discrimination

Experiments that **validate the docking reward (oracle)** before it drives an
active-learning run: does the neosubstrate differential actually separate real
molecular glues from decoys? A glue must bind the E3-ligase pocket (via a
conserved **warhead**) *and* present an **arm** that makes a productive contact
with the neosubstrate. A good reward must reward the second part, not just warhead
presence. The lab notebook in [`../../Logs/`](../../Logs) records what each run found.

| dir | system (PDB) | E3 / neosubstrate | warhead | docking strategy |
|-----|--------------|-------------------|---------|------------------|
| `docking_crbn/` | **5HXB** — CRBN·DDB1·GSPT1 + CC-885 | CRBN / GSPT1 | glutarimide | **warhead-anchored** (deep cage can't be sampled blind) |
| `docking_6td3/`  | **6TD3** — DDB1·CDK12-cyclinK + CR8 | DDB1 / cyclin K | purine (ATP-site) | **straight box docking** (druggable pocket samples fine) |

**Headline finding** (see `../../Logs/003`): on the same neosubstrate-differential
metric, 6TD3 cleanly separates real glues from decoys (+78 pts) while 5HXB does
not (−3 pts) → **6TD3 is the oracle testbed**; 5HXB hits a cooperativity ceiling.
(Related ablations on this metric live in [`../ablations/`](../ablations).)

## Pipeline (5 steps per system)

```
1. PREP      clean*.py + obabel        cif -> receptor tiers (pdbqt) + native ligand
2. VALIDATE  redock the native ligand   confirm the pose is sampled/recovered
3. DECOYS    make_decoys*.py            warhead + random arm = baseline non-glues
4. DOCK      dock_cluster*.py           known + decoys -> Tier2 score + neosubstrate differential
5. COMPARE   compare_systems.py         known-vs-decoy separation; cross-system
```

### 1. Prep — structure → receptor tiers
`clean.py` (5HXB) and `clean_6td3.py` (6TD3) carve the mmCIF (kept in the shared
inputs dir `data/models/`) into docking tiers (Tier 1 = E3 pocket only, Tier 2 =
E3 + neosubstrate) and extract the native ligand. Then convert to pdbqt:
```bash
python clean_6td3.py                                   # -> data/models/6TD3_tier{1,2,3}*.pdb, crystal_RC8.pdb
obabel ../../data/models/6TD3_tier2_CDK12_DDB1.pdb -O docking_6td3/6TD3_tier2.pdbqt -xr -p 7.4
obabel ../../data/models/6TD3_tier1_CDK12.pdb     -O docking_6td3/6TD3_tier1.pdbqt -xr -p 7.4
```
(`data/models/*.cif`, `*.pdb`, and the receptor `*.pdbqt` are git-ignored —
regenerate them; download the `.cif` from RCSB by PDB id.)

### 2. Validate — can the engine recover the native pose?
```bash
python docking_6td3/redock_cr8.py        # 6TD3: blind redock CR8 -> RMSD to crystal (expect ~1.2 A)
python docking_crbn/anchor_dock.py       # CRBN: anchored recovery of CC-885 + IMiD references
```

### 3. Decoys — the discrimination control
`make_decoys*.py` builds realistic "fake glues" = the conserved warhead + a random
drug-like arm (amide/urea/sulfonamide/reductive-amination). If decoys score like
real glues, the proxy only reads pocket binding; if real glues win, it rewards a
productive arm. The curated known-glue inputs live in `data/validation-molecules/`.
```bash
python docking_6td3/make_decoys_cdk.py   # -> docking_6td3/decoys_cdk.smiles
python docking_crbn/make_decoys.py       # -> docking_crbn/decoys.smiles
```

### 4. Dock — the workhorse
`dock_cluster.py` (6TD3) / `dock_cluster_crbn.py` (CRBN) dock the known set + decoys,
take the best pose, and score it against Tier 1 to isolate the **neosubstrate
differential** (Tier2 − Tier1, via gnina `--score_only`). They shard molecules
across GPUs and **batch a whole shard into one gnina call** (the CNN model loads
once, not per molecule — the dominant speedup, see `../../Logs/004`).

```bash
# local, single GPU:
GNINA=/path/to/run_gnina.sh N_GPU=1 PROCS_PER_GPU=4 OUTDIR=./out WORK=/tmp/dock \
  python docking_6td3/dock_cluster.py
# SciNet Balam debug node (4x A100):
sbatch docking_6td3/submit_dock_6td3.sh
```

### 5. Compare — discrimination & cross-system
```bash
CRBN_DIR=docking_crbn TD3_DIR=docking_6td3 python compare_systems.py
```
Prints, per system, the known-vs-decoy gap on the neosubstrate differential (the
number that says whether the oracle is reward-worthy).

## Environment variables
| var | used by | meaning |
|-----|---------|---------|
| `GNINA` | dock/validate scripts | path to the gnina launcher (default: `/scratch/markymoo/gnina/run_gnina.sh`) |
| `N_GPU`, `PROCS_PER_GPU` | dock_cluster* | sharding fan-out (workers = N_GPU × PROCS_PER_GPU) |
| `OUTDIR`, `WORK` | dock_cluster* | result dir / scratch temp dir |
| `ANCHOR_PASS`, `ANCHOR_EMBED` | anchor_dock (CRBN) | `B`/`free` = flexible warhead anchor (default) |
| `MAX` | dock_cluster* | cap molecules per set (smoke testing) |
| `RESULTS_DIR`, `CRBN_DIR`, `TD3_DIR` | compare_systems | where result CSVs live |

## Adding a new glue system
1. Drop the ternary `.cif` in `data/models/`; write a `clean_<sys>.py` (copy
   `clean_6td3.py`) with the chain map → Tier 1 (E3) / Tier 2 (E3 + neosubstrate) +
   native ligand.
2. Pick a docking strategy: deep druggable pocket → reuse `dock_cluster.py` (box
   docking); shallow cage that blind docking can't reach → reuse `anchor_dock.py`.
3. Write `make_decoys_<sys>.py` (warhead scaffold + the shared reaction set in
   `make_decoys.py`); curated known glues go in `data/validation-molecules/`.
4. Add a `submit_<sys>.sh` and run; record the outcome as a new `../../Logs/NNN_*.md` entry.

## Layout
```
clean.py, clean_6td3.py     prep (cif -> tiers + native ligand; reads/writes ../../data/models/)
compare_systems.py          shared discrimination / cross-system analysis
docking_6td3/               6TD3/CR8: redock_cr8, make_decoys_cdk, dock_cluster, submit_dock_6td3
docking_crbn/               5HXB/CRBN: anchor_dock, make_decoys, dock_cluster_crbn, submit_dock_crbn

# shared inputs (consolidated, not in this dir):
../../data/models/                 structures + generated tiers (git-ignored)
../../data/validation-molecules/   curated known-glue datasets (DDB1_CDK12_Glues, CRBN_GSPT1_Glues)
```
