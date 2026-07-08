# data/models/

Protein structures and model checkpoints used as **inputs** to the pipeline —
part of the consolidated `data/` inputs directory (see [`../README.md`](../README.md)).

What belongs here:
- Source structures (`*.cif` from RCSB) and the **generated docking tiers** the
  oracle-validation prep writes (`experiments/oracle_validation/clean*.py`
  reads/writes here): Tier 1 (E3 pocket) / Tier 2 (E3 + neosubstrate) `.pdb`, the
  native ligand, and the `.pdbqt` receptors.
- Trained RGFN checkpoints and oracle model weights (e.g. gneprop/seno `.ckpt`).

> Historically a top-level `models/` dir (now folded into `data/models/` so all
> pipeline inputs live under `data/`). On Balam these often live in `$SCRATCH`;
> keep large/regenerable artifacts out of git.

**Weights and large structures are git-ignored** (`.gitignore`:
`data/models/**/*.{ckpt,pt,pth}` + the global `*.cif/*.pdb/*.pdbqt` ignores).
Commit only small, essential reference files; document where the rest come from /
how to regenerate them.
