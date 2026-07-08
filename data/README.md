# data/ — the single inputs directory

Everything the pipelines read as **input** lives here (and only here). This folds
in what used to be a separate top-level `models/` dir, so there is one obvious
home for inputs.

| Path | What | Owner |
|---|---|---|
| `chemistry.xlsx` | RGFN reaction building-block library | **upstream** (hardcoded in `configs/envs/reaction.gin`; do not move) |
| `targets/` | docking receptor `.pdbqt` (sEH, Mpro, ClpP, TBLR1) | **upstream** (hardcoded in `rgfn/.../docking_proxy.py`; do not move) |
| `models/` | protein structures (`.cif`), generated docking tiers, model checkpoints | ours |
| `validation-molecules/` | curated known-glue molecule sets used as validation inputs | ours |
| `synthetic/` | datasets the pipeline *generates* (the "synthetic datasets out" goal) | ours (outputs, git-ignored) |

> Why `data/` and not a fresh `inputs/`: `chemistry.xlsx` and `targets/` are
> **upstream** and their paths are hardcoded in `rgfn/` + `configs/`. Renaming
> `data/` would break the upstream-mergeable boundary (see `CLAUDE.md`), so `data/`
> is the inputs root and our other inputs fold into it.

Run **outputs** do not live here — those go to a run's own dir under
`experiments/<group>/<run>/` (see [`../experiments/README.md`](../experiments/README.md)).
Large/regenerable artifacts are git-ignored (see `.gitignore`).
