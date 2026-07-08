# Candidate dataset format (v1.0)

The **one** on-disk format for *candidate* datasets — the molecules a generator
proposed, with scores and (optionally) synthesis routes. Every generator we
benchmark emits this, so the harness reads them all uniformly.

- **Canonical implementation:** `glue/datasets/candidates.py`
  (`CandidateDataset`, `from_smiles_table`, `read_candidate_dataset`,
  `validate_candidate_dataset`). `CANDIDATE_SCHEMA_VERSION = "1.0"`.
- **Why it lives in `glue/`:** `validation/` may import from `glue/`, never the
  reverse (`docs/ARCHITECTURE.md`). Our production loop must *write* this format,
  so it lives where the loop can import it; the `validation/generators/` adapters
  and `validation/harness/` import it from here.

## Layout

One dataset is a directory:

```
<dataset>/
  manifest.json    # dataset card (provenance)
  candidates.csv   # one row per candidate (the standard table)
  routes.jsonl     # one JSON synthesis route per candidate that has one (optional)
```

## `candidates.csv`

| group | column | meaning |
|---|---|---|
| **required** | `candidate_id` | stable id, unique within the dataset |
| | `smiles` | the proposed molecule (canonical preferred) |
| | `generator` | software that produced it (`rgfn`, `synflownet`, `vae_bo`, `fraggfn`, …) |
| **standard** | `step` | generation step / active-learning round (blank for one-shot generators) |
| | `score` | primary oracle objective for this candidate (blank if unscored) |
| | `oracle` | scorer behind `score` (e.g. `docking_6td3_gpu`) |
| | `valid` | 1 if RDKit parses the SMILES, else 0 |
| | `has_route` | 1 if a synthesis route is recorded in `routes.jsonl` |
| | `num_reactions` | synthesis length (route step count), blank if no route |
| **descriptors** | `mol_weight`,`qed`,`clogp`,`h_donors`,`h_acceptors`,`heavy_atoms`,`rotatable_bonds`,`tpsa`,`num_rings`,`lipinski_pass`,`ligand_efficiency` | from `glue/metrics/dataset_metrics.py`; written so a reader needn't depend on RDKit (recomputable from `smiles`) |
| **extra** | *(any)* | additional columns preserved verbatim — e.g. an oracle's per-pose breakdown (`vina_t2`, `cnnsc_t2`, `n_poses`, `status`) or a glanceable `route_str` |

`score` orientation is **not** encoded per row — it is a property of the oracle,
recorded once in the manifest as `score_higher_is_better` (docking ΔG /
differential is lower-is-better → `false`). The harness ranks accordingly.

## `routes.jsonl`

One JSON object per line (only for synthesizable generators); same structure as
`glue/active_learning/route.py`:

```json
{"candidate_id": "rgfn-000007", "product_smiles": "...", "num_reactions": 2,
 "building_block": {"smiles": "...", "idx": 123},
 "steps": [{"step": 1, "reaction_idx": 17, "reaction_smarts": "A.B>>C",
            "reactant": "...", "fragments": ["..."], "product": "..."}, ...]}
```

Non-synthesizable generators (VAE-BO, FragGFN) omit routes; `has_route` is 0 and
`routes.jsonl` may be absent.

## `manifest.json` (dataset card)

```json
{
  "schema_version": "1.0",
  "generator": "rgfn",
  "generator_version": null,
  "oracle": "docking_6td3_gpu",
  "score_higher_is_better": false,
  "score_units": null,
  "system": "6td3",
  "seed": 42,
  "budget": null,
  "n_candidates": 96,
  "n_valid": 96,
  "has_routes": true,
  "columns": ["candidate_id", "smiles", "generator", "step", "..."],
  "created": "2026-06-29T20:00:00+00:00",
  "source": "configs/glue/active_learning_6td3_gpu.gin",
  "notes": "..."
}
```

## Writing one

Our active-learning loop writes it automatically via `SuggestionLog` (the AL round
is the `step` column) to `<run>/active_learning/suggestions/`.

A new generator writes it directly:

```python
from glue.datasets.candidates import CandidateDataset

with CandidateDataset(out_dir, generator="vae_bo", oracle="docking_6td3_gpu",
                      system="6td3", seed=0, score_higher_is_better=False) as ds:
    for smi, score in results:
        ds.add(smiles=smi, score=score)            # + step=/route=/extra= as available
```

Or wrap an existing flat CSV of SMILES(+score):

```python
import csv
from glue.datasets.candidates import from_smiles_table

with open("vae_bo_raw.csv") as fh:
    from_smiles_table(csv.DictReader(fh), out_dir, generator="vae_bo",
                      smiles_col="SMILES", score_col="docking", oracle="docking_6td3_gpu",
                      system="6td3", score_higher_is_better=False)
```

## Reading one (harness)

```python
from glue.datasets.candidates import read_candidate_dataset, validate_candidate_dataset

issues = validate_candidate_dataset(path)        # [] == conformant
data = read_candidate_dataset(path)              # {"manifest", "candidates", "routes"}
```

## Conventions

- **Where datasets land:** generated candidate datasets under `data/synthetic/`
  (gitignored); active-learning runs alongside their run outputs on `$SCRATCH`;
  committed benchmark result tables under `validation/results/`.
- **Descriptors vs. evaluation metrics:** per-molecule descriptors and set
  diversity live in `glue/metrics/`. Evaluation-only metrics (recovery,
  anti-gaming correlations, SA distribution) live in `validation/harness/` and
  are computed *over* a candidate dataset — never written into it.
