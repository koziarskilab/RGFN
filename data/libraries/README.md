# data/libraries/ — standard fragment + reaction sets

Each subdirectory is one **chemistry library**: a building-block set + a
reaction-template set + optional cost annotation (per-block price, per-reaction
yield), in the canonical format specified by `docs/CHEM_LIBRARY_FORMAT.md`:

```
<name>/
├── fragments.csv     # smiles[, group, cost, catalog_id]
├── reactions.csv     # reaction[, family, comments, yield]
└── manifest.json     # provenance (name, version, source, counts, defaults, checksums)
```

A library is selected for a run by one gin binding
(`GlueReactionDataFactory.library_dir`), so `library` is an explicit benchmark axis
alongside generator / oracle / seed / budget.

**Planned libraries** (not built yet — see the doc's build order):

| name | contents |
|---|---|
| `chemistry_xlsx` | faithful copy of upstream `data/chemistry.xlsx` (no cost) — back-compat |
| `scent_small` | copy of SCENT's `data/small/` (chemistry **+** prices + yields) |
| `glue_standard_v1` | **the standard set** — `chemistry.xlsx` chemistry annotated with SCENT's prices + yields (matched by canonical SMILES/SMARTS) |

Small curated libraries are committed here; large/generated ones are git-ignored
with only their `manifest.json` retained for provenance.
