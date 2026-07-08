# Chemistry library format — a standard, swappable fragment + reaction set

**Status:** design + non-functional stubs (2026-07-02). Nothing here is wired into a
run yet; the stubs (`glue/chemistry/`, `validation/harness/cost.py`) raise
`NotImplementedError` and exist to fix the interfaces. See "Build order" at the end.

This document specifies a **canonical, modular library** of the two inputs every
reaction-based generator in this project builds molecules from — a **fragment /
building-block set** and a **reaction-template set** — plus an optional **cost
annotation** (per-block price, per-reaction yield). The goal is a *single standard
set* that RGFN, RxnFlow, and SCENT can all consume, so a benchmark run varies the
**generator** while holding the **chemistry** fixed.

Read `docs/ARCHITECTURE.md` (upstream-vs-ours, the one-way dependency arrow) and
`CLAUDE.md` before implementing. This lives entirely in *our* tree (`glue/`,
`validation/`, `data/libraries/`); it never edits `rgfn/`.

---

## 1. Why this exists

Today every reaction generator uses a **different** library, which is a confound in
the head-to-head benchmark (`Logs/016`, `Logs/017`; the SCENT README says so
outright):

| Generator | Library today | Cost-aware? |
|---|---|---|
| RGFN   | `data/chemistry.xlsx` (~350 frags, 64–66 SMARTS templates) | no |
| SCENT  | `external/scent/data/small/` (418 frags **+ prices**, ~123 templates **+ yields**) | **yes, natively** |
| RxnFlow| ZINCFrag blocks + `real.txt` / `hb_edited.txt` templates | no |
| FragGFN| Recursion `gflownet` atom-fragment vocab (no reactions) | n/a (non-synth foil) |

Two facts make a standard set cheap to build:

1. **The chemistry is already shared.** SCENT's `data/small/` is essentially
   `chemistry.xlsx`'s chemistry — the *same* SMARTS reaction families and heavily
   overlapping fragment SMILES (e.g. SCENT's `Fc1ncc(Br)cn1` is `chemistry.xlsx`'s
   `FC1=NC=C(Br)C=N1` after canonicalization) — because SCENT is a fork of RGFN
   from the same lab. SCENT just *adds* two annotation files:
   `fragment_to_real_cost.json` (per-block price, \$0.001–\$196) and
   `templates_yields.csv` (per-family yield, e.g. 0.75).
2. **There is exactly one place chemistry enters RGFN.**
   `rgfn/gfns/reaction_gfn/api/reaction_data_factory.py::ReactionDataFactory`
   (`@gin.configurable`) reads the workbook and hands `get_fragments()` /
   `get_anchored_reactions()` to `ReactionEnv`. Fragments/reactions define the
   *action space*, not the reward. A "standard set" is precisely *what this factory
   loads.*

**Decision (2026-07-02):** the first standard set is a **faithful repackaging** —
`chemistry.xlsx`'s fragments + reactions, annotated with **SCENT's existing prices
and yields**. No new chemistry, no new cost sourcing. This isolates the generator
algorithm in the benchmark and unblocks the cost axis with data already in-repo.

---

## 2. Two axes, kept separate

- **Library = an input** (fragments + reactions). Standardized, swappable, selected
  by one gin knob.
- **Cost = an annotation on the library** (block price + reaction yield). Used two
  ways:
  - **natively** by SCENT during generation (its `PathCostProxy` +
    cost-guided backward policy steer toward cheap routes);
  - **retroactively** by everyone else — a post-hoc evaluator prices each generated
    molecule's recorded route with the *same* numbers and the *same* formula.

"Retroactive cost for everything except SCENT" is not a special case; it is the
direct consequence of cost being an annotation. Any generator that emits a route
(`routes.jsonl`, see `glue/active_learning/route.py`) can be priced after the fact.

---

## 3. Canonical on-disk format

A library is a directory `data/libraries/<name>/` with three files. `data/` is our
inputs dir (`CLAUDE.md`); small curated libraries are committed, large/generated
ones are git-ignored under `data/libraries/` with only their `manifest.json` kept.

```
data/libraries/<name>/
├── fragments.csv     # building blocks
├── reactions.csv     # reaction templates
└── manifest.json     # provenance
```

### `fragments.csv`
| column | required | meaning |
|---|---|---|
| `smiles` | ✅ | building-block SMILES (canonicalized on load) |
| `group` | – | family tag (upstream's `Group` column; metadata only) |
| `cost` | – | price per unit (SCENT units; unitless float). Missing → default cost. |
| `catalog_id` | – | provenance (Enamine/Mcule/ZINC id), for later real-catalog work |

`smiles` alone is the exact input the upstream `ReactionDataFactory.fragment_path`
CSV already accepts (`SMILES` column) — so this format is a strict superset.

### `reactions.csv`
| column | required | meaning |
|---|---|---|
| `reaction` | ✅ | reaction SMARTS, `reactants >> products`, reactants split on `.` |
| `family` | – | upstream's `Family` column (metadata) |
| `comments` | – | upstream's `Comments` column (metadata) |
| `yield` | – | reaction yield in (0, 1]. Missing → default yield. |

Columns mirror the upstream workbook's `Reactions_*` sheet (`Family`, `Reaction`,
`Comments`) plus SCENT's `yield` — i.e. `templates_yields.csv` *is* this file.

### `manifest.json`
```json
{
  "name": "glue_standard_v1",
  "version": "1.0",
  "source": "chemistry.xlsx (Reactions/Fragments) + scent/data/small cost+yield",
  "created": "2026-07-02",
  "n_fragments": 350, "n_reactions": 66,
  "n_fragments_priced": 331, "n_reactions_with_yield": 60,
  "cost_units": "scent_small (unitless, relative)",
  "default_cost": 1.0, "default_yield": 0.5,
  "fragments_sha256": "...", "reactions_sha256": "..."
}
```
Provenance is load-bearing for the release-ready objective (`RESEARCH_CONTEXT.md`
Obj 6): every result traces to a library name+version+checksum.

---

## 4. The provider: `glue.chemistry.ChemLibrary`

`glue/chemistry/library.py` — the single source of truth. One in-memory object with
loaders (many formats in) and exporters (each generator's native format out).

```
ChemLibrary
  .fragments : list[FragmentSpec(smiles, cost, group, catalog_id)]
  .reactions : list[ReactionSpec(smarts, yield_, family, comments)]
  .manifest  : dict

  # loaders (all normalize to the same object)
  ChemLibrary.from_canonical_dir(path)                 # the format in §3
  ChemLibrary.from_chemistry_xlsx(path, docking=False) # upstream workbook
  ChemLibrary.from_scent_small(path)                   # SCENT's txt+json+csv

  # exporters (feed the SAME chemistry to each generator, in its own format)
  .to_canonical_dir(path)
  .to_scent_format(outdir)   # fragments.txt, templates.txt,
                             #   fragment_to_real_cost.json, templates_yields.csv
  .to_rxnflow_format(outdir) # building-block file + template file
  .to_fragment_csv(path)     # SMILES column for ReactionDataFactory.fragment_path

  # cost tables (shared by native + retroactive pricing)
  .cost_of(smiles) -> float          # canonicalized lookup, default_cost fallback
  .yield_of(smarts) -> float         # canonicalized lookup, default_yield fallback
  .fragment_to_cost() -> dict
  .reaction_to_yield() -> dict
```

Canonicalization matters: routes and cost tables must agree on SMILES/SMARTS keys,
so every lookup canonicalizes both sides (RDKit `MolToSmiles`; reactions via a
stable parse of both sides of `>>`). Unmatched keys fall back to
`manifest.default_cost` / `default_yield`, and the count of matched-vs-fallback is
recorded in the manifest so coverage is never silent.

---

## 5. Wiring each generator to the standard set

### RGFN (native — no upstream edit)
`glue/chemistry/reaction_data_factory.py::GlueReactionDataFactory(ReactionDataFactory)`
— a subclass that reads a `ChemLibrary` and builds the *same* attributes upstream
builds (`reactions`, `disconnections`, `anchored_reactions`,
`reaction_anchor_map`, `anchored_disconnections`, `fragments`) using the *same*
upstream classes (`Reaction`, `Molecule`, `AnchoredReaction`). Action-space
semantics are therefore byte-identical to today; only the *source* of the lists
changes. Registered via `glue/registry.py`, selected in `configs/glue/`:

```gin
# configs/glue/lib_standard.gin  (include from any glue run config)
include 'configs/rgfn_base.gin'
data_factory/gin.singleton.constructor = @GlueReactionDataFactory
GlueReactionDataFactory.library_dir = 'data/libraries/glue_standard_v1'
GlueReactionDataFactory.docking     = True
train_env/ReactionEnv.data_factory  = @data_factory/gin.singleton()
valid_env/ReactionEnv.data_factory  = @data_factory/gin.singleton()
```

(Contrast: upstream's `configs/envs/reaction.gin` binds
`ReactionDataFactory.reaction_path = 'data/chemistry.xlsx'`. We overlay, never edit
it — `configs/glue/README.md`.)

### SCENT (native cost)
Export the standard library into SCENT's format and point SCENT's
`configs/envs/settings/*.gin` at it:

```
ChemLibrary.from_canonical_dir('data/libraries/glue_standard_v1')
           .to_scent_format('<run>/scent_lib/')
# then in the scent gin:  cost_path=<...>/fragment_to_real_cost.json
#                         yield_path=<...>/templates_yields.csv
```

SCENT then runs its full cost-aware method on the **same chemistry and the same
costs** as everyone else — closing the confound its own README flags. Nothing new
to build here; `to_scent_format` is the only glue.

### RxnFlow
`to_rxnflow_format` emits a building-block file + template file. ⚠️ **Known
constraint** (`docs/REFACTOR_LOG.md`): template count and library size interact —
too few templates over a large block set drove RxnFlow non-finite (71 templates on
the 10k debug set failed; 109 Enamine-REAL templates worked). The standard set's
~66 reactions may be too sparse for RxnFlow; if so, RxnFlow keeps a template
superset and we document the deviation rather than forcing an unstable run.

### FragGFN
Non-synthesizable foil; its atom-fragment action space has no reaction templates,
so it does **not** consume the reaction library. Optionally seed its fragment vocab
from the same building blocks for fairness, but it stays out of the reaction-library
standardization by design.

---

## 6. Cost — the retroactive evaluator

`validation/harness/cost.py` — a post-hoc metric over a finished candidate dataset,
a direct sibling of `validation/harness/synthesizability.py` (same "one standard
on-disk format, one evaluator run uniformly on every entrant" pattern). It is
**evaluation-only**: never in the training loop, never written into the dataset.

The pricing recursion is exactly SCENT's `PathCostProxy._compute_costs`:

```
route_cost = cost_of(building_block.smiles)
for step in route.steps:                       # in order
    added = sum(cost_of(f) for f in step.fragments)
    route_cost = (route_cost + added) / yield_of(step.reaction_smarts)
```

Lower yield ⇒ higher effective cost; each added fragment adds its price. This is
byte-for-byte the accumulation SCENT uses internally, so SCENT's *own* cost numbers
and our retroactive numbers for RGFN/RxnFlow are on one scale.

**I/O** (mirrors `synthesizability.py`): read `manifest.json` + `candidates.csv` +
`routes.jsonl` from a candidate dataset, load cost/yield from a `--library` dir,
price each candidate with a route, write:
- `cost.csv` — one row per candidate: `smiles, has_route, num_reactions, route_cost, n_fallback_frags, n_fallback_rxns`
- `cost_summary.json` — top-k mean/median cost, fraction priced, fallback rates.

**No route (FragGFN, VAE-BO):** no route ⇒ no by-construction cost (parallel to
`has_route=0`). Optional later extension: price an AiZynth-found route instead, for
an apples-to-apples number on non-synthesizable entrants.

**Dependency-light:** stdlib + RDKit only (for canonicalization), reads the
candidate CSV/JSON directly rather than importing `glue` — same reasoning as
`synthesizability.py` (importing `glue` pulls torch/dgl). Runs in the `rgfn` env or
any env with RDKit.

Run:
```
python validation/harness/cost.py \
    --dataset data/synthetic/<run>/candidates \
    --library data/libraries/glue_standard_v1 \
    --top-k 16
```

---

## 7. Modularity payoff

A benchmark run becomes parameterized by **(generator, library, oracle, seed,
budget)** — `library` is now an explicit controlled axis via one gin binding
(`GlueReactionDataFactory.library_dir`). Candidate library directories:
`chemistry_xlsx` (back-compat), `scent_small`, `glue_standard_v1`, later
`enamine_real`. This directly unlocks currently-open objectives:

- **Obj 2** — "*contribution of the building-block set*" ablation (swap `library`,
  hold everything else).
- **Obj 5** — a **synthesis-cost axis** alongside the AiZynth synthesizability
  report; the `Logs/017` follow-up ("price RGFN/FragGFN routes with SCENT's cost
  model") becomes a one-command post-hoc step.
- **Obj 6** — every result carries a library name+version+checksum.

---

## 8. Build order (later, on approval)

1. `glue/chemistry/library.py` — implement `ChemLibrary` loaders/exporters/cost maps.
2. Materialize `data/libraries/glue_standard_v1/` by
   `from_chemistry_xlsx(docking=True)` merged with `from_scent_small()` costs
   (match by canonical SMILES / SMARTS; record coverage in the manifest).
3. `glue/chemistry/reaction_data_factory.py` — implement `GlueReactionDataFactory`;
   add `configs/glue/lib_standard.gin`; smoke-test that
   "N fragments, M reactions, K anchored reactions" matches the xlsx run.
4. `validation/harness/cost.py` — implement the retroactive evaluator + tests
   (mirror `validation/harness/test_synthesizability.py`).
5. `to_scent_format` / `to_rxnflow_format` exporters + baseline configs pointing at
   the exported standard library; note any RxnFlow template-sparsity deviation.
6. Log the whole thing as an experiment (`experiment-log` skill).
```
