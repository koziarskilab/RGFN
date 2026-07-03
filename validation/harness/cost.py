"""Post-hoc synthesis-cost evaluation of a candidate dataset (retroactive pricing).

**Non-functional stub (2026-07-02).** Interfaces only; every function raises
``NotImplementedError``. See ``docs/CHEM_LIBRARY_FORMAT.md`` §6.

What this is
------------
An *evaluation-only* metric computed **over** a finished candidate dataset
(``docs/CANDIDATE_DATASET_FORMAT.md``) — never written into it, never in the
training loop. It is the cost sibling of ``validation/harness/synthesizability.py``
and follows the same "one standard on-disk format, one evaluator run uniformly on
every entrant" pattern.

It prices each generated molecule's *recorded synthesis route* (``routes.jsonl``,
produced by ``glue/active_learning/route.py`` for the synthesizable entrants:
RGFN, RxnFlow, SCENT) using a standard library's per-block prices and per-reaction
yields (``glue.chemistry.ChemLibrary``). This is the "retroactive cost" path from
``docs/CHEM_LIBRARY_FORMAT.md`` §2: SCENT consumes cost *natively* during
generation, everyone else is priced *after the fact* here — from the same numbers,
with the same formula, so all entrants land on one scale.

The pricing recursion is exactly SCENT's ``PathCostProxy._compute_costs``::

    route_cost = cost_of(building_block.smiles)
    for step in route.steps:                       # in order
        added = sum(cost_of(f) for f in step.fragments)
        route_cost = (route_cost + added) / yield_of(step.reaction_smarts)

No route (FragGFN, VAE-BO) ⇒ no by-construction cost, parallel to ``has_route=0``.

Dependency-light on purpose
---------------------------
Reads the candidate dataset directly (csv + jsonl + json) and imports only the
standard library + RDKit (for canonicalizing route SMILES/SMARTS against the cost
table). It does **not** import ``glue`` — importing anything under ``glue`` pulls in
``glue.registry`` -> torch/dgl (same reasoning as ``synthesizability.py``). It runs
in the ``rgfn`` env or any env with RDKit.

Run it
------
    python validation/harness/cost.py \
        --dataset data/synthetic/<run>/candidates \
        --library data/libraries/glue_standard_v1 \
        --top-k 16

Outputs (next to the dataset by default, or under ``--out``):
    cost.csv          one row per candidate (has_route, num_reactions, route_cost, fallbacks)
    cost_summary.json aggregate report (top-k mean/median cost, priced fraction, fallback rates)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MANIFEST_FILE = "manifest.json"
CANDIDATES_FILE = "candidates.csv"
ROUTES_FILE = "routes.jsonl"
PER_MOL_FILE = "cost.csv"
SUMMARY_FILE = "cost_summary.json"


# --- candidate-dataset + library I/O (inline; see docstring on not importing glue) ---


def read_candidates(dataset_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read ``manifest.json`` + ``candidates.csv`` with only the standard library."""
    raise NotImplementedError("stub: mirror validation/harness/synthesizability.read_candidates")


def read_routes(dataset_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Read ``routes.jsonl`` keyed by canonical product SMILES.

    Route schema is ``glue/active_learning/route.py::extract_route`` output:
    ``{product_smiles, num_reactions, building_block:{smiles,idx}, steps:[{...,
    reaction_smarts, fragments:[...], product}]}``.
    """
    raise NotImplementedError("stub: see glue/active_learning/route.py for the schema")


def load_cost_tables(
    library_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Return ``(fragment_to_cost, reaction_to_yield, defaults)`` from a canonical library.

    Reads ``fragments.csv`` / ``reactions.csv`` / ``manifest.json`` directly (does
    not import ``glue.chemistry``, to stay torch/dgl-free). ``defaults`` carries
    ``default_cost`` / ``default_yield`` for unmatched keys.
    """
    raise NotImplementedError("stub: see docs/CHEM_LIBRARY_FORMAT.md §3–4")


# --- pricing ------------------------------------------------------------------------


def price_route(
    route: Dict[str, Any],
    fragment_to_cost: Dict[str, float],
    reaction_to_yield: Dict[str, float],
    default_cost: float,
    default_yield: float,
) -> Dict[str, Any]:
    """Price one route with SCENT's recursion (see module docstring).

    Returns ``{route_cost, n_fallback_frags, n_fallback_rxns}``. Canonicalizes each
    fragment SMILES and reaction SMARTS before lookup so keys agree with the cost
    table; unmatched keys use the defaults and increment the fallback counters.
    """
    raise NotImplementedError("stub: implement the PathCostProxy recursion")


def summarize(rows: List[Dict[str, Any]], top_k: Optional[int]) -> Dict[str, Any]:
    """Aggregate per-candidate costs into the report.

    Headline numbers: top-k mean/median ``route_cost`` (the "average synthesis cost
    of the top-k" axis from ``Logs/017``), fraction of candidates priced (had a
    route), and the fallback rates (coverage of the cost table over these routes).
    """
    raise NotImplementedError("stub: mirror validation/harness/synthesizability.summarize")


# --- CLI ----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, help="candidate dataset dir (manifest+csv+routes)"
    )
    parser.add_argument("--library", required=True, help="data/libraries/<name>/ for prices+yields")
    parser.add_argument("--out", default=None, help="output dir (default: next to --dataset)")
    parser.add_argument("--top-k", type=int, default=16, help="top-k for the headline cost numbers")
    parser.parse_args(argv)
    raise NotImplementedError("stub: wire read -> price_route -> summarize -> write cost.csv/json")


if __name__ == "__main__":
    raise SystemExit(main())
