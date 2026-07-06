"""Post-hoc synthesis-cost evaluation of a candidate dataset (retroactive pricing).

See ``docs/CHEM_LIBRARY_FORMAT.md`` §6.

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
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

MANIFEST_FILE = "manifest.json"
CANDIDATES_FILE = "candidates.csv"
ROUTES_FILE = "routes.jsonl"
PER_MOL_FILE = "cost.csv"
SUMMARY_FILE = "cost_summary.json"

DEFAULT_COST = 1.0
DEFAULT_YIELD = 0.5


# --- canonicalization (inline rdkit; kept glue-free so this runs in any rdkit env) ---


def _canonical_smiles(smiles: str) -> Optional[str]:
    """RDKit canonical SMILES, or None if unparseable. Mirrors glue.chemistry.library."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def _canonical_reaction(smarts: str) -> str:
    """Reactant-order-invariant reaction key.

    Route records store the *anchored* reaction (RGFN/SCENT move the anchor reactant
    to the front), so a template ``A.B >> C`` is written ``B.A >> C`` in ~half of
    routes. Keying on the whitespace-normalized string alone therefore misses ~45%
    of reactions. We sort the ``.``-separated components on each side (``.`` is the
    top-level component separator in SMARTS, exactly how upstream ``Reaction`` splits
    reactants), which makes the key invariant to reactant order and recovers 100%
    coverage against the library's ``reactions.csv``. Applied to BOTH sides of the
    lookup (library load + route pricing), so they always agree.
    """
    parts = smarts.split(">>")
    if len(parts) != 2:
        return " >> ".join(p.strip() for p in parts)
    left, right = parts
    norm = lambda side: ".".join(sorted(p.strip() for p in side.strip().split(".")))
    return f"{norm(left)} >> {norm(right)}"


# --- candidate-dataset + library I/O (inline; see docstring on not importing glue) ---


def read_candidates(dataset_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read ``manifest.json`` + ``candidates.csv`` with only the standard library."""
    dataset_dir = Path(dataset_dir)
    manifest: Dict[str, Any] = {}
    mpath = dataset_dir / MANIFEST_FILE
    if mpath.exists():
        with open(mpath) as fh:
            manifest = json.load(fh)
    cpath = dataset_dir / CANDIDATES_FILE
    if not cpath.exists():
        raise FileNotFoundError(f"no {CANDIDATES_FILE} in {dataset_dir}")
    with open(cpath, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return manifest, rows


def read_routes(dataset_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Read ``routes.jsonl`` keyed by canonical product SMILES (``{}`` if absent).

    Route schema is ``glue/active_learning/route.py::extract_route`` output:
    ``{product_smiles, num_reactions, building_block:{smiles,idx}, steps:[{...,
    reaction_smarts, fragments:[...], product}]}``.
    """
    routes: Dict[str, Dict[str, Any]] = {}
    p = Path(dataset_dir) / ROUTES_FILE
    if not p.exists():
        return routes
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = _canonical_smiles(rec.get("product_smiles", "")) or rec.get("product_smiles", "")
            if key:
                routes[key] = rec
    return routes


def load_cost_tables(
    library_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Return ``(fragment_to_cost, reaction_to_yield, defaults)`` from a canonical library.

    Reads ``fragments.csv`` / ``reactions.csv`` / ``manifest.json`` directly (does
    not import ``glue.chemistry``, to stay torch/dgl-free). Keys are canonicalized
    the same way the route SMILES/SMARTS will be, so lookups agree. ``defaults``
    carries ``default_cost`` / ``default_yield`` for unmatched keys.
    """
    library_dir = Path(library_dir)
    manifest: Dict[str, Any] = {}
    if (library_dir / MANIFEST_FILE).exists():
        with open(library_dir / MANIFEST_FILE) as fh:
            manifest = json.load(fh)

    fragment_to_cost: Dict[str, float] = {}
    with open(library_dir / "fragments.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            cost = row.get("cost", "")
            if cost in ("", None):
                continue
            canon = _canonical_smiles(row["smiles"].strip())
            if canon is not None:
                fragment_to_cost[canon] = float(cost)

    reaction_to_yield: Dict[str, float] = {}
    with open(library_dir / "reactions.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            y = row.get("yield", "")
            if y in ("", None):
                continue
            reaction_to_yield[_canonical_reaction(row["reaction"].strip())] = float(y)

    defaults = {
        "default_cost": float(manifest.get("default_cost", DEFAULT_COST)),
        "default_yield": float(manifest.get("default_yield", DEFAULT_YIELD)),
    }
    return fragment_to_cost, reaction_to_yield, defaults


# --- pricing ------------------------------------------------------------------------


def price_route(
    route: Dict[str, Any],
    fragment_to_cost: Dict[str, float],
    reaction_to_yield: Dict[str, float],
    default_cost: float,
    default_yield: float,
) -> Dict[str, Any]:
    """Price one route with SCENT's ``PathCostProxy`` recursion (see module docstring).

    Returns ``{route_cost, n_fallback_frags, n_fallback_rxns, n_steps}``. Canonicalizes
    each fragment SMILES and reaction SMARTS before lookup so keys agree with the cost
    table; unmatched keys use the defaults and increment the fallback counters.
    """
    n_fallback_frags = 0
    n_fallback_rxns = 0

    def cost_of(smiles: str) -> float:
        nonlocal n_fallback_frags
        canon = _canonical_smiles(smiles)
        if canon is not None and canon in fragment_to_cost:
            return fragment_to_cost[canon]
        n_fallback_frags += 1
        return default_cost

    def yield_of(smarts: str) -> float:
        nonlocal n_fallback_rxns
        key = _canonical_reaction(smarts)
        if key in reaction_to_yield:
            return reaction_to_yield[key]
        n_fallback_rxns += 1
        return default_yield

    bb = route.get("building_block") or {}
    route_cost = cost_of(bb.get("smiles", ""))
    steps = route.get("steps") or []
    for step in steps:
        added = sum(cost_of(f) for f in (step.get("fragments") or []))
        route_cost = (route_cost + added) / yield_of(step.get("reaction_smarts", ""))

    return {
        "route_cost": route_cost,
        "n_fallback_frags": n_fallback_frags,
        "n_fallback_rxns": n_fallback_rxns,
        "n_steps": len(steps),
    }


def summarize(rows: List[Dict[str, Any]], top_k: Optional[int]) -> Dict[str, Any]:
    """Aggregate per-candidate costs into the report.

    ``rows`` are the per-candidate dicts written to ``cost.csv`` (each carries
    ``score``, ``has_route``, ``route_cost`` or None, and the fallback counts).
    Headline numbers: top-k mean/median ``route_cost`` (the "average synthesis cost
    of the top-k" axis from ``Logs/017``), fraction of candidates priced (had a
    route), and the fallback rates (cost-table coverage over these routes).
    """
    priced = [r for r in rows if r.get("route_cost") is not None]
    costs = [r["route_cost"] for r in priced]

    # Top-k by generator score (higher-is-better for sEH; costs of the best molecules).
    scored = [r for r in priced if r.get("score") is not None]
    scored.sort(key=lambda r: r["score"], reverse=True)
    topk = scored[:top_k] if top_k else scored
    topk_costs = [r["route_cost"] for r in topk]

    def stats(xs: List[float]) -> Dict[str, Optional[float]]:
        if not xs:
            return {"n": 0, "median": None, "mean": None, "min": None, "max": None}
        return {
            "n": len(xs),
            "median": median(xs),
            "mean": mean(xs),
            "min": min(xs),
            "max": max(xs),
        }

    any_fallback = sum(
        1 for r in priced if (r.get("n_fallback_frags", 0) or r.get("n_fallback_rxns", 0))
    )
    return {
        "n_total": len(rows),
        "n_priced": len(priced),
        "priced_fraction": (len(priced) / len(rows)) if rows else 0.0,
        "route_cost_all": stats(costs),
        "route_cost_top_k": stats(topk_costs),
        "top_k": top_k,
        "route_len_mean": (mean([r["n_steps"] for r in priced]) if priced else None),
        "fallback": {
            "frags_mean": (mean([r["n_fallback_frags"] for r in priced]) if priced else None),
            "rxns_mean": (mean([r["n_fallback_rxns"] for r in priced]) if priced else None),
            "any_fallback_fraction": (any_fallback / len(priced)) if priced else None,
        },
    }


# --- CLI ----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, help="candidate dataset dir (manifest+csv+routes)"
    )
    parser.add_argument("--library", required=True, help="data/libraries/<name>/ for prices+yields")
    parser.add_argument("--out", default=None, help="output dir (default: next to --dataset)")
    parser.add_argument(
        "--top-k", type=int, default=100, help="top-k for the headline cost numbers"
    )
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset)
    out_dir = Path(args.out) if args.out else dataset_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, candidates = read_candidates(dataset_dir)
    routes = read_routes(dataset_dir)
    frag_cost, rxn_yield, defaults = load_cost_tables(Path(args.library))
    higher_better = bool(manifest.get("score_higher_is_better", True))

    out_rows: List[Dict[str, Any]] = []
    for c in candidates:
        smi = c.get("smiles", "")
        canon = _canonical_smiles(smi) or smi
        route = routes.get(canon)
        try:
            score = float(c["score"]) if c.get("score") not in (None, "") else None
        except (TypeError, ValueError):
            score = None
        # sort key: higher-is-better -> use score as-is; else negate so top_k picks best
        sort_score = score if (score is None or higher_better) else -score
        row: Dict[str, Any] = {
            "smiles": canon,
            "score": sort_score,
            "raw_score": score,
            "has_route": 1 if route else 0,
            "route_cost": None,
            "n_fallback_frags": "",
            "n_fallback_rxns": "",
            "n_steps": "",
        }
        if route:
            priced = price_route(
                route, frag_cost, rxn_yield, defaults["default_cost"], defaults["default_yield"]
            )
            row.update(priced)
        out_rows.append(row)

    with open(out_dir / PER_MOL_FILE, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "smiles",
                "raw_score",
                "has_route",
                "route_cost",
                "n_steps",
                "n_fallback_frags",
                "n_fallback_rxns",
            ],
        )
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    report = summarize(out_rows, args.top_k)
    report["dataset"] = str(dataset_dir)
    report["library"] = str(args.library)
    report["generator"] = manifest.get("generator")
    with open(out_dir / SUMMARY_FILE, "w") as fh:
        json.dump(report, fh, indent=2)

    rc = report["route_cost_top_k"]
    print(
        f"[cost] {manifest.get('generator', dataset_dir.name)}: priced "
        f"{report['n_priced']}/{report['n_total']} | top-{args.top_k} cost "
        f"median={rc['median']} mean={rc['mean']} | route_len~{report['route_len_mean']} "
        f"| any-fallback={report['fallback']['any_fallback_fraction']}"
    )
    print(f"[cost] wrote {out_dir/PER_MOL_FILE} + {out_dir/SUMMARY_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
