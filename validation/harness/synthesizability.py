"""Post-hoc synthesizability evaluation of a candidate dataset (AiZynthFinder + SA).

What this is
------------
An *evaluation-only* metric, computed **over** a finished candidate dataset
(``docs/CANDIDATE_DATASET_FORMAT.md``) — never written into it, never in the
training loop. ``docs/CANDIDATE_DATASET_FORMAT.md`` and ``validation/harness/
README.md`` place "SA distribution"-style synthesizability metrics here, in the
harness, by design. This module implements exactly the synthesizability report
the source papers publish, run uniformly on every entrant (RGFN, RxnFlow,
FragGFN, SynFlowNet, VAE-BO) because they all emit the one standard format.

What it reports (faithful to the papers)
----------------------------------------
For each unique, valid molecule we run a retrosynthetic search with
**AiZynthFinder** (`[genheden2020aizynth]`) over the standard public dataset
(USPTO expansion templates + ZINC in-stock building blocks) and record whether a
full route to in-stock precursors was found. Aggregated, the headline number is:

  - ``aizynth_success_rate``  -- fraction of molecules AiZynth solved. This is the
    ``AiZynth`` column in `[koziarski2024rgfn]` Table 1 (RGFN ~0.56) and
    `[gainski2025scent]` Table 1 (up to ~0.75), and RxnFlow's "Synthesizability %"
    (`[seo2024rxnflow]`).
  - ``steps_mean`` / ``steps_median``  -- route length over solved molecules
    ("synthetic complexity" / number of synthesis steps, `[seo2024rxnflow]`).
  - ``sa_*``  -- Ertl & Schuffenhauer SA-Score distribution (RDKit contrib
    ``sascorer``), the cheap companion every paper also reports.

We additionally cross-check the generator's *self-reported* route claim
(``has_route`` / ``num_reactions`` columns, which RGFN/RxnFlow set by
construction) against AiZynth's *independent* verdict — the by-construction
guarantee vs. an external retrosynthesis tool.

Why it lives here and runs as a standalone bridge
-------------------------------------------------
AiZynthFinder pins its own heavy stack (ONNX, large template models) and is
installed in a **dedicated ``aizynth`` conda env** (``external/setup_aizynthfinder.sh``),
exactly the "one env per benchmarked tool, one shared on-disk standard" model the
generator adapters already use with ``scripts/score_batch.py``. So this file is
written to run *inside* that env:

  - it imports only ``rdkit`` + ``aizynthfinder`` (both present in the ``aizynth``
    env) and the standard library;
  - it reads the candidate dataset **directly** (csv + json) rather than importing
    ``glue.datasets.candidates`` -- importing anything under ``glue`` would pull in
    ``glue.registry`` -> torch/dgl, which the ``aizynth`` env does not have. The
    candidate format is trivial to read, so we mirror it here (the canonical
    writer/reader still lives in ``glue/datasets/candidates.py``).

Run it
------
    conda run -n aizynth python validation/harness/synthesizability.py \
        --dataset data/synthetic/<run>/candidates \
        --config  data/models/aizynthfinder/config.yml \
        --nproc 8

Outputs (next to the dataset by default, or under ``--out``):
    synthesizability.csv          one row per candidate (solved, n_steps, sa_score, ...)
    synthesizability_summary.json the aggregate report (the numbers for a results table)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Tuple

# --- candidate-dataset I/O (inline; see module docstring on why we don't import glue) ---

MANIFEST_FILE = "manifest.json"
CANDIDATES_FILE = "candidates.csv"
PER_MOL_FILE = "synthesizability.csv"
SUMMARY_FILE = "synthesizability_summary.json"


def read_candidates(dataset_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read ``manifest.json`` + ``candidates.csv`` with only the standard library.

    Returns ``(manifest, rows)``. Kept deliberately minimal so this evaluator runs
    in the ``aizynth`` env without importing ``glue`` (which would pull torch/dgl).
    The canonical reader is ``glue.datasets.candidates.read_candidate_dataset``.
    """
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


# --- RDKit helpers: canonicalization + SA score --------------------------------------


_RDKIT_QUIET = False


def _quiet_rdkit() -> None:
    """Suppress RDKit's per-molecule parse warnings (we handle None mols ourselves)."""
    global _RDKIT_QUIET
    if not _RDKIT_QUIET:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        _RDKIT_QUIET = True


def _canonical(smiles: str) -> Optional[str]:
    from rdkit import Chem

    _quiet_rdkit()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


_SASCORER = None


def _sa_scorer():
    """Lazily import RDKit's contrib SA-Score module (Ertl & Schuffenhauer 2009)."""
    global _SASCORER
    if _SASCORER is None:
        from rdkit.Chem import RDConfig

        sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_dir not in sys.path:
            sys.path.append(sa_dir)
        import sascorer  # type: ignore

        _SASCORER = sascorer
    return _SASCORER


def sa_score(smiles: str) -> Optional[float]:
    """Synthetic Accessibility score (1 = easy ... 10 = hard), or None on parse fail."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    try:
        return float(_sa_scorer().calculateScore(mol))
    except Exception:
        return None


# --- AiZynthFinder driver (parallel over molecules) ----------------------------------

# One AiZynthFinder per worker process; building it loads the (large) template model,
# so we amortize that over a chunk of molecules instead of per call.
_FINDER = None
_FINDER_OPTS: Dict[str, Any] = {}


def _init_finder(
    config: str,
    stock: str,
    expansion: str,
    filter_policy: Optional[str],
    time_limit: Optional[int],
    iteration_limit: Optional[int],
) -> None:
    """multiprocessing.Pool initializer: build + configure one finder per worker."""
    global _FINDER, _FINDER_OPTS
    from aizynthfinder.aizynthfinder import AiZynthFinder

    finder = AiZynthFinder(configfile=config)

    def _select(component, requested, kind):
        """Select ``requested`` key if present, else the first available, else skip."""
        try:
            available = list(component.items)  # configured keys
        except Exception:
            available = []
        key = requested if requested in available else (available[0] if available else None)
        if key is not None:
            component.select(key)
        return key

    _FINDER_OPTS = {
        "stock": _select(finder.stock, stock, "stock"),
        "expansion": _select(finder.expansion_policy, expansion, "expansion"),
        "filter": _select(finder.filter_policy, filter_policy, "filter") if filter_policy else None,
    }
    # Best-effort search-budget overrides (pydantic config in aizynthfinder>=4).
    for attr, val in (("time_limit", time_limit), ("iteration_limit", iteration_limit)):
        if val is not None:
            try:
                setattr(finder.config.search, attr, val)
            except Exception:
                pass
    _FINDER = finder


def _search_one(smiles: str) -> Dict[str, Any]:
    """Run one retrosynthesis. Returns a flat dict of the stats we report."""
    out: Dict[str, Any] = {
        "solved": 0,
        "n_steps": None,
        "n_solved_routes": None,
        "top_score": None,
        "search_time": None,
        "error": None,
    }
    try:
        _FINDER.target_smiles = smiles
        t0 = time.perf_counter()
        _FINDER.tree_search()
        _FINDER.build_routes()
        out["search_time"] = round(time.perf_counter() - t0, 3)
        stats = _FINDER.extract_statistics()
        out["solved"] = int(bool(stats.get("is_solved")))
        # number_of_steps = length of the top-ranked route (only meaningful if solved).
        steps = stats.get("number_of_steps")
        out["n_steps"] = int(steps) if (out["solved"] and steps is not None) else None
        out["n_solved_routes"] = stats.get("number_of_solved_routes")
        out["top_score"] = stats.get("top_score")
    except Exception as exc:  # never let one bad molecule kill the batch
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def run_aizynth(
    smiles_list: List[str],
    *,
    config: str,
    stock: str,
    expansion: str,
    filter_policy: Optional[str],
    time_limit: Optional[int],
    iteration_limit: Optional[int],
    nproc: int,
) -> List[Dict[str, Any]]:
    """Retrosynthesis over a list of SMILES; one result dict per input (order preserved)."""
    init_args = (config, stock, expansion, filter_policy, time_limit, iteration_limit)
    if nproc <= 1:
        _init_finder(*init_args)
        return [_search_one(s) for s in smiles_list]

    import multiprocessing as mp

    with mp.Pool(processes=nproc, initializer=_init_finder, initargs=init_args) as pool:
        # chunksize 1: searches are long and uneven, so balance dynamically.
        return pool.map(_search_one, smiles_list, chunksize=1)


# --- aggregation ----------------------------------------------------------------------


def _summ(values: List[float], prefix: str) -> Dict[str, float]:
    vals = [v for v in values if v is not None and v == v]
    if not vals:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": mean(vals),
        f"{prefix}_std": pstdev(vals) if len(vals) > 1 else 0.0,
        f"{prefix}_median": median(vals),
        f"{prefix}_min": min(vals),
        f"{prefix}_max": max(vals),
    }


def summarize(
    manifest: Dict[str, Any], per_mol: List[Dict[str, Any]], opts: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the aggregate synthesizability report from per-molecule rows.

    Headline numbers (success rate, steps, SA) are aggregated over UNIQUE molecules
    (one representative per canonical SMILES) — a molecule is solved or not, and the
    papers report over the unique set, so duplicate rows must not double-count.
    Row-level counts (``n_candidates``, the self-reported route claim) stay per-row.
    """
    valid = [r for r in per_mol if r["valid"]]

    # Collapse to one record per canonical molecule for the headline metrics.
    seen: Dict[str, Dict[str, Any]] = {}
    for r in valid:
        seen.setdefault(r["canonical"], r)
    unique_valid = list(seen.values())
    evaluated = [r for r in unique_valid if r.get("evaluated")]
    solved = [r for r in evaluated if r["solved"]]

    n_eval = len(evaluated)
    success = (len(solved) / n_eval) if n_eval else float("nan")

    # Generator's own route claim (RGFN/RxnFlow set this by construction; FragGFN/VAE-BO don't).
    n_self_route = sum(1 for r in per_mol if str(r.get("has_route") or "") in ("1", "1.0", "True"))
    self_steps = [
        r["self_num_reactions"] for r in per_mol if r.get("self_num_reactions") is not None
    ]

    summary: Dict[str, Any] = {
        "generator": manifest.get("generator"),
        "system": manifest.get("system"),
        "oracle": manifest.get("oracle"),
        "seed": manifest.get("seed"),
        "n_candidates": len(per_mol),
        "n_valid": len(valid),
        "n_unique_valid": len(unique_valid),
        "n_evaluated": n_eval,  # unique valid mols actually run through AiZynth
        "n_solved": len(solved),
        "aizynth_success_rate": success,  # the headline number (RGFN/SCENT "AiZynth")
        "steps_mean": (mean([r["n_steps"] for r in solved]) if solved else float("nan")),
        "steps_median": (median([r["n_steps"] for r in solved]) if solved else float("nan")),
        "n_errors": sum(1 for r in evaluated if r.get("error")),
        # cheap SA companion, over unique valid molecules
        **_summ([r["sa_score"] for r in unique_valid], "sa"),
        # independent vs. by-construction cross-check
        "self_reported_route_rate": (n_self_route / len(per_mol) if per_mol else float("nan")),
        "self_reported_steps_mean": (mean(self_steps) if self_steps else float("nan")),
        "config": opts,
    }
    return summary


# --- orchestration --------------------------------------------------------------------


def evaluate_dataset(
    dataset_dir: Path,
    *,
    out_dir: Optional[Path] = None,
    config: str,
    stock: str = "zinc",
    expansion: str = "uspto",
    filter_policy: Optional[str] = "uspto",
    time_limit: Optional[int] = None,
    iteration_limit: Optional[int] = None,
    nproc: int = 1,
    top_k: Optional[int] = None,
    dedup: bool = True,
    created: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full synthesizability evaluation over one candidate dataset.

    Returns the summary dict and writes ``synthesizability.csv`` +
    ``synthesizability_summary.json`` to ``out_dir`` (defaults to ``dataset_dir``).
    """
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir) if out_dir else dataset_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, rows = read_candidates(dataset_dir)

    # Optionally evaluate only the best top_k by oracle score (papers run on top-k sets).
    if top_k is not None and top_k > 0:
        hib = bool(manifest.get("score_higher_is_better", False))

        def _score(r):
            try:
                return float(r.get("score"))
            except (TypeError, ValueError):
                return float("-inf") if hib else float("inf")

        rows = sorted(rows, key=_score, reverse=hib)[:top_k]

    # Per-candidate base record: validity (canonical SMILES) + SA score + self-claim.
    per_mol: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        smi = r.get("smiles", "")
        canon = _canonical(smi)
        try:
            self_nr = (
                int(float(r["num_reactions"])) if r.get("num_reactions") not in (None, "") else None
            )
        except (TypeError, ValueError):
            self_nr = None
        per_mol.append(
            {
                "candidate_id": r.get("candidate_id") or f"row-{i}",
                "smiles": smi,
                "canonical": canon,
                "valid": canon is not None,
                "sa_score": sa_score(smi) if canon is not None else None,
                "has_route": r.get("has_route"),
                "self_num_reactions": self_nr,
                # filled in below for evaluated molecules
                "evaluated": False,
                "solved": 0,
                "n_steps": None,
                "n_solved_routes": None,
                "top_score": None,
                "search_time": None,
                "error": None,
            }
        )

    # Build the unique set of valid molecules to actually search (dedup is the default;
    # RGFN reports over unique molecules, and retrosynthesis is the expensive step).
    key_to_idxs: Dict[str, List[int]] = {}
    for idx, rec in enumerate(per_mol):
        if not rec["valid"]:
            continue
        key = rec["canonical"] if dedup else f"{rec['canonical']}::{idx}"
        key_to_idxs.setdefault(key, []).append(idx)

    unique_keys = list(key_to_idxs)
    unique_smiles = [per_mol[key_to_idxs[k][0]]["canonical"] for k in unique_keys]

    print(
        f"[synthesizability] {dataset_dir.name}: {len(per_mol)} candidates, "
        f"{sum(r['valid'] for r in per_mol)} valid, "
        f"{len(unique_smiles)} unique to search (nproc={nproc})",
        flush=True,
    )

    results = run_aizynth(
        unique_smiles,
        config=config,
        stock=stock,
        expansion=expansion,
        filter_policy=filter_policy,
        time_limit=time_limit,
        iteration_limit=iteration_limit,
        nproc=nproc,
    )

    # Scatter each unique result back onto every candidate that shares the molecule.
    for key, res in zip(unique_keys, results):
        for idx in key_to_idxs[key]:
            per_mol[idx].update(res)
            per_mol[idx]["evaluated"] = True

    opts = {
        "config": str(config),
        "stock": stock,
        "expansion": expansion,
        "filter_policy": filter_policy,
        "time_limit": time_limit,
        "iteration_limit": iteration_limit,
        "nproc": nproc,
        "top_k": top_k,
        "dedup": dedup,
        "created": created,
        "aizynthfinder_version": _aizynth_version(),
    }
    summary = summarize(manifest, per_mol, opts)

    _write_per_mol(out_dir / PER_MOL_FILE, per_mol)
    with open(out_dir / SUMMARY_FILE, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"[synthesizability] success_rate={summary['aizynth_success_rate']:.3f} "
        f"({summary['n_solved']}/{summary['n_evaluated']})  "
        f"steps_mean={summary['steps_mean']:.2f}  sa_mean={summary['sa_mean']:.2f}",
        flush=True,
    )
    print(f"[synthesizability] wrote {out_dir/PER_MOL_FILE} and {out_dir/SUMMARY_FILE}", flush=True)
    return summary


_PER_MOL_COLS = [
    "candidate_id",
    "smiles",
    "valid",
    "evaluated",
    "solved",
    "n_steps",
    "n_solved_routes",
    "top_score",
    "sa_score",
    "has_route",
    "self_num_reactions",
    "search_time",
    "error",
]


def _write_per_mol(path: Path, per_mol: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_PER_MOL_COLS, extrasaction="ignore")
        w.writeheader()
        for rec in per_mol:
            w.writerow({k: ("" if rec.get(k) is None else rec.get(k)) for k in _PER_MOL_COLS})


def _aizynth_version() -> Optional[str]:
    # aizynthfinder exposes no __version__ attribute; read the installed dist version.
    try:
        from importlib.metadata import version

        return version("aizynthfinder")
    except Exception:
        try:
            import aizynthfinder

            return getattr(aizynthfinder, "__version__", None)
        except Exception:
            return None


# --- CLI ------------------------------------------------------------------------------


def _default_config() -> str:
    """Default to the public dataset config that setup_aizynthfinder.sh downloads."""
    return os.environ.get("AIZYNTH_CONFIG", "data/models/aizynthfinder/config.yml")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="AiZynthFinder + SA-Score synthesizability evaluation over a "
        "standard candidate dataset (post-hoc, runs in the `aizynth` env)."
    )
    p.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="candidate dataset dir (manifest.json + candidates.csv)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output dir for results (default: alongside the dataset)",
    )
    p.add_argument(
        "--config",
        default=_default_config(),
        help="AiZynthFinder config.yml (default: %(default)s)",
    )
    p.add_argument("--stock", default="zinc", help="stock key in the config")
    p.add_argument("--expansion", default="uspto", help="expansion-policy key in the config")
    p.add_argument(
        "--filter",
        dest="filter_policy",
        default="uspto",
        help="filter-policy key in the config ('none' to disable)",
    )
    p.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="per-molecule search time limit in seconds (overrides config)",
    )
    p.add_argument(
        "--iteration-limit",
        type=int,
        default=None,
        help="per-molecule search iteration limit (overrides config)",
    )
    p.add_argument("--nproc", type=int, default=1, help="parallel worker processes")
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="evaluate only the best K candidates by oracle score (default: all)",
    )
    p.add_argument(
        "--no-dedup",
        action="store_true",
        help="evaluate every row (default: dedup by canonical SMILES)",
    )
    p.add_argument("--created", default=None, help="ISO timestamp to stamp in the summary")
    args = p.parse_args(argv)

    filt = None if (args.filter_policy or "").lower() == "none" else args.filter_policy
    evaluate_dataset(
        args.dataset,
        out_dir=args.out,
        config=args.config,
        stock=args.stock,
        expansion=args.expansion,
        filter_policy=filt,
        time_limit=args.time_limit,
        iteration_limit=args.iteration_limit,
        nproc=args.nproc,
        top_k=args.top_k,
        dedup=not args.no_dedup,
        created=args.created,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
