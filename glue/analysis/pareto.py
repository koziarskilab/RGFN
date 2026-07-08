"""Pareto-front extraction over a sweep's tidy result rows.

The sweep (``glue.analysis.sweep``) emits one row per
``(hub strategy × molecule strategy × m × k × trial)`` with metric columns. The end
goal is a **Pareto front** trading, e.g., diversity against concurrency or against
cost. This module is that last, small step: given the rows and a list of objectives
(which column to maximize/minimize), return the non-dominated subset.

Kept dependency-light on purpose — it works on a plain ``list[dict]`` (no pandas), so
it runs anywhere the CSV can be read and is trivially unit-testable. Building the
actual plot is left to the caller; this just hands them the frontier.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

# An objective is (column_name, "min" | "max").
Objective = Tuple[str, str]


def _as_float(row: Dict[str, Any], col: str):
    v = row.get(col)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


def dominates(a: Dict[str, Any], b: Dict[str, Any], objectives: Sequence[Objective]) -> bool:
    """True if row ``a`` Pareto-dominates row ``b``: no worse on every objective and
    strictly better on at least one. Rows with a missing/NaN value on any objective
    cannot dominate (comparison is undefined), keeping the front conservative."""
    strictly_better_somewhere = False
    for col, direction in objectives:
        av, bv = _as_float(a, col), _as_float(b, col)
        if av is None or bv is None:
            return False
        if direction == "max":
            if av < bv:
                return False
            if av > bv:
                strictly_better_somewhere = True
        elif direction == "min":
            if av > bv:
                return False
            if av < bv:
                strictly_better_somewhere = True
        else:
            raise ValueError(f"objective direction must be 'min' or 'max', got {direction!r}")
    return strictly_better_somewhere


def pareto_front(
    rows: Sequence[Dict[str, Any]], objectives: Sequence[Objective]
) -> List[Dict[str, Any]]:
    """Return the non-dominated rows for the given objectives.

    Args:
        rows: tidy result rows (e.g. from ``results.csv`` via ``csv.DictReader``).
        objectives: e.g. ``[("internal_diversity", "max"), ("concurrency", "min")]``
            or ``[("internal_diversity", "max"), ("cost_per_molecule", "min")]``.

    Returns the subset of ``rows`` not dominated by any other row (order preserved).
    Rows missing a value on any objective are excluded from the front.
    """
    # Guard against a silent empty front when an objective column is misspelled or
    # absent from every row (a real footgun after aggregate_trials renames X -> X_mean).
    if rows:
        for col, _ in objectives:
            if not any(col in r for r in rows):
                raise KeyError(
                    f"objective column {col!r} not present in any row; "
                    f"available columns: {sorted(rows[0].keys())}"
                )
    usable = [r for r in rows if all(_as_float(r, c) is not None for c, _ in objectives)]
    front: List[Dict[str, Any]] = []
    for r in usable:
        if not any(dominates(other, r, objectives) for other in usable if other is not r):
            front.append(r)
    return front


def aggregate_trials(
    rows: Sequence[Dict[str, Any]],
    group_keys: Sequence[str],
    metric_cols: Sequence[str],
) -> List[Dict[str, Any]]:
    """Average metric columns across trials for each unique combination of ``group_keys``.

    Multiple trials (seeds) give one row each; for a stable front you usually want the
    per-configuration mean (and std) rather than every trial's point. Returns one row
    per group with ``<metric>_mean`` / ``<metric>_std`` / ``n_trials`` columns plus the
    group-key columns, so it can feed straight back into :func:`pareto_front`.
    """
    from statistics import mean, pstdev

    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        key = tuple(r.get(k) for k in group_keys)
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for key, members in groups.items():
        agg: Dict[str, Any] = {k: v for k, v in zip(group_keys, key)}
        agg["n_trials"] = len(members)
        for col in metric_cols:
            vals = [_as_float(m, col) for m in members]
            vals = [v for v in vals if v is not None]
            if vals:
                agg[f"{col}_mean"] = mean(vals)
                agg[f"{col}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
            else:
                agg[f"{col}_mean"] = float("nan")
                agg[f"{col}_std"] = float("nan")
        out.append(agg)
    return out
