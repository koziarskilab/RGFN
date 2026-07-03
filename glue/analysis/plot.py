"""Pareto-front plots for a diversification sweep.

Renders the trade-off the sweep exists to expose — **diversity vs. concurrency** and
**diversity vs. cost** — as a scatter of strategy configurations with the non-dominated
frontier highlighted. Reads a sweep's ``results.csv`` (or rows), averages across trials
(mean ± std error bars), and writes PNGs.

Design (per the project's dataviz guidance):
  * Form = scatter + highlighted frontier — the right form for a 2-objective trade-off.
  * Color = **categorical / identity** on the hub selector, assigned in a *fixed* order
    (color follows the entity, not its rank) from the **Okabe–Ito** palette — a published
    colorblind-safe set, so no palette validation step is needed.
  * **Secondary encoding**: marker *shape* on the molecule selector, so a configuration's
    identity never rests on color alone (CVD / print safe). Both encodings get a legend.
  * Recessive grid/axes, ≥8px markers, thin 2px frontier line, selective annotation of
    frontier points only (never a label on every point).

matplotlib is imported lazily (Agg backend) so importing ``glue.analysis`` never requires
it; plotting is best-effort and callers guard for its absence.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from glue.analysis.pareto import aggregate_trials, pareto_front

# Okabe–Ito colorblind-safe categorical palette (skip yellow/black — reserved for ink).
_OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_FRONT_INK = "#222222"
_GRID_INK = "#DDDDDD"


def _stable_map(values: Sequence[str], options: Sequence) -> Dict[str, Any]:
    """Map each distinct value to an option in a *fixed* (sorted) order, so identity →
    color/shape is stable across runs and never reassigned by rank."""
    return {v: options[i % len(options)] for i, v in enumerate(sorted(set(values)))}


def plot_pareto(
    rows: Sequence[Dict[str, Any]],
    x_col: str,
    y_col: str,
    out_path,
    *,
    minimize_x: bool = True,
    maximize_y: bool = True,
    x_err: Optional[str] = None,
    y_err: Optional[str] = None,
    color_by: str = "hub_selector",
    marker_by: str = "mol_selector",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    annotate_front: bool = True,
) -> Path:
    """Scatter ``rows`` on (x_col, y_col) with the Pareto frontier drawn through it.

    Points are colored by ``color_by`` and shaped by ``marker_by`` (both in a legend).
    The frontier is computed with ``pareto_front`` for the given objective directions and
    drawn as a connected line; frontier points are emphasized and (optionally) annotated.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    rows = [r for r in rows if _f(r, x_col) is not None and _f(r, y_col) is not None]
    if not rows:
        raise ValueError(f"no rows with numeric {x_col!r} and {y_col!r} to plot")

    color_of = _stable_map([str(r.get(color_by)) for r in rows], _OKABE_ITO)
    marker_of = _stable_map([str(r.get(marker_by)) for r in rows], _MARKERS)

    front = pareto_front(
        rows,
        [(x_col, "min" if minimize_x else "max"), (y_col, "max" if maximize_y else "min")],
    )
    front_ids = {id(r) for r in front}

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=140)
    ax.set_axisbelow(True)
    ax.grid(True, color=_GRID_INK, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # error bars (recessive), then markers on top.
    for r in rows:
        xe = _f(r, x_err) if x_err else None
        ye = _f(r, y_err) if y_err else None
        if xe or ye:
            ax.errorbar(
                _f(r, x_col),
                _f(r, y_col),
                xerr=xe,
                yerr=ye,
                fmt="none",
                ecolor="#BBBBBB",
                elinewidth=0.8,
                capsize=2,
                zorder=1,
            )
    for r in rows:
        on_front = id(r) in front_ids
        ax.scatter(
            _f(r, x_col),
            _f(r, y_col),
            s=110 if on_front else 64,
            c=color_of[str(r.get(color_by))],
            marker=marker_of[str(r.get(marker_by))],
            edgecolors=_FRONT_INK if on_front else "white",
            linewidths=1.6 if on_front else 0.6,
            zorder=3 if on_front else 2,
        )

    # frontier line through the non-dominated points, ordered along x.
    fl = sorted(front, key=lambda r: _f(r, x_col))
    if len(fl) >= 2:
        ax.plot(
            [_f(r, x_col) for r in fl],
            [_f(r, y_col) for r in fl],
            color=_FRONT_INK,
            linewidth=2.0,
            linestyle="--",
            zorder=2,
            label="_nolegend_",
        )
    if annotate_front:
        # Dedupe labels at coincident points (configs that collapse to the same result,
        # e.g. k larger than the available children) so annotations don't overprint.
        seen_xy = set()
        for r in fl:
            xv, yv = _f(r, x_col), _f(r, y_col)
            key = (round(xv, 4), round(yv, 4))
            if key in seen_xy:
                continue
            seen_xy.add(key)
            ax.annotate(
                f"m{r.get('m')}·k{r.get('k')}",
                (xv, yv),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=7,
                color=_FRONT_INK,
            )

    ax.set_xlabel(x_label or x_col, fontsize=11)
    ax.set_ylabel(y_label or y_col, fontsize=11)
    ax.set_title(title or f"Pareto front: {y_label or y_col} vs {x_label or x_col}", fontsize=12)

    _dual_legend(ax, plt, color_of, marker_of, color_by, marker_by)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _dual_legend(ax, plt, color_of, marker_of, color_by, marker_by) -> None:
    """Two legends placed OUTSIDE the axes (right side): color→``color_by`` (identity)
    and shape→``marker_by`` (secondary). Kept off the plot so they never collide with
    data points or frontier annotations, whatever the data distribution."""
    from matplotlib.lines import Line2D

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=c,
            markeredgecolor="white",
            markersize=9,
            label=name,
        )
        for name, c in sorted(color_of.items())
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=m,
            linestyle="none",
            markerfacecolor="#666666",
            markeredgecolor="white",
            markersize=9,
            label=name,
        )
        for name, m in sorted(marker_of.items())
    ]
    front_handle = [
        Line2D([0], [0], color=_FRONT_INK, linewidth=2.0, linestyle="--", label="Pareto front")
    ]
    leg1 = ax.legend(
        handles=color_handles,
        title=color_by,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=marker_handles + front_handle,
        title=marker_by,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )


def plot_sweep_fronts(results_csv, out_dir=None) -> List[Path]:
    """Convenience: read a sweep ``results.csv``, average trials, and write the two
    standard fronts (diversity vs concurrency, diversity vs cost/molecule) as PNGs.

    Returns the list of written PNG paths (skips a front whose columns are absent — e.g.
    cost columns when the sweep ran without a priced library).
    """
    results_csv = Path(results_csv)
    out_dir = Path(out_dir) if out_dir else results_csv.parent
    with open(results_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []

    metrics = [
        "internal_diversity",
        "num_modes",
        "concurrency",
        "cost_per_molecule",
        "cost_saving_ratio",
        "reward_mean",
        "n_molecules",
    ]
    have = [m for m in metrics if any(m in r for r in rows)]
    agg = aggregate_trials(rows, ["hub_selector", "mol_selector", "m", "k"], have)
    for a in agg:  # carry m/k as ints for annotation
        for key in ("m", "k"):
            a[key] = a.get(key)

    written: List[Path] = []
    fronts = [
        (
            "internal_diversity",
            "concurrency",
            "diversity_vs_concurrency.png",
            "concurrency (parallel lanes)",
            "internal diversity",
            True,
        ),
        (
            "internal_diversity",
            "cost_per_molecule",
            "diversity_vs_cost.png",
            "cost per molecule (hub-batched)",
            "internal diversity",
            True,
        ),
    ]
    for y_base, x_base, fname, xlab, ylab, minimize_x in fronts:
        xm, ym = f"{x_base}_mean", f"{y_base}_mean"
        if not any(xm in a for a in agg) or not any(ym in a for a in agg):
            continue
        p = plot_pareto(
            agg,
            x_col=xm,
            y_col=ym,
            out_path=out_dir / fname,
            minimize_x=minimize_x,
            maximize_y=True,
            x_err=f"{x_base}_std",
            y_err=f"{y_base}_std",
            x_label=xlab,
            y_label=ylab,
            title=f"{ylab} vs {xlab}",
        )
        written.append(p)
    return written


def _f(row: Dict[str, Any], col: Optional[str]):
    if not col:
        return None
    try:
        v = float(row.get(col))
    except (TypeError, ValueError):
        return None
    return v if v == v else None
