"""Top-k-vs-oracle-calls curve — the ``[bengio2021gflownet]`` Fig. 7 comparison.

Reads the per-round ``oracle_calls.csv`` traces written by the active-learning
loop (:class:`glue.active_learning.acquisition_trace.AcquisitionTrace`) across
seeds and acquisition arms, and draws the headline oracle-efficiency plot: best
molecule reward found so far (y) versus cumulative expensive-oracle calls (x),
one line per arm (learned ``policy`` vs ``random`` acquisition), with a mean line
and a ±std band over seeds.

This lives in ``validation/`` (the comparative-evaluation layer) and only reads
files ``glue/`` wrote — the one-way dependency rule (``docs/ARCHITECTURE.md``) is
respected. It imports nothing from ``glue``; the trace CSV is the contract.

Usage::

    python -m validation.harness.acquisition_curve \
        --runs $SCRATCH/rgfn_runs/experiments/active_learning/6td3_curve \
        --out  validation/results/6td3_acquisition_curve \
        --metric topk_best --title "6TD3 — RGFN vs random acquisition"

``--runs`` accepts any mix of ``oracle_calls.csv`` files and directories (searched
recursively). Each file self-describes its ``acquisition`` arm and ``seed`` from
its rows, so runs can be pointed at a whole campaign directory.
"""

import argparse
import csv
import glob
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Okabe-Ito colourblind-safe palette (matches glue/analysis/plot.py), keyed by arm.
_ARM_COLOR = {
    "policy": "#0072B2",  # blue  — the learned RGFN acquisition
    "random": "#D55E00",  # vermillion — the uniform-policy baseline
}
_ARM_LABEL = {"policy": "RGFN (learned)", "random": "random acquisition"}
_FALLBACK_COLORS = ["#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def find_trace_files(paths: Sequence[str]) -> List[Path]:
    """Expand files / dirs / globs into a de-duplicated list of trace CSVs."""
    found: List[Path] = []
    for raw in paths:
        for hit in glob.glob(raw) or [raw]:
            p = Path(hit)
            if p.is_dir():
                found.extend(sorted(p.rglob("oracle_calls.csv")))
            elif p.name == "oracle_calls.csv" or p.suffix == ".csv":
                found.append(p)
    # De-dup, preserve order.
    seen, unique = set(), []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def load_rows(files: Sequence[Path]) -> List[Dict[str, object]]:
    """Load every trace row, tagging each with its source file.

    A ``seed`` blank in the CSV falls back to the file's parent-of-parent dir name
    so runs that predate seed logging still separate by run dir.
    """
    rows: List[Dict[str, object]] = []
    for f in files:
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    calls = int(r["oracle_calls_cumulative"])
                except (KeyError, ValueError):
                    continue
                seed = (r.get("seed") or "").strip()
                rows.append(
                    {
                        "acquisition": (r.get("acquisition") or "policy").strip(),
                        "seed": seed if seed else f.parent.name,
                        "oracle_calls": calls,
                        "topk_mean": _to_float(r.get("topk_mean")),
                        "topk_best": _to_float(r.get("topk_best")),
                        "source": str(f),
                    }
                )
    return rows


def aggregate(
    rows: Sequence[Dict[str, object]], metric: str
) -> Dict[str, List[Tuple[int, float, float, int]]]:
    """Group by (arm, oracle_calls) → ``(calls, mean, std, n)`` sorted by calls.

    ``std`` is the sample std (ddof=1) over seeds, 0.0 for a single seed.
    """
    buckets: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for r in rows:
        v = r[metric]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        buckets[(r["acquisition"], r["oracle_calls"])].append(float(v))
    out: Dict[str, List[Tuple[int, float, float, int]]] = defaultdict(list)
    for (arm, calls), vals in buckets.items():
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        out[arm].append((calls, mean, std, n))
    for arm in out:
        out[arm].sort(key=lambda t: t[0])
    return dict(out)


def write_aggregate_csv(
    agg: Dict[str, List[Tuple[int, float, float, int]]], metric: str, path: Path
) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["acquisition", "oracle_calls", f"{metric}_mean", f"{metric}_std", "n_seeds"])
        for arm, series in agg.items():
            for calls, mean, std, n in series:
                w.writerow([arm, calls, f"{mean:.6g}", f"{std:.6g}", n])


def plot_curve(
    agg: Dict[str, List[Tuple[int, float, float, int]]],
    metric: str,
    out_png: Path,
    lower_is_better: bool,
    title: Optional[str] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, (arm, series) in enumerate(sorted(agg.items())):
        xs = [c for c, _, _, _ in series]
        ys = [m for _, m, _, _ in series]
        es = [s for _, _, s, _ in series]
        color = _ARM_COLOR.get(arm, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
        label = _ARM_LABEL.get(arm, arm)
        ax.plot(xs, ys, marker="o", color=color, label=label, zorder=3)
        if any(e > 0 for e in es):
            lo = [y - e for y, e in zip(ys, es)]
            hi = [y + e for y, e in zip(ys, es)]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.18, zorder=2)

    nice = {"topk_best": "best reward so far", "topk_mean": "top-k mean reward so far"}
    ylab = nice.get(metric, metric)
    ax.set_xlabel("cumulative oracle calls")
    ax.set_ylabel(f"{ylab}  ({'lower = better' if lower_is_better else 'higher = better'})")
    ax.set_title(title or f"{ylab} vs oracle calls")
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", nargs="+", required=True, help="oracle_calls.csv files / dirs / globs"
    )
    ap.add_argument("--out", required=True, help="output directory for the PNG + aggregated CSV")
    ap.add_argument("--metric", default="topk_mean", choices=["topk_mean", "topk_best"])
    ap.add_argument(
        "--higher-is-better",
        dest="lower_is_better",
        action="store_false",
        help="oracle reward is higher-is-better (default: lower-is-better, our docking convention)",
    )
    ap.set_defaults(lower_is_better=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    files = find_trace_files(args.runs)
    if not files:
        raise SystemExit(f"No oracle_calls.csv found under: {args.runs}")
    rows = load_rows(files)
    agg = aggregate(rows, args.metric)
    if not agg:
        raise SystemExit("No finite metric values to plot (all NaN?).")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"acquisition_curve_{args.metric}.csv"
    png_path = out_dir / f"acquisition_curve_{args.metric}.png"
    write_aggregate_csv(agg, args.metric, csv_path)
    plot_curve(agg, args.metric, png_path, args.lower_is_better, args.title)

    print(f"[acquisition_curve] {len(files)} trace file(s), arms: {sorted(agg)}")
    for arm, series in sorted(agg.items()):
        seeds = max((n for _, _, _, n in series), default=0)
        last = series[-1] if series else None
        tail = f"@{last[0]} calls: {last[1]:.3f}±{last[2]:.3f}" if last else "(empty)"
        print(f"  {arm:>7}: {len(series)} points, up to {seeds} seed(s); {args.metric} {tail}")
    print(f"[acquisition_curve] wrote {csv_path}")
    print(f"[acquisition_curve] wrote {png_path}")


if __name__ == "__main__":
    main()
