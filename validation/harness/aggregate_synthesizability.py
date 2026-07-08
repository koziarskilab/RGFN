#!/usr/bin/env python
"""Assemble a multi-generator synthesizability comparison table.

``validation/harness/synthesizability.py`` scores ONE candidate dataset (AiZynthFinder
route-finding + SA), writing a ``synthesizability_summary.json`` next to it. This script
gathers those per-generator summaries into a single comparison — the headline table for a
benchmark write-up (e.g. the matched four-way fixed-reward sEH run: RGFN / FragGFN /
RxnFlow / SCENT).

Dependency-light (stdlib only): reads the per-dataset ``synthesizability_summary.json``
and, when present, pulls mean medchem descriptors (MW, QED) from ``candidates.csv`` and the
route flag from ``manifest.json``. Runs in any env (no torch/rdkit needed).

    python validation/harness/aggregate_synthesizability.py \
        --dataset rgfn=<run>/rgfn/candidates \
        --dataset fraggfn=<run>/fraggfn/candidates \
        --dataset rxnflow=<run>/rxnflow/candidates \
        --dataset scent=<run>/scent/candidates \
        --out validation/results/seh_fixed_reward/comparison.csv \
        --out-md validation/results/seh_fixed_reward/comparison.md \
        --title "Matched four-way fixed-reward sEH benchmark"
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

# Columns pulled from each generator's synthesizability_summary.json, in table order.
_SUMMARY_COLUMNS: List[Tuple[str, str]] = [
    ("n_candidates", "n_candidates"),
    ("n_unique_valid", "n_unique_valid"),
    ("n_evaluated", "n_evaluated"),
    ("aizynth_success_rate", "aizynth_success"),
    ("steps_mean", "steps_mean"),
    ("steps_median", "steps_median"),
    ("sa_mean", "sa_mean"),
    ("self_reported_route_rate", "self_route_rate"),
]
# Medchem means computed from candidates.csv (per-molecule descriptor columns).
_MEDCHEM_COLUMNS: List[str] = ["mol_weight", "qed", "clogp"]


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:  # NaN
            return ""
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:.1f}"
    return str(v)


def _medchem_means(candidates_csv: Path) -> Dict[str, float]:
    """Mean of each medchem descriptor column over valid rows (empty dict if unavailable)."""
    if not candidates_csv.exists():
        return {}
    vals: Dict[str, List[float]] = {c: [] for c in _MEDCHEM_COLUMNS}
    with open(candidates_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("valid", "1")) not in ("1", "True", "true"):
                continue
            for c in _MEDCHEM_COLUMNS:
                try:
                    vals[c].append(float(row[c]))
                except (KeyError, TypeError, ValueError):
                    pass
    return {f"{c}_mean": (mean(v) if v else float("nan")) for c, v in vals.items()}


def _row_for(name: str, dataset_dir: Path) -> Optional[Dict]:
    summary_path = dataset_dir / "synthesizability_summary.json"
    if not summary_path.exists():
        print(f"[aggregate] WARNING no summary for {name!r}: {summary_path} — skipping", flush=True)
        return None
    with open(summary_path) as fh:
        summary = json.load(fh)
    manifest = {}
    mpath = dataset_dir / "manifest.json"
    if mpath.exists():
        with open(mpath) as fh:
            manifest = json.load(fh)

    row: Dict = {
        "generator": name,
        "reward": summary.get("oracle") or manifest.get("oracle"),
        "system": summary.get("system") or manifest.get("system"),
        "has_route": bool(manifest.get("has_routes")),
    }
    for src, dst in _SUMMARY_COLUMNS:
        row[dst] = summary.get(src)
    row.update(_medchem_means(dataset_dir / "candidates.csv"))
    return row


def _write_csv(rows: List[Dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["generator", "reward", "system", "has_route"]
    cols += [dst for _, dst in _SUMMARY_COLUMNS]
    cols += [f"{c}_mean" for c in _MEDCHEM_COLUMNS]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[aggregate] wrote {out}", flush=True)


def _write_md(rows: List[Dict], out: Path, title: Optional[str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "generator",
        "has_route",
        "n_unique_valid",
        "n_evaluated",
        "aizynth_success",
        "steps_mean",
        "sa_mean",
        "self_route_rate",
        "mol_weight_mean",
        "qed_mean",
    ]
    headers = [
        "Generator",
        "Synth?",
        "n_uniq",
        "n_eval",
        "AiZynth success",
        "steps",
        "SA",
        "self-route",
        "MW",
        "QED",
    ]
    lines = []
    if title:
        lines.append(f"# {title}\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells = []
        for c in cols:
            if c == "has_route":
                # blank for a pending placeholder row (no 'has_route' key), else yes/no.
                cells.append("" if "has_route" not in r else ("yes" if r["has_route"] else "no"))
            else:
                cells.append(_fmt(r.get(c)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append(
        "\n*AiZynth success = fraction of unique valid molecules with a full retrosynthetic "
        "route to in-stock precursors (the headline synthesizability metric). self-route = "
        "the generator's by-construction route claim (RGFN/RxnFlow/SCENT = 1.0; FragGFN = 0).*\n"
    )
    pending = [r["generator"] for r in rows if "has_route" not in r]
    if pending:
        lines.append(
            f"\n> **Blank row(s) pending:** {', '.join(pending)} — run still training when this "
            "table was built. Re-run this system's AiZynth aggregation to fill in "
            "(see `Logs/021` ▶ PICK UP HERE).\n"
        )
    out.write_text("\n".join(lines))
    print(f"[aggregate] wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="a generator's candidate-dataset dir (repeatable), e.g. rgfn=<run>/rgfn/candidates",
    )
    ap.add_argument("--out", required=True, help="output comparison CSV")
    ap.add_argument("--out-md", default=None, help="optional markdown table")
    ap.add_argument("--title", default=None, help="title for the markdown table")
    ap.add_argument(
        "--expect",
        action="append",
        default=[],
        help="generator that MUST appear as a row, in this order (repeatable). If its "
        "dataset/summary is missing (e.g. a run still training), emit a BLANK placeholder "
        "row to fill in later. Without --expect, only scored datasets appear (old behavior).",
    )
    args = ap.parse_args()

    provided: Dict[str, Path] = {}
    for spec in args.dataset:
        if "=" not in spec:
            raise SystemExit(f"--dataset must be NAME=DIR, got {spec!r}")
        name, d = spec.split("=", 1)
        provided[name.strip()] = Path(d.strip())

    # Row order: --expect if given (so pending entrants get blank placeholder rows), else
    # just the datasets that were provided.
    order = args.expect if args.expect else list(provided)
    rows: List[Dict] = []
    for name in order:
        row = _row_for(name, provided[name]) if name in provided else None
        if row is None:
            if args.expect:  # explicitly expected -> keep a blank row to fill in later
                print(f"[aggregate] {name!r}: no summary -> BLANK placeholder row", flush=True)
                row = {"generator": name}
            else:
                continue
        rows.append(row)
    if not rows:
        raise SystemExit("no datasets with a synthesizability_summary.json were found")

    _write_csv(rows, Path(args.out))
    if args.out_md:
        _write_md(rows, Path(args.out_md), args.title)
    # Echo the table to stdout for quick inspection (placeholder rows print as PENDING).
    for r in rows:
        if "has_route" not in r:
            print(f"  {r['generator']:>8} | PENDING (blank placeholder row)", flush=True)
            continue
        print(
            f"  {r['generator']:>8} | synth={'yes' if r['has_route'] else 'no ':>3} | "
            f"AiZynth={_fmt(r.get('aizynth_success'))} | SA={_fmt(r.get('sa_mean'))} | "
            f"MW={_fmt(r.get('mol_weight_mean'))} | QED={_fmt(r.get('qed_mean'))}",
            flush=True,
        )


if __name__ == "__main__":
    main()
