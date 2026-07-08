#!/usr/bin/env python
"""Retroactive per-round metrics for a finished active-learning run.

The loop already writes ``suggestions/batch_metrics.csv`` live, but the point of
keeping the standard ``candidates.csv`` (step + SMILES + score per molecule) is
that the *same* metrics can be recomputed after the fact — on this run, on an old
run that predates a metric, with a different novelty reference, or on ANY
generator's standard candidate dataset (SynFlowNet, VAE-BO, …) — without
re-docking anything. This script is that recomputation: it reads a standard
``candidates.csv`` (or any CSV with a round/step, SMILES, and score column),
groups by step, and emits the full batch-metrics table via
``glue.metrics.dataset_metrics``.

Usage:
    python experiments/active_learning/6td3/analyze_suggestions.py \
        --suggestions <run>/active_learning/suggestions/candidates.csv \
        --seed experiments/active_learning/6td3/seed_6td3.csv \
        --out <run>/active_learning/suggestions/batch_metrics_recomputed.csv

Defaults assume the standard schema (``step``/``smiles``/``score``); override the
column flags for a legacy ``suggestions_all.csv`` (``--round-col al_round
--label-col oracle_label``). Run from the repo root (so ``import glue`` resolves).
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from glue.metrics.dataset_metrics import batch_metrics


def _read_smiles_column(path: Path, smiles_col: str) -> list:
    with open(path, newline="") as fh:
        return [row[smiles_col] for row in csv.DictReader(fh) if row.get(smiles_col)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--suggestions",
        required=True,
        help="standard candidates.csv (or any step/smiles/score CSV)",
    )
    ap.add_argument("--seed", default=None, help="seed D_0 CSV for the novelty metric (optional)")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--round-col", default="step")
    ap.add_argument("--label-col", default="score")
    ap.add_argument("--seed-smiles-col", default="smiles")
    ap.add_argument("--oracle-threshold", type=float, default=-2.0)
    ap.add_argument("--oracle-higher-is-better", action="store_true")
    ap.add_argument("--out", default=None, help="write the per-round metrics table here (CSV)")
    args = ap.parse_args()

    seed_smiles = _read_smiles_column(Path(args.seed), args.seed_smiles_col) if args.seed else None

    by_round = defaultdict(lambda: {"smiles": [], "labels": []})
    with open(args.suggestions, newline="") as fh:
        for row in csv.DictReader(fh):
            rnd = int(float(row[args.round_col]))
            by_round[rnd]["smiles"].append(row[args.smiles_col])
            try:
                by_round[rnd]["labels"].append(float(row[args.label_col]))
            except (TypeError, ValueError, KeyError):
                by_round[rnd]["labels"].append(float("nan"))

    rows = []
    for rnd in sorted(by_round):
        m = batch_metrics(
            by_round[rnd]["smiles"],
            labels=by_round[rnd]["labels"],
            reference_smiles=seed_smiles,
            oracle_threshold=args.oracle_threshold,
            oracle_higher_is_better=args.oracle_higher_is_better,
        )
        m = {"al_round": rnd, **m}
        rows.append(m)
        print(
            f"round {rnd}: n={m['n_suggested']} unique={m['n_unique']} "
            f"modes={m['num_modes']} scaffolds={m['num_scaffolds']} "
            f"MW={m.get('mol_weight_mean', float('nan')):.1f} "
            f"QED={m.get('qed_mean', float('nan')):.3f} "
            f"div={m.get('internal_diversity', float('nan')):.3f} "
            f"novelty={m.get('novelty_vs_seed', float('nan')):.3f} "
            f"oracle_med={m.get('oracle_median', float('nan')):.2f}"
        )

    if args.out and rows:
        cols = sorted({k for r in rows for k in r})
        cols = ["al_round"] + [c for c in cols if c != "al_round"]
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
