#!/usr/bin/env python
"""Ingest a generator's pre-scored candidates into the standard candidate dataset.

The **fixed-reward** counterpart of ``scripts/score_batch.py --finalize``. In a
fixed-reward run there is **no oracle** — the generator trains against a fixed reward
generator (e.g. the pretrained sEH MPNN) and already knows each molecule's score (the
reward generator's own value). So there is nothing to *score* across the env boundary;
we only need to write the molecules out in the project's standard candidate-dataset
format (``glue.datasets.candidates``) so a baseline's fixed-reward output sits next to
RGFN's and the benchmark harness reads them uniformly.

Baseline generators (FragGFN, RxnFlow, SCENT) run in their own conda envs and cannot
import ``glue`` (it pulls in ``rgfn``/dgl). So, exactly like the oracle bridge, they
shell out to this CLI under the ``rgfn`` env::

    conda run -n rgfn python scripts/ingest_candidates.py \
        --pairs pairs.csv --out-dir <run>/candidates \
        --generator fraggfn --reward-name seh_proxy --system seh --seed 42 \
        --score-higher-is-better --score-units "seh_mpnn_reward (higher is better)"

``pairs.csv`` columns: ``smiles``, ``score`` (the reward generator's value), plus any
extra per-molecule columns (preserved verbatim). ``--routes`` is an optional JSONL of
``{"smiles": <canonical>, ...route fields...}`` (the ``glue.active_learning.route``
schema) for synthesizable entrants — joined onto candidates by SMILES so they emit
``has_route=1`` + ``routes.jsonl``; omit it for non-synthesizable entrants (FragGFN).

Runs ONLY in the ``rgfn`` env (it imports ``glue``); the baseline's own env never does.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import glue  # noqa: F401  (side effect: registers our components)
from glue.datasets.candidates import CandidateDataset, validate_candidate_dataset


def _load_routes(path: str) -> Dict[str, Dict]:
    """Build ``{canonical_smiles: route_dict}`` from a routes JSONL (or ``{}``)."""
    routes: Dict[str, Dict] = {}
    if not path:
        return routes
    p = Path(path)
    if not p.exists():
        print(f"[ingest] WARNING --routes file not found: {path}", flush=True)
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
            smi = (rec.pop("smiles", "") or "").strip()
            if smi:
                routes[smi] = rec
    return routes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True, help="CSV with 'smiles','score' (+ extra) columns")
    ap.add_argument("--out-dir", required=True, help="destination candidate-dataset directory")
    ap.add_argument("--generator", required=True, help="generator name (e.g. fraggfn, rxnflow)")
    ap.add_argument(
        "--reward-name",
        default=None,
        help="name of the fixed reward generator (stored in the manifest 'oracle' field "
        "for schema compatibility; it is a reward generator, not an oracle)",
    )
    ap.add_argument("--system", default=None, help="target system (e.g. seh) for provenance")
    ap.add_argument("--seed", type=int, default=None, help="run seed (provenance)")
    ap.add_argument("--score-units", dest="score_units", default=None)
    ap.add_argument("--source", default=None, help="pointer to the config/run that produced this")
    ap.add_argument(
        "--score-higher-is-better",
        action="store_true",
        help="orientation of 'score' (set for the sEH neural reward — higher is better)",
    )
    ap.add_argument(
        "--routes",
        default=None,
        help="optional routes JSONL for synthesizable entrants (joined by SMILES)",
    )
    ap.add_argument("--step", type=int, default=1, help="value for the standard 'step' column")
    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        raise SystemExit(f"--pairs not found: {args.pairs}")
    routes_by_smi = _load_routes(args.routes)

    ds = CandidateDataset(
        args.out_dir,
        generator=args.generator,
        oracle=args.reward_name,  # provenance: the fixed reward generator (not an oracle)
        system=args.system,
        seed=args.seed,
        score_higher_is_better=bool(args.score_higher_is_better),
        score_units=args.score_units,
        source=args.source,
        notes=(
            "Fixed-reward (single-shot) run: no oracle. 'score' is the reward generator's "
            "own value; 'step' is always 1 (no active-learning rounds)."
        ),
    )
    n = 0
    with open(pairs_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.pop("smiles", "") or "").strip()
            if not smi:
                continue
            score = row.pop("score", "")
            row.pop("step", None)
            extra = {k: v for k, v in row.items() if v not in ("", None)}
            ds.add(
                smiles=smi,
                score=float(score) if score not in ("", None) else None,
                step=args.step,
                route=routes_by_smi.get(smi),
                extra=extra or None,
            )
            n += 1
    ds.write()
    n_routes = len(routes_by_smi)
    issues = validate_candidate_dataset(args.out_dir)
    print(
        f"[ingest] wrote {n} candidates ({n_routes} with routes) -> {args.out_dir}; "
        f"conformance: {'OK' if not issues else issues}",
        flush=True,
    )
    if issues:
        raise SystemExit(f"candidate dataset not conformant: {issues}")


if __name__ == "__main__":
    main()
