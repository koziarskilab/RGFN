#!/usr/bin/env python
"""Oracle bridge — score a batch of SMILES with a named glue oracle (the shared
scoring standard for *every* benchmark entrant).

Why this exists
---------------
Baseline generators we benchmark against RGFN (FragGFN, SynFlowNet, VAE-BO, …)
run in their OWN conda environments — e.g. Recursion's ``gflownet`` (FragGFN) pins
python 3.10 / torch 2.1.2, which can't coexist with the ``rgfn`` env (python 3.11
/ torch 2.3). But a fair comparison requires every entrant to be scored by the
**same** oracle on the **same** budget. This CLI is that single shared standard:
it runs under the ``rgfn`` env (where ``glue`` + gnina + QuickVina2-GPU live) and
any generator, from any env, labels its query batch by shelling out to it::

    conda run -n rgfn python scripts/score_batch.py \
        --oracle docking_6td3_gpu --in batch.smi --out labels.csv \
        --suggestions-dir <run>/active_learning/suggestions --step 1 \
        --oracle-arg num_modes=9 --oracle-arg exhaustiveness=8000

So the docking oracle is invoked identically whether the molecules came from RGFN
(in-process, ``glue/active_learning/loop.py``) or from a non-synthesizable
baseline across the env boundary. See ``validation/generators/fraggfn/README.md``.

Logging — the standard candidate-dataset format
----------------------------------------------
With ``--suggestions-dir`` the bridge also writes the project's **standard
candidate-dataset format** (``glue.datasets.candidates``) so a baseline's output
sits next to RGFN's and the harness reads them uniformly:

  * per round it appends a raw shard ``shards/round_<step>.csv`` (crash-safe
    across subprocess calls) and one ``batch_metrics.csv`` row (same
    ``glue.metrics.dataset_metrics.batch_metrics`` the RGFN ``SuggestionLog`` uses);
  * a **synthesizable** entrant (e.g. RxnFlow) additionally passes ``--routes`` — a
    per-round JSONL of ``{"smiles": <canonical>, ...route fields...}`` (the route
    schema of ``glue.active_learning.route``) — which the bridge stores next to the
    shard (``shards/round_<step>_routes.jsonl``);
  * ``--finalize`` assembles the shards into the canonical ``candidates.csv`` +
    ``manifest.json``, joining any stored routes by SMILES so synthesizable entrants
    emit ``has_route=1`` + ``routes.jsonl`` and non-synthesizable ones (FragGFN)
    emit ``has_route=0`` and no ``routes.jsonl`` — the headline differentiator.

This module runs ONLY in the ``rgfn`` env (it imports ``glue``); the baseline's
own env never imports ``glue``/``rgfn``.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import glue  # noqa: F401  (side effect: registers our gin components / oracles)
from glue.datasets.candidates import CandidateDataset
from glue.metrics.dataset_metrics import batch_metrics
from glue.oracles import (
    Docking6TD3GpuOracle,
    Docking6TD3Oracle,
    DockingClpPOracle,
    DockingSEHOracle,
    MockGlueOracle,
)

# Oracle registry: --oracle <name> -> class. Names match each oracle's ``.name``
# where practical so logs/provenance line up across the pipeline.
ORACLES = {
    "docking_6td3_gpu": Docking6TD3GpuOracle,
    "docking_6td3": Docking6TD3Oracle,
    "docking_seh": DockingSEHOracle,
    "docking_clpp": DockingClpPOracle,
    "mock": MockGlueOracle,
}

SHARD_DIRNAME = "shards"
BATCH_METRICS_FILE = "batch_metrics.csv"


# --------------------------------------------------------------------- helpers
def _read_smiles(path: str) -> List[str]:
    """Read SMILES from a ``.smi`` (one per line) or a CSV with a ``smiles`` column."""
    p = Path(path)
    text = p.read_text().splitlines()
    if not text:
        return []
    # CSV with header containing 'smiles'?
    first = text[0].lower()
    if "smiles" in first and ("," in text[0]):
        with open(p, newline="") as fh:
            reader = csv.DictReader(fh)
            col = "smiles" if "smiles" in reader.fieldnames else reader.fieldnames[0]
            return [row[col].strip() for row in reader if row.get(col, "").strip()]
    # Plain one-SMILES-per-line (ignore blank lines / comments).
    return [ln.split()[0].strip() for ln in text if ln.strip() and not ln.startswith("#")]


def _coerce(val: str):
    """Coerce a CLI string to int/float/bool/str (for --oracle-arg KEY=VAL)."""
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            pass
    return val


def _parse_oracle_args(pairs: Optional[List[str]]) -> Dict:
    kwargs: Dict = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--oracle-arg must be KEY=VAL, got {item!r}")
        k, v = item.split("=", 1)
        kwargs[k.strip()] = _coerce(v.strip())
    return kwargs


def _score(oracle, smiles: List[str]):
    """Return ``(scores, details)``. ``details`` is a list of per-molecule breakdown
    dicts when the oracle exposes ``score_detailed`` (e.g. the GPU differential
    oracle), else ``None``. Mirrors ``ActiveLearningLoop._score_batch`` so RGFN and
    the baselines share identical scoring semantics."""
    score_detailed = getattr(oracle, "score_detailed", None)
    if callable(score_detailed):
        details = score_detailed(smiles)
        scores = [d.get("dvina", float("nan")) for d in details]
        return scores, details
    return list(oracle.score(smiles)), None


def _write_labels(path: str, smiles, scores, details) -> None:
    """Write the labels CSV returned to the caller (smiles,label,+breakdown)."""
    extra_cols: List[str] = []
    if details:
        for d in details:
            for k in d or {}:
                if k not in extra_cols and k != "dvina":
                    extra_cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "label", *extra_cols])
        for i, (smi, sc) in enumerate(zip(smiles, scores)):
            d = details[i] if details else {}
            w.writerow([smi, sc, *[(d or {}).get(c, "") for c in extra_cols]])


def _append_shard(sug_dir: Path, step: int, smiles, scores, details) -> None:
    shard_dir = sug_dir / SHARD_DIRNAME
    shard_dir.mkdir(parents=True, exist_ok=True)
    extra_cols: List[str] = []
    if details:
        for d in details:
            for k in d or {}:
                if k not in extra_cols and k != "dvina":
                    extra_cols.append(k)
    path = shard_dir / f"round_{step:03d}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "score", "step", *extra_cols])
        for i, (smi, sc) in enumerate(zip(smiles, scores)):
            d = details[i] if details else {}
            w.writerow([smi, sc, step, *[(d or {}).get(c, "") for c in extra_cols]])


def _append_batch_metrics(
    sug_dir: Path, step: int, smiles, scores, ref_smiles, threshold, higher_is_better
) -> Dict:
    metrics = batch_metrics(
        list(smiles),
        labels=list(scores),
        reference_smiles=ref_smiles,
        oracle_threshold=threshold,
        oracle_higher_is_better=higher_is_better,
    )
    metrics = {"al_round": step, **metrics}
    path = sug_dir / BATCH_METRICS_FILE
    exists = path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metrics.keys()), extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(metrics)
    return metrics


def _store_routes(sug_dir: Path, step: int, routes_path: str) -> int:
    """Copy a synthesizable entrant's per-round routes JSONL next to the shard.

    Each input line is ``{"smiles": <canonical>, ...route fields...}`` (the route
    schema of ``glue.active_learning.route``). We just persist it verbatim so
    ``--finalize`` can join routes onto candidates by SMILES — crash-safe across the
    per-round subprocess calls, exactly like the shard CSVs. Returns the line count.
    """
    src = Path(routes_path)
    if not src.exists():
        print(f"[score_batch] WARNING --routes file not found: {routes_path}", flush=True)
        return 0
    shard_dir = sug_dir / SHARD_DIRNAME
    shard_dir.mkdir(parents=True, exist_ok=True)
    dst = shard_dir / f"round_{step:03d}_routes.jsonl"
    lines = [ln for ln in src.read_text().splitlines() if ln.strip()]
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def _load_routes(sug_dir: Path) -> Dict[str, Dict]:
    """Build ``{canonical_smiles: route_dict}`` from all stored per-round route files.

    Empty (FragGFN / no ``--routes`` was ever passed) → ``{}``, so finalize falls
    back to ``route=None`` and ``has_route=0``. The ``smiles`` key is popped from each
    record; the remainder is the route dict handed to ``CandidateDataset.add(route=)``.
    """
    routes_by_smi: Dict[str, Dict] = {}
    shard_dir = sug_dir / SHARD_DIRNAME
    if not shard_dir.exists():
        return routes_by_smi
    for rf in sorted(shard_dir.glob("round_*_routes.jsonl")):
        with open(rf) as fh:
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
                    routes_by_smi[smi] = rec
    return routes_by_smi


def _finalize(sug_dir: Path, args, higher_is_better: bool) -> None:
    """Assemble per-round shards into the canonical standard candidate dataset.

    Synthesizable entrants (RxnFlow, RGFN) supplied ``--routes`` per round; those are
    joined onto candidates by SMILES here so ``has_route``/``num_reactions``/
    ``routes.jsonl`` are populated. Non-synthesizable entrants (FragGFN) supplied no
    routes → ``route=None`` → ``has_route=0`` (unchanged behaviour)."""
    shard_dir = sug_dir / SHARD_DIRNAME
    shards = sorted(shard_dir.glob("round_*.csv")) if shard_dir.exists() else []
    routes_by_smi = _load_routes(sug_dir)
    ds = CandidateDataset(
        sug_dir,
        generator=args.generator,
        oracle=args.oracle,
        system=args.system,
        seed=args.seed,
        score_higher_is_better=higher_is_better,
        score_units=args.score_units,
        source=args.source,
        notes=f"{args.generator} active-learning query batches; the standard 'step' column "
        "is the AL round. Routes recorded for synthesizable entrants (has_route=1).",
    )
    n = 0
    for shard in shards:
        with open(shard, newline="") as fh:
            for row in csv.DictReader(fh):
                smi = row.pop("smiles", "").strip()
                if not smi:
                    continue
                score = row.pop("score", "")
                step = row.pop("step", "")
                extra = {k: v for k, v in row.items() if v not in ("", None)}
                ds.add(
                    smiles=smi,
                    score=float(score) if score not in ("", None) else None,
                    step=int(float(step)) if step not in ("", None) else None,
                    route=routes_by_smi.get(smi),  # None for non-synthesizable entrants
                    extra=extra or None,
                )
                n += 1
    ds.write()
    n_routes = sum(1 for s in routes_by_smi)
    print(
        f"[score_batch] finalized {n} candidates from {len(shards)} shard(s) "
        f"({n_routes} with routes) -> {sug_dir}",
        flush=True,
    )


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle", required=True, choices=sorted(ORACLES), help="oracle name")
    ap.add_argument(
        "--in", dest="inp", help="input SMILES (.smi one-per-line, or CSV with 'smiles')"
    )
    ap.add_argument("--out", help="output labels CSV (smiles,label,+breakdown)")
    ap.add_argument(
        "--oracle-arg",
        action="append",
        default=[],
        help="oracle constructor kwarg KEY=VAL (repeatable). For docking_6td3_gpu "
        "pass the same knobs as configs/glue/active_learning_6td3_gpu.gin.",
    )
    # Standard-format logging (optional).
    ap.add_argument(
        "--suggestions-dir", help="write standard candidate-dataset shards + batch_metrics here"
    )
    ap.add_argument(
        "--step", type=int, default=1, help="AL round number (the standard 'step' column)"
    )
    ap.add_argument(
        "--reference-csv", help="seed D_0 CSV (for the novelty metric); 'smiles' column"
    )
    ap.add_argument(
        "--routes",
        help="per-round synthesis routes JSONL (synthesizable entrants only): one "
        "{'smiles': <canonical>, ...route fields...} per line. Stored next to the "
        "shard and joined onto candidates by SMILES on --finalize.",
    )
    ap.add_argument("--generator", default="fraggfn", help="generator name stamped in the dataset")
    ap.add_argument("--system", default=None, help="target system (e.g. 6td3) for provenance")
    ap.add_argument("--seed", type=int, default=None, help="run seed (provenance)")
    ap.add_argument("--score-units", dest="score_units", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument(
        "--threshold",
        type=float,
        default=-2.0,
        help="'good glue' cutoff reported in batch metrics (default -2.0, the 6td3 mark)",
    )
    ap.add_argument(
        "--finalize",
        action="store_true",
        help="assemble shards into canonical candidates.csv+manifest.json and exit",
    )
    args = ap.parse_args()

    oracle_cls = ORACLES[args.oracle]
    higher_is_better = bool(getattr(oracle_cls, "higher_is_better", False))

    # --finalize: no scoring, just assemble the standard dataset from shards.
    if args.finalize:
        if not args.suggestions_dir:
            raise SystemExit("--finalize requires --suggestions-dir")
        _finalize(Path(args.suggestions_dir), args, higher_is_better)
        return

    if not args.inp:
        raise SystemExit("--in is required (unless --finalize)")
    smiles = _read_smiles(args.inp)
    if not smiles:
        raise SystemExit(f"no SMILES read from {args.inp}")

    oracle = oracle_cls(**_parse_oracle_args(args.oracle_arg))
    print(
        f"[score_batch] scoring {len(smiles)} SMILES with {args.oracle} "
        f"(higher_is_better={higher_is_better})",
        flush=True,
    )
    scores, details = _score(oracle, smiles)

    n_valid = sum(1 for s in scores if s is not None and s == s)
    print(f"[score_batch] scored {n_valid}/{len(smiles)} successfully", flush=True)

    if args.out:
        _write_labels(args.out, smiles, scores, details)
        print(f"[score_batch] wrote labels -> {args.out}", flush=True)

    if args.suggestions_dir:
        sug_dir = Path(args.suggestions_dir)
        sug_dir.mkdir(parents=True, exist_ok=True)
        _append_shard(sug_dir, args.step, smiles, scores, details)
        if args.routes:
            n_r = _store_routes(sug_dir, args.step, args.routes)
            print(f"[score_batch] round {args.step}: stored {n_r} route(s)", flush=True)
        ref = None
        if args.reference_csv and Path(args.reference_csv).exists():
            ref = _read_smiles(args.reference_csv)
        m = _append_batch_metrics(
            sug_dir, args.step, smiles, scores, ref, args.threshold, higher_is_better
        )
        print(
            f"[score_batch] round {args.step}: shard + batch_metrics written "
            f"(modes={m.get('num_modes')}, div={m.get('internal_diversity', float('nan')):.2f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
