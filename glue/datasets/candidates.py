"""The standard on-disk format for *candidate* datasets — molecules a generator
proposed, with scores and (optionally) synthesis routes.

Why this exists
---------------
We benchmark our RGFN pipeline against baseline generators (SynFlowNet, FragGFN,
VAE-BO, …; ``validation/generators/``). For the comparison to be apples-to-apples,
every entrant — ours included — must emit its candidates in **one** format the
benchmark harness can read uniformly. This module *is* that format: a single,
generator-agnostic writer/reader plus a documented schema.

Where it lives, and why here
----------------------------
``validation/`` may import from ``glue/`` but **never** the reverse
(``docs/ARCHITECTURE.md``; ``validation/README.md``: "If you find yourself wanting
``glue/`` to import something here, push it down into ``glue/``"). Our production
loop must be able to *write* this format, so the format must live where the loop
can import it — here, in ``glue/datasets/``. Baseline-generator adapters under
``validation/generators/`` and the harness then import this module.

The format
----------
One dataset is a directory::

    <dataset>/
      manifest.json    # dataset card (provenance: generator, oracle, system, seed, ...)
      candidates.csv   # one row per candidate (the standard table)
      routes.jsonl     # one JSON synthesis route per candidate that has one (optional file)

``candidates.csv`` columns:
    Required : candidate_id, smiles, generator
    Standard : step, score, oracle, valid, has_route, num_reactions
               + per-molecule descriptors (mol_weight, qed, clogp, h_donors,
                 h_acceptors, heavy_atoms, rotatable_bonds, tpsa, num_rings,
                 lipinski_pass, ligand_efficiency) — recomputable from SMILES, but
                 written so a reader needn't depend on RDKit.
    Extra    : any additional columns (e.g. an oracle's per-pose breakdown
               vina_t2/cnnsc_t2/…) are preserved verbatim.

``routes.jsonl`` lines (same structure as ``glue.active_learning.route``)::

    {"candidate_id": ..., "product_smiles": ..., "num_reactions": N,
     "building_block": {"smiles": ..., "idx": ...},
     "steps": [{"step": 1, "reaction_idx": ..., "reaction_smarts": "...",
                "reactant": "...", "fragments": [...], "product": "..."}, ...]}

Non-synthesizable generators (VAE-BO, FragGFN) simply omit routes; ``has_route``
is 0 and ``routes.jsonl`` may be absent.
"""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from glue.metrics.dataset_metrics import descriptors_for_smiles

CANDIDATE_SCHEMA_VERSION = "1.0"

# Identity of a candidate row.
REQUIRED_COLUMNS: List[str] = ["candidate_id", "smiles", "generator"]
# The canonical, ordered columns every conformant dataset uses (beyond required).
_PROVENANCE_COLUMNS: List[str] = ["step", "score", "oracle", "valid", "has_route", "num_reactions"]
# Descriptor columns — keys of glue.metrics.dataset_metrics.descriptors_for_smiles.
DESCRIPTOR_COLUMNS: List[str] = [
    "mol_weight",
    "qed",
    "clogp",
    "h_donors",
    "h_acceptors",
    "heavy_atoms",
    "rotatable_bonds",
    "tpsa",
    "num_rings",
    "lipinski_pass",
    "ligand_efficiency",
]
STANDARD_COLUMNS: List[str] = REQUIRED_COLUMNS + _PROVENANCE_COLUMNS + DESCRIPTOR_COLUMNS

CANDIDATES_FILE = "candidates.csv"
ROUTES_FILE = "routes.jsonl"
MANIFEST_FILE = "manifest.json"


class CandidateDataset:
    """Generator-agnostic writer for the standard candidate dataset format.

    Accumulates candidate records in memory and rewrites the three files
    atomically on :meth:`flush` / :meth:`write` (so an in-progress dataset is
    always consistent and a crash mid-run leaves a valid prefix). At the scales we
    benchmark (hundreds to low millions of rows) a full rewrite per flush is cheap
    and removes any header/column-drift hazard.

    Typical use by a generator/adapter::

        ds = CandidateDataset(out_dir, generator="vae_bo", oracle="docking_6td3_gpu",
                              system="6td3", score_higher_is_better=False, seed=42)
        for smi, score in results:
            ds.add(smiles=smi, score=score)
        ds.write()

    Or as a context manager (writes on exit)::

        with CandidateDataset(out_dir, generator="rgfn", ...) as ds:
            ds.add(smiles=..., score=..., step=1, route=route_dict, extra={"vina_t2": ...})
    """

    def __init__(
        self,
        out_dir,
        generator: str,
        oracle: Optional[str] = None,
        system: Optional[str] = None,
        seed: Optional[int] = None,
        score_higher_is_better: Optional[bool] = None,
        score_units: Optional[str] = None,
        generator_version: Optional[str] = None,
        budget: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        notes: Optional[str] = None,
        created: Optional[str] = None,
        compute_descriptors: bool = True,
        schema_version: str = CANDIDATE_SCHEMA_VERSION,
    ):
        """
        Args:
            out_dir: dataset directory (created if absent).
            generator: name of the software that produced these candidates
                (e.g. ``"rgfn"``, ``"synflownet"``, ``"vae_bo"``).
            oracle: name of the scorer behind ``score`` (e.g. ``"docking_6td3_gpu"``).
            system: target system (e.g. ``"6td3"``, ``"seh"``).
            seed: RNG seed of the generating run (provenance).
            score_higher_is_better: orientation of ``score`` (docking ΔG/differential
                is lower-is-better → ``False``). Recorded so the harness ranks correctly.
            score_units: free-text units/definition of ``score`` (provenance).
            generator_version: version/commit of the generator (provenance).
            budget: oracle-call / compute budget dict (for equal-budget comparisons).
            source: pointer to the config/spec that produced the run.
            notes: free-text.
            created: ISO timestamp; if None, stamped at first write.
            compute_descriptors: fill the descriptor columns from SMILES via RDKit
                (set False to skip, e.g. on a minimal environment).
            schema_version: format version stamped into the manifest.
        """
        self.out_dir = Path(out_dir)
        self.generator = generator
        self.oracle = oracle
        self.system = system
        self.seed = seed
        self.score_higher_is_better = score_higher_is_better
        self.score_units = score_units
        self.generator_version = generator_version
        self.budget = budget
        self.source = source
        self.notes = notes
        self.created = created
        self.compute_descriptors = compute_descriptors
        self.schema_version = schema_version

        self._rows: List[Dict[str, Any]] = []
        self._routes: List[Dict[str, Any]] = []
        self._extra_columns: List[str] = []  # discovered extra keys, in first-seen order

    # ------------------------------------------------------------------- add
    def add(
        self,
        smiles: str,
        score: Optional[float] = None,
        step: Optional[int] = None,
        route: Optional[Dict[str, Any]] = None,
        candidate_id: Optional[str] = None,
        oracle: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add one candidate. Returns its ``candidate_id``.

        ``route`` is a structured route dict (as produced by
        ``glue.active_learning.route.extract_route``); its ``num_reactions`` is
        copied into the row and the route is written to ``routes.jsonl``. ``extra``
        holds any additional per-candidate columns (e.g. an oracle's pose
        breakdown), preserved verbatim.
        """
        cid = (
            candidate_id if candidate_id is not None else f"{self.generator}-{len(self._rows):06d}"
        )
        row: Dict[str, Any] = {
            "candidate_id": cid,
            "smiles": smiles,
            "generator": self.generator,
            "step": step,
            "score": score,
            "oracle": oracle if oracle is not None else self.oracle,
        }
        if self.compute_descriptors:
            desc = descriptors_for_smiles(smiles, score)
            row["valid"] = 0 if desc is None else 1
            for k in DESCRIPTOR_COLUMNS:
                row[k] = (desc or {}).get(k)
        else:
            row["valid"] = None

        has_route = route is not None and bool(route.get("steps"))
        row["has_route"] = int(bool(has_route))
        row["num_reactions"] = route.get("num_reactions") if route is not None else None

        if extra:
            for k, v in extra.items():
                if k not in self._extra_columns and k not in STANDARD_COLUMNS:
                    self._extra_columns.append(k)
                row[k] = v

        self._rows.append(row)
        if has_route:
            self._routes.append({"candidate_id": cid, **route})
        return cid

    def add_many(self, records: Iterable[Dict[str, Any]]) -> None:
        """Add a batch of candidate dicts (each forwarded to :meth:`add` by key)."""
        for rec in records:
            self.add(**rec)

    # ----------------------------------------------------------------- write
    def columns(self) -> List[str]:
        """The full, ordered CSV column list for this dataset."""
        return STANDARD_COLUMNS + self._extra_columns

    def manifest(self) -> Dict[str, Any]:
        """The dataset card written to ``manifest.json``."""
        n_valid = sum(1 for r in self._rows if r.get("valid") == 1)
        return {
            "schema_version": self.schema_version,
            "generator": self.generator,
            "generator_version": self.generator_version,
            "oracle": self.oracle,
            "score_higher_is_better": self.score_higher_is_better,
            "score_units": self.score_units,
            "system": self.system,
            "seed": self.seed,
            "budget": self.budget,
            "n_candidates": len(self._rows),
            "n_valid": n_valid,
            "has_routes": bool(self._routes),
            "columns": self.columns(),
            "created": self.created,
            "source": self.source,
            "notes": self.notes,
        }

    def write(self) -> Path:
        """Write/overwrite manifest.json, candidates.csv, and routes.jsonl atomically.

        Returns the dataset directory.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.created is None:
            from datetime import datetime, timezone

            self.created = datetime.now(timezone.utc).isoformat(timespec="seconds")

        cols = self.columns()
        self._atomic_write(
            self.out_dir / CANDIDATES_FILE,
            lambda fh: self._dump_csv(fh, cols),
        )
        if self._routes:
            self._atomic_write(
                self.out_dir / ROUTES_FILE,
                lambda fh: [fh.write(json.dumps(r) + "\n") for r in self._routes],
            )
        self._atomic_write(
            self.out_dir / MANIFEST_FILE,
            lambda fh: json.dump(self.manifest(), fh, indent=2),
        )
        return self.out_dir

    # flush is an alias used by incremental writers (e.g. per AL round).
    flush = write

    def _dump_csv(self, fh, cols: List[str]) -> None:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(self._rows)

    @staticmethod
    def _atomic_write(path: Path, writer_fn) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", newline="") as fh:
            writer_fn(fh)
        os.replace(tmp, path)

    # ------------------------------------------------------------- ctx manager
    def __enter__(self) -> "CandidateDataset":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.write()


# --------------------------------------------------------------------- ingest
def from_smiles_table(
    rows: Iterable[Dict[str, Any]],
    out_dir,
    generator: str,
    smiles_col: str = "smiles",
    score_col: Optional[str] = "score",
    step_col: Optional[str] = None,
    id_col: Optional[str] = None,
    **manifest_kwargs,
) -> Path:
    """Wrap an external generator's plain SMILES(+score) table into the standard.

    This is the on-ramp for a baseline that only emits a flat CSV/list of rows:
    point it at the rows and the column names, and you get a conformant dataset
    (descriptors auto-filled, manifest written). ``manifest_kwargs`` are passed to
    :class:`CandidateDataset` (oracle, system, seed, score_higher_is_better, …).

    Args:
        rows: an iterable of dicts (e.g. ``csv.DictReader`` over the generator's CSV).
        out_dir: destination dataset directory.
        generator: generator name to stamp on every row + the manifest.
        smiles_col / score_col / step_col / id_col: column names in ``rows``
            (``score_col``/``step_col``/``id_col`` may be None if absent).

    Returns the dataset directory.
    """
    ds = CandidateDataset(out_dir, generator=generator, **manifest_kwargs)
    for rec in rows:
        smi = rec.get(smiles_col)
        if not smi:
            continue
        score = _to_float(rec.get(score_col)) if score_col else None
        step = _to_int(rec.get(step_col)) if step_col else None
        cid = rec.get(id_col) if id_col else None
        ds.add(smiles=smi, score=score, step=step, candidate_id=cid)
    return ds.write()


# ----------------------------------------------------------------------- read
def read_candidate_dataset(path) -> Dict[str, Any]:
    """Read a standard candidate dataset directory.

    Returns ``{"manifest": dict, "candidates": List[dict], "routes": {cid: route}}``.
    The harness uses this to load any entrant's output uniformly.
    """
    path = Path(path)
    manifest = {}
    mpath = path / MANIFEST_FILE
    if mpath.exists():
        with open(mpath) as fh:
            manifest = json.load(fh)

    candidates: List[Dict[str, Any]] = []
    cpath = path / CANDIDATES_FILE
    if cpath.exists():
        with open(cpath, newline="") as fh:
            candidates = list(csv.DictReader(fh))

    routes: Dict[str, Any] = {}
    rpath = path / ROUTES_FILE
    if rpath.exists():
        with open(rpath) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    routes[r.get("candidate_id")] = r
    return {"manifest": manifest, "candidates": candidates, "routes": routes}


def validate_candidate_dataset(path) -> List[str]:
    """Return a list of human-readable conformance problems (empty == valid).

    Checks: the three files' presence, required columns, a recognised
    schema_version, and that route ``candidate_id``s reference real candidates.
    Intended as a fast gate the harness can run before trusting a dataset.
    """
    path = Path(path)
    issues: List[str] = []
    if not (path / CANDIDATES_FILE).exists():
        issues.append(f"missing {CANDIDATES_FILE}")
        return issues  # nothing else checkable
    data = read_candidate_dataset(path)
    manifest, candidates, routes = data["manifest"], data["candidates"], data["routes"]

    if not manifest:
        issues.append(f"missing/empty {MANIFEST_FILE}")
    elif manifest.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        issues.append(
            f"schema_version {manifest.get('schema_version')!r} != {CANDIDATE_SCHEMA_VERSION!r}"
        )

    if candidates:
        present = set(candidates[0].keys())
        missing = [c for c in REQUIRED_COLUMNS if c not in present]
        if missing:
            issues.append(f"candidates.csv missing required columns: {missing}")
        ids = {r.get("candidate_id") for r in candidates}
        orphan = [cid for cid in routes if cid not in ids]
        if orphan:
            issues.append(
                f"{len(orphan)} route(s) reference unknown candidate_id (e.g. {orphan[0]})"
            )
    else:
        issues.append("candidates.csv has no rows")
    return issues


def _to_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None
