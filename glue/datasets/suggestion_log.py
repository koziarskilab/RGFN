"""Per-round provenance log of the molecules the active-learning loop suggests.

``OracleLabeledDataset`` is the *proxy-fit store*: canonical SMILES -> label, the
minimal thing the loop must accumulate to refit ``M`` (``[bengio2021gflownet]``
Alg. 1). It deliberately keeps nothing else. For analysis we want, for every
molecule the policy *suggested* in round ``i``: which round proposed it, the
synthesis route that builds it (the point of a *reaction* GFlowNet), and its
medchem descriptors + the oracle's score breakdown.

This sidecar records that — but rather than invent a bespoke layout, it writes the
**standard candidate-dataset format** (``glue.datasets.candidates``) so an
active-learning run is just another conformant entrant the benchmark harness can
read next to SynFlowNet / VAE-BO / FragGFN outputs. Under
``<run_dir>/active_learning/suggestions/`` it produces:

    manifest.json     # dataset card (generator="rgfn", oracle, system, seed, ...)
    candidates.csv    # standard table — one row per suggested molecule, all rounds
                      #   (the AL round is the standard ``step`` column)
    routes.jsonl      # structured synthesis route per molecule
    batch_metrics.csv # NON-standard analysis sidecar: one row per round of
                      #   set-level diversity / medchem / score-distribution metrics

Nothing here feeds training; it is pure observation, safe to extend or drop.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import gin

from glue.active_learning.route import route_to_str
from glue.datasets.candidates import CandidateDataset
from glue.metrics.dataset_metrics import batch_metrics


@gin.configurable()
class SuggestionLog:
    """Writer for per-round suggested-molecule provenance, in the standard format."""

    def __init__(
        self,
        run_dir: Optional[str] = None,
        generator: str = "rgfn",
        oracle_name: Optional[str] = None,
        system: Optional[str] = None,
        seed: Optional[int] = None,
        score_units: Optional[str] = None,
        source: Optional[str] = None,
        oracle_threshold: float = -2.0,
        oracle_higher_is_better: bool = False,
    ):
        """
        Args:
            run_dir: base run dir; files land in ``<run_dir>/active_learning/suggestions/``.
                If ``None``, the loop sets it from the trainer's run dir.
            generator: generator name stamped into the standard dataset (default
                ``"rgfn"`` — this loop is the RGFN entrant).
            oracle_name / system / seed / score_units / source: manifest provenance,
                forwarded to the standard :class:`~glue.datasets.candidates.CandidateDataset`.
            oracle_threshold: the "good glue" cutoff reported in batch metrics
                (default -2.0 — the 6TD3 differential mark from ``Logs/006``/011).
            oracle_higher_is_better: orientation of the score (recorded in the
                manifest and used for the threshold test).
        """
        self.generator = generator
        self.oracle_name = oracle_name
        self.system = system
        self.seed = seed
        self.score_units = score_units
        self.source = source
        self.oracle_threshold = oracle_threshold
        self.oracle_higher_is_better = oracle_higher_is_better
        self._dir: Optional[Path] = None
        self._candidates: Optional[CandidateDataset] = None
        if run_dir is not None:
            self.set_run_dir(run_dir)

    def set_run_dir(self, run_dir: str) -> None:
        self._dir = Path(run_dir) / "active_learning" / "suggestions"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._candidates = CandidateDataset(
            self._dir,
            generator=self.generator,
            oracle=self.oracle_name,
            system=self.system,
            seed=self.seed,
            score_higher_is_better=self.oracle_higher_is_better,
            score_units=self.score_units,
            source=self.source,
            notes="Active-learning query batches; the standard 'step' column is the AL round.",
        )

    # ------------------------------------------------------------------ logging
    def log_round(
        self,
        rnd: int,
        smiles: Sequence[str],
        routes: Sequence[Dict],
        labels: Sequence[float],
        details: Optional[Sequence[Optional[Dict]]] = None,
        reference_smiles: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Append one round's candidates (standard format) + batch metrics.

        Args:
            rnd: 1-based round number (becomes the standard ``step``).
            smiles: suggested SMILES (the query batch).
            routes: parallel structured routes from ``extract_route``.
            labels: parallel oracle scores (NaN/None where docking failed).
            details: parallel per-molecule oracle breakdown dicts (optional); their
                keys (e.g. vina_t2/cnnsc_t2) become preserved extra columns.
            reference_smiles: seed D_0 (for the novelty metric).

        Returns the batch-metrics dict so the loop can forward it to wandb.
        """
        if self._candidates is None:
            raise RuntimeError("SuggestionLog.set_run_dir must be called before log_round.")
        details = list(details) if details is not None else [None] * len(smiles)

        for smi, route, label, detail in zip(smiles, routes, labels, details):
            extra = {"route_str": route_to_str(route)}
            if detail:
                # Preserve the oracle's breakdown; 'dvina' is already the score.
                extra.update({k: v for k, v in detail.items() if k != "dvina"})
            self._candidates.add(smiles=smi, score=label, step=rnd, route=route, extra=extra)
        # Rewrite the standard files (atomic, crash-safe) after each round.
        self._candidates.flush()

        # Set-level batch metrics for this round (non-standard analysis sidecar).
        metrics = batch_metrics(
            list(smiles),
            labels=list(labels),
            reference_smiles=reference_smiles,
            oracle_threshold=self.oracle_threshold,
            oracle_higher_is_better=self.oracle_higher_is_better,
        )
        metrics = {"al_round": rnd, **metrics}
        self._append_csv(self._dir / "batch_metrics.csv", list(metrics.keys()), [metrics])
        return metrics

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _append_csv(path: Path, columns: List[str], rows: List[Dict]) -> None:
        exists = path.exists()
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerows(rows)
