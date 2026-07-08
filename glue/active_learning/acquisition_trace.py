"""Per-round oracle-call + top-molecule trace for the active-learning loop.

This is the substrate for the **top-k-vs-oracle-calls curve** (``[bengio2021gflownet]``
Fig. 7): the headline "does the learned generator find better molecules *per
expensive oracle call* than random acquisition?" plot the journals ask for
(``docs/RESEARCH_CONTEXT.md`` Objective 1). Two quantities must be captured *while
the loop runs* — they cannot be reconstructed from the final dataset alone:

  1. **Oracle-call budget spent** — how many molecules were submitted to the
     expensive oracle ``O`` by the end of each round (cumulative). In the
     active-learning setting this is the true cost unit and the x-axis of the
     curve. Counted as molecules *submitted* to ``O`` (docking failures still
     consumed a call), matching the paper's accounting of oracle evaluations.
  2. **Best molecules found so far, tied to that budget** — the running Top-K of
     the accumulated dataset ``D`` at each round, so every budget checkpoint has a
     recoverable "these were the best candidates after N oracle calls" (the
     y-axis, plus the SMILES of the single best so the actual molecule is
     recoverable without re-reading the per-round dataset dumps).

Written to ``<run_dir>/active_learning/oracle_calls.csv``, one row per round. Pure
observation — never feeds training. The aggregator
``validation/harness/acquisition_curve.py`` reads these across seeds x arms and
draws the curve.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


class AcquisitionTrace:
    """Append-only per-round record of oracle-call budget and best-so-far molecules.

    One instance per active-learning run; :meth:`record` is called once per round
    *after* the dataset has grown, and returns the row it wrote so the loop can
    forward the summary to its metrics logger.
    """

    COLUMNS = [
        "round",
        "acquisition",  # "policy" (learned GFN) or "random" (uniform baseline)
        "seed",
        "oracle_calls_round",  # molecules submitted to O this round
        "oracle_calls_cumulative",  # ... summed over all rounds so far (the x-axis)
        "n_labelled_round",  # of those, how many O actually scored (finite label)
        "dataset_size",  # |D| after this round's accumulation
        "top_k",  # k actually available (<= requested when |D| is small)
        "topk_mean",  # mean oracle label over the current Top-K (the y-axis)
        "topk_best",  # single best oracle label in D so far
        "topk_best_smiles",  # ... and the molecule that achieved it (recoverable)
    ]

    def __init__(
        self,
        csv_path,
        acquisition: str,
        higher_is_better: bool,
        top_k: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Args:
            csv_path: where to write ``oracle_calls.csv`` (truncated + header on init).
            acquisition: ``"policy"`` (learned RGFN acquisition) or ``"random"``
                (uniform-policy baseline) — the arm this run belongs to.
            higher_is_better: orientation of the oracle score, so "best" is picked
                from the correct end (our docking oracles are lower-is-better).
            top_k: size of the running Top-K summary (defaults to 100).
            seed: run RNG seed, recorded per row so the aggregator can group
                across seeds.
        """
        self.csv_path = Path(csv_path)
        self.acquisition = acquisition
        self.higher_is_better = higher_is_better
        self.top_k = top_k
        self.seed = seed
        self._cumulative = 0
        # Write the header up front so a run that dies mid-round-1 still leaves a
        # readable (if empty) trace.
        with open(self.csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(self.COLUMNS)

    @property
    def cumulative(self) -> int:
        """Total oracle calls spent so far (across all recorded rounds)."""
        return self._cumulative

    def record(
        self,
        rnd: int,
        oracle_calls_round: int,
        n_labelled_round: int,
        smiles: Sequence[str],
        labels: Sequence[float],
    ) -> Dict[str, object]:
        """Record one round and return the written row.

        Args:
            rnd: 1-based round number.
            oracle_calls_round: molecules submitted to ``O`` this round (the whole
                query batch, including any that failed to score).
            n_labelled_round: how many of those got a finite label.
            smiles / labels: the *full* accumulated dataset ``D`` after this round
                (parallel lists), used to compute the running Top-K.
        """
        self._cumulative += oracle_calls_round
        top = self._top_k(smiles, labels)
        topk_mean = sum(l for _, l in top) / len(top) if top else float("nan")
        topk_best, topk_best_smiles = (top[0][1], top[0][0]) if top else (float("nan"), "")
        row: Dict[str, object] = {
            "round": rnd,
            "acquisition": self.acquisition,
            "seed": self.seed if self.seed is not None else "",
            "oracle_calls_round": oracle_calls_round,
            "oracle_calls_cumulative": self._cumulative,
            "n_labelled_round": n_labelled_round,
            "dataset_size": len(smiles),
            "top_k": len(top),
            "topk_mean": topk_mean,
            "topk_best": topk_best,
            "topk_best_smiles": topk_best_smiles,
        }
        with open(self.csv_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.COLUMNS).writerow(row)
        return row

    def _top_k(self, smiles: Sequence[str], labels: Sequence[float]) -> List[Tuple[str, float]]:
        """Top-K ``(smiles, label)`` from ``D``, best-first for this oracle's sign.

        Independent of ``OracleLabeledDataset.top_k`` (which hardcodes ascending):
        here we honour ``higher_is_better`` so the trace is correct for a future
        higher-is-better oracle too. Non-finite labels are dropped.
        """
        pairs = [(s, l) for s, l in zip(smiles, labels) if l is not None and l == l]
        pairs.sort(key=lambda sl: sl[1], reverse=self.higher_is_better)
        return pairs[: self.top_k]
