"""``OracleStepTimer`` — per-substep wall-clock timing *inside* an oracle call.

Companion to ``glue/active_learning/timing.PhaseTimer``, one level down. PhaseTimer
records how long the whole ``oracle_score`` phase took; this records where that
time went *within* a docking oracle's ``score()`` call — embedding, the Tier-2
conformational search, pose selection, the Tier-1 rescore — so we can say which
sub-step dominates before investing in a speed-up (e.g. GPU docking for the
Tier-2 search; see ``Logs/012``).

Design mirrors ``PhaseTimer``:
    - **Disabled by default** (``csv_path=None``): a silent no-op, so an oracle can
      always hold one and the loop *opts in* (``enable_step_timing``) only for
      oracles that implement the hook.
    - **Append-on-finish**: each sub-step is written the moment it ends, before the
      next begins, so a mid-step crash still leaves a complete record of what did
      finish (the lesson of experiment 009's SIGXCPU at the oracle).
    - Each row carries the round index and the molecule count for that sub-step, so
      seconds/molecule is recoverable per step (the Tier-2 search scales with
      molecule size — exactly what we want to watch).
"""

import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional


def _fmt(seconds: float) -> str:
    """Human-readable duration: ``45.2s`` or ``3m 05s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s:02d}s"


class OracleStepTimer:
    """Times named sub-steps of an oracle's per-round ``score()`` call.

    Construct with ``csv_path=None`` for a disabled (silent, no-file) timer; pass a
    path to record. Call :meth:`new_round` once at the top of each ``score()`` call,
    then wrap each sub-step in :meth:`step`.
    """

    def __init__(self, csv_path: Optional[Path] = None):
        """
        Args:
            csv_path: where to append ``(round, step, seconds, n_molecules)`` rows;
                ``None`` disables timing entirely (no print, no file).
        """
        self.csv_path = Path(csv_path) if csv_path else None
        self._round = 0
        # step -> cumulative seconds across all rounds (for an optional summary).
        self._totals: Dict[str, float] = {}
        if self.csv_path is not None and not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="") as fh:
                csv.writer(fh).writerow(["round", "step", "seconds", "n_molecules"])

    @property
    def enabled(self) -> bool:
        return self.csv_path is not None

    def new_round(self) -> int:
        """Advance to the next round; returns the new 1-based round index."""
        self._round += 1
        return self._round

    @contextmanager
    def step(self, name: str, n: int = 0):
        """Time a named sub-step processing ``n`` molecules; records even on raise."""
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._totals[name] = self._totals.get(name, 0.0) + elapsed
            per = f"  {elapsed / n:.2f}s/mol" if n else ""
            print(
                f"[dock] round {self._round}: {name} {_fmt(elapsed)} (n={n}){per}",
                flush=True,
            )
            with open(self.csv_path, "a", newline="") as fh:
                csv.writer(fh).writerow([self._round, name, f"{elapsed:.3f}", n])
