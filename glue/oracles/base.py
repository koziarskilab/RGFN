"""The oracle interface ``O`` for the active-learning loop.

In ``[bengio2021gflownet]`` Alg. 1 the *oracle* ``O`` is the expensive, trusted
scorer queried only on the per-round query batch; its labels enter training only
by retraining the proxy ``M`` (never as a direct RGFN reward). Here ``O`` is our
docking neosubstrate differential today, and an MD-stability score later (see
``docs/RESEARCH_CONTEXT.md``).

This module defines the thin, modular seam that lets the loop stay oracle-agnostic:
any scorer that maps a list of SMILES to a list of floats can drive the loop.
Concrete oracles (``Docking6TD3Oracle``, ``MockGlueOracle``, future MD oracles)
subclass :class:`GlueOracle`; the loop only ever sees this interface.
"""

from abc import ABC, abstractmethod
from typing import List


class GlueOracle(ABC):
    """Expensive, trusted scorer ``O`` over SMILES.

    Implementations should be self-contained science (docking, MD, ...), free of
    RGFN plumbing — exactly mirroring the ``glue.oracles`` vs ``glue.proxies``
    split documented in ``glue/oracles/__init__.py``.
    """

    #: Human-readable name, used in logs and dataset provenance.
    name: str = "glue_oracle"

    #: Whether a higher score means a better glue. The learned proxy reads this
    #: so the reward sign is handled consistently end-to-end.
    higher_is_better: bool = True

    @abstractmethod
    def score(self, smiles: List[str]) -> List[float]:
        """Score a batch of SMILES.

        Args:
            smiles: list of SMILES strings for the query batch ``B``.

        Returns:
            One float per input SMILES, in the same order. Implementations should
            return ``float('nan')`` (not raise) for molecules they fail to score,
            so the loop can drop them when refitting the proxy.
        """
        ...
