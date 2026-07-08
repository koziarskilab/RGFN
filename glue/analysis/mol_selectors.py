"""Molecule-selection strategies — *which* ``k`` products to take from a chosen hub.

Given a hub's diversification set (its children, from either expander), pick the ``k``
that go into that lane's batch. This is the within-hub half of the trade-off: pure
reward vs. reward-with-spread. Add a strategy by subclassing :class:`MoleculeSelector`
and registering it in ``glue.analysis.registry``.

Strategies provided:
    top_k_reward          the ``k`` highest-reward children (may cluster together).
    top_k_reward_diverse  greedy: highest reward first, skip near-duplicates (Tanimoto
                          >= threshold to something already chosen) — "top-k but spread".
    scaffold_diverse_k    the best child per distinct Murcko scaffold, until ``k``.
    random_k              a random ``k`` (baseline / control).
"""

from __future__ import annotations

import abc
import random
from typing import List, Optional

from glue.analysis import similarity
from glue.analysis.hub_graph import Child
from glue.metrics.dataset_metrics import murcko_scaffold


class MoleculeSelector(abc.ABC):
    """Pick ``k`` children from a hub's diversification set."""

    name: str = "base"

    @abc.abstractmethod
    def select(
        self, children: List[Child], k: int, higher_is_better: bool, gfn=None
    ) -> List[Child]:
        ...

    @staticmethod
    def _sorted_by_reward(children: List[Child], higher_is_better: bool) -> List[Child]:
        scored = [c for c in children if c.score is not None]
        scored.sort(key=lambda c: c.score, reverse=higher_is_better)
        # children without a score go last (they were never evaluated)
        return scored + [c for c in children if c.score is None]


class TopKRewardSelector(MoleculeSelector):
    """The ``k`` highest-reward children."""

    name = "top_k_reward"

    def select(self, children, k, higher_is_better, gfn=None):
        return self._sorted_by_reward(children, higher_is_better)[:k]


class TopKRewardDiverseSelector(MoleculeSelector):
    """Reward-ranked, then greedily de-duplicated by Tanimoto (top-k but spread out)."""

    name = "top_k_reward_diverse"

    def __init__(self, similarity_threshold: float = similarity.MODE_SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold

    def select(self, children, k, higher_is_better, gfn=None):
        ordered = self._sorted_by_reward(children, higher_is_better)
        return similarity.greedy_diverse_select(
            ordered,
            k=k,
            get_smiles=lambda c: c.smiles,
            similarity_threshold=self.similarity_threshold,
        )


class ScaffoldDiverseKSelector(MoleculeSelector):
    """Best-scoring child of each distinct Murcko scaffold, until ``k`` scaffolds."""

    name = "scaffold_diverse_k"

    def select(self, children, k, higher_is_better, gfn=None):
        ordered = self._sorted_by_reward(children, higher_is_better)
        seen: set = set()
        out: List[Child] = []
        for c in ordered:
            scaf = murcko_scaffold(c.smiles)
            if scaf is None or scaf in seen:
                continue
            seen.add(scaf)
            out.append(c)
            if len(out) >= k:
                break
        return out


class RandomKSelector(MoleculeSelector):
    """A random ``k`` (baseline that ignores reward and similarity)."""

    name = "random_k"

    def __init__(self, seed: Optional[int] = 0):
        self.seed = seed

    def select(self, children, k, higher_is_better, gfn=None):
        rng = random.Random(self.seed)
        pool = list(children)
        rng.shuffle(pool)
        return pool[:k]
