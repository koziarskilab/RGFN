"""Hub-selection strategies — *which* ``m`` pre-terminal intermediates to diversify.

Each strategy ranks the graph's candidate hubs and returns the top ``m``. They score
hubs from the **observed** signals collected during sampling (flow = visit count, and
the sampled children's SMILES/scores), so hub selection stays cheap — expensive
enumeration is deferred to the ``m`` winners at molecule-selection time. Add a strategy
by subclassing :class:`HubSelector` and registering it in ``glue.analysis.registry``.

Strategies provided:
    highest_flow            most sampled trajectories through the hub (the flow signal).
    most_modes              most distinct Tanimoto modes reachable one reaction away
                            (raw branching diversity of the diverging step).
    highest_expected_reward flow × mean child reward, or total reward mass — hubs that
                            carry a lot of high-reward flow.
    highest_child_reward    hub with the single best reachable child.

Any strategy can be wrapped in :class:`DiverseHubSelector` to additionally force the
chosen hubs to be dissimilar to each other (spread the parallel lanes across scaffold
space, not m variations of one intermediate).
"""

from __future__ import annotations

import abc
from statistics import mean
from typing import List, Optional

from glue.analysis import similarity
from glue.analysis.hub_graph import Hub, HubGraph


class HubSelector(abc.ABC):
    """Rank candidate hubs and return the top ``m``."""

    name: str = "base"

    def __init__(self, min_depth: int = 1, max_depth: Optional[int] = None, min_children: int = 1):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_children = min_children

    def candidates(self, graph: HubGraph) -> List[Hub]:
        return graph.candidate_hubs(
            min_depth=self.min_depth, max_depth=self.max_depth, min_children=self.min_children
        )

    @abc.abstractmethod
    def score(self, hub: Hub, graph: HubGraph) -> float:
        """Higher == more preferred. (All built-ins rank by descending score.)"""

    def select(self, graph: HubGraph, m: int, gfn=None) -> List[Hub]:
        ranked = sorted(self.candidates(graph), key=lambda h: self.score(h, graph), reverse=True)
        return ranked[:m]


class HighestFlowHubSelector(HubSelector):
    """Top hubs by empirical flow (trajectory visit count)."""

    name = "highest_flow"

    def score(self, hub: Hub, graph: HubGraph) -> float:
        return float(hub.visit_count)


class MostModesHubSelector(HubSelector):
    """Top hubs by number of distinct Tanimoto modes reachable one reaction away.

    Optionally restrict the children counted to the top ``reward_quantile`` by score
    first, so a hub is rewarded for diverse *good* products, not diverse junk.
    """

    name = "most_modes"

    def __init__(
        self,
        similarity_threshold: float = similarity.MODE_SIMILARITY_THRESHOLD,
        reward_quantile: Optional[float] = None,
        min_children: int = 2,
        **kwargs,
    ):
        super().__init__(min_children=min_children, **kwargs)
        self.similarity_threshold = similarity_threshold
        self.reward_quantile = reward_quantile

    def _kept_smiles(self, hub: Hub, graph: HubGraph) -> List[str]:
        children = hub.children_list()
        if self.reward_quantile is not None:
            scored = [c for c in children if c.score is not None]
            if scored:
                scored.sort(key=lambda c: c.score, reverse=graph.higher_is_better)
                keep_n = max(1, int(round(len(scored) * self.reward_quantile)))
                children = scored[:keep_n]
        return [c.smiles for c in children]

    def score(self, hub: Hub, graph: HubGraph) -> float:
        modes = similarity.count_modes(self._kept_smiles(hub, graph), self.similarity_threshold)
        # Tie-break by flow so equally-modal hubs prefer the higher-flow one.
        return modes + min(hub.visit_count, 999) / 1000.0


class HighestExpectedRewardHubSelector(HubSelector):
    """Top hubs by flow-weighted reward — where the good, high-flow mass concentrates.

    ``aggregate``:
        "mean"       flow × mean child reward (quality per unit flow)
        "sum"        flow × sum child reward
        "total_mass" sum of child rewards (ignores flow; total reachable reward)
    Scores are made orientation-agnostic (lower-is-better proxies are negated) so
    "higher score == better" holds for the ranking.
    """

    name = "highest_expected_reward"

    def __init__(self, aggregate: str = "mean", **kwargs):
        super().__init__(**kwargs)
        assert aggregate in ("mean", "sum", "total_mass")
        self.aggregate = aggregate

    def score(self, hub: Hub, graph: HubGraph) -> float:
        vals = [c.score for c in hub.children_list() if c.score is not None]
        if not vals:
            return float("-inf")
        oriented = vals if graph.higher_is_better else [-v for v in vals]
        if self.aggregate == "mean":
            return hub.visit_count * mean(oriented)
        if self.aggregate == "sum":
            return hub.visit_count * sum(oriented)
        return sum(oriented)  # total_mass


class HighestChildRewardHubSelector(HubSelector):
    """Top hubs by their single best reachable child (peak, not breadth)."""

    name = "highest_child_reward"

    def score(self, hub: Hub, graph: HubGraph) -> float:
        vals = [c.score for c in hub.children_list() if c.score is not None]
        if not vals:
            return float("-inf")
        best = max(vals) if graph.higher_is_better else min(vals)
        return best if graph.higher_is_better else -best


class DiverseHubSelector(HubSelector):
    """Decorator: take the base strategy's ranking but greedily drop hubs too similar
    to already-chosen ones, so the ``m`` parallel lanes cover distinct scaffolds."""

    name = "diverse"

    def __init__(self, base: HubSelector, similarity_threshold: float = 0.6):
        super().__init__(
            min_depth=base.min_depth, max_depth=base.max_depth, min_children=base.min_children
        )
        self.base = base
        self.similarity_threshold = similarity_threshold
        self.name = f"diverse[{base.name}]"

    def score(self, hub: Hub, graph: HubGraph) -> float:
        return self.base.score(hub, graph)

    def select(self, graph: HubGraph, m: int, gfn=None) -> List[Hub]:
        ranked = sorted(
            self.candidates(graph), key=lambda h: self.base.score(h, graph), reverse=True
        )
        return similarity.greedy_diverse_select(
            ranked,
            k=m,
            get_smiles=lambda h: h.smiles,
            similarity_threshold=self.similarity_threshold,
        )
