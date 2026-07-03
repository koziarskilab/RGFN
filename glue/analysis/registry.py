"""Name → class registries so strategies are string-configurable from a sweep spec / CLI.

This is what makes the pipeline "easily expandable": a new strategy is one subclass plus
one line here, after which it is referable by name from ``scripts/analyze_gfn.py`` flags,
a JSON spec, or a submit script — no other file changes.

    hub selector   : registry.build_hub_selector("most_modes", {"reward_quantile": 0.5})
    mol selector   : registry.build_mol_selector("top_k_reward_diverse", {"similarity_threshold": 0.7})
    child expander : registry.build_expander("enumerative", {"max_fragments_per_pattern": 50})
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from glue.analysis.expanders import ChildExpander, EnumerativeExpander, ObservedExpander
from glue.analysis.hub_selectors import (
    HighestChildRewardHubSelector,
    HighestExpectedRewardHubSelector,
    HighestFlowHubSelector,
    HighestTBFlowHubSelector,
    HubSelector,
    MostModesHubSelector,
)
from glue.analysis.mol_selectors import (
    MoleculeSelector,
    RandomKSelector,
    ScaffoldDiverseKSelector,
    TopKRewardDiverseSelector,
    TopKRewardSelector,
)

HUB_SELECTORS: Dict[str, type] = {
    HighestFlowHubSelector.name: HighestFlowHubSelector,
    HighestTBFlowHubSelector.name: HighestTBFlowHubSelector,
    MostModesHubSelector.name: MostModesHubSelector,
    HighestExpectedRewardHubSelector.name: HighestExpectedRewardHubSelector,
    HighestChildRewardHubSelector.name: HighestChildRewardHubSelector,
}

MOL_SELECTORS: Dict[str, type] = {
    TopKRewardSelector.name: TopKRewardSelector,
    TopKRewardDiverseSelector.name: TopKRewardDiverseSelector,
    ScaffoldDiverseKSelector.name: ScaffoldDiverseKSelector,
    RandomKSelector.name: RandomKSelector,
}

EXPANDERS: Dict[str, type] = {
    ObservedExpander.name: ObservedExpander,
    EnumerativeExpander.name: EnumerativeExpander,
}


def build_hub_selector(name: str, kwargs: Optional[Dict[str, Any]] = None) -> HubSelector:
    return _build(HUB_SELECTORS, name, kwargs, "hub selector")


def build_mol_selector(name: str, kwargs: Optional[Dict[str, Any]] = None) -> MoleculeSelector:
    return _build(MOL_SELECTORS, name, kwargs, "molecule selector")


def build_expander(name: str, kwargs: Optional[Dict[str, Any]] = None) -> ChildExpander:
    return _build(EXPANDERS, name, kwargs, "expander")


def _build(table: Dict[str, type], name: str, kwargs: Optional[Dict[str, Any]], kind: str):
    if name not in table:
        raise KeyError(f"unknown {kind} {name!r}; available: {sorted(table)}")
    return table[name](**(kwargs or {}))
