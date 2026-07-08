"""``run_selection`` — one end-to-end selection for a single strategy configuration.

Ties the pieces together: from a built :class:`~glue.analysis.hub_graph.HubGraph`,
pick ``m`` hubs (a :class:`~glue.analysis.hub_selectors.HubSelector`), expand each hub's
children (a :class:`~glue.analysis.expanders.ChildExpander`), pick ``k`` products per hub
(a :class:`~glue.analysis.mol_selectors.MoleculeSelector`), assemble a
:class:`~glue.analysis.batch_plan.BatchPlan`, and compute its
:func:`~glue.analysis.metrics.plan_metrics`.

Enumeration (if the expander is enumerative) touches only the ``m`` selected hubs, not
the whole graph — hub *ranking* uses the cheap observed signals, so the expensive step
is paid only for the winners.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from glue.analysis.batch_plan import BatchPlan, Lane
from glue.analysis.expanders import ChildExpander, ObservedExpander
from glue.analysis.hub_graph import HubGraph
from glue.analysis.hub_selectors import HubSelector
from glue.analysis.metrics import plan_metrics
from glue.analysis.mol_selectors import MoleculeSelector
from glue.chemistry.library import ChemLibrary


def run_selection(
    graph: HubGraph,
    gfn,
    hub_selector: HubSelector,
    mol_selector: MoleculeSelector,
    m: int,
    k: int,
    expander: Optional[ChildExpander] = None,
    library: Optional[ChemLibrary] = None,
    per_reaction_cost: float = 1.0,
    yield_adjusted: bool = False,
    reference_smiles: Optional[List[str]] = None,
) -> Tuple[BatchPlan, Dict[str, Any]]:
    """Run one configuration and return ``(plan, metrics)``.

    Args:
        graph: the hub graph for this trial.
        gfn: the loaded ``TrainedGFN`` (needed for enumeration/scoring).
        hub_selector / mol_selector: the two strategies.
        m: number of hubs (lanes).
        k: products per hub.
        expander: how to get each hub's children (default: observed-only).
        library / per_reaction_cost / yield_adjusted: cost-axis inputs (optional).
        reference_smiles: optional novelty reference.
    """
    expander = expander or ObservedExpander()
    hubs = hub_selector.select(graph, m, gfn)

    lanes: List[Lane] = []
    for lane_id, hub in enumerate(hubs):
        children = expander.expand(hub, gfn)
        chosen = mol_selector.select(children, k, graph.higher_is_better, gfn)
        lanes.append(Lane(lane_id=lane_id, hub=hub, children=chosen))

    plan = BatchPlan(
        lanes=lanes,
        meta={
            "hub_selector": hub_selector.name,
            "mol_selector": mol_selector.name,
            "expander": expander.name,
            "m": m,
            "k": k,
        },
    )
    metrics = plan_metrics(
        plan,
        higher_is_better=graph.higher_is_better,
        library=library,
        per_reaction_cost=per_reaction_cost,
        yield_adjusted=yield_adjusted,
        reference_smiles=reference_smiles,
    )
    return plan, metrics
