"""Plan-level metrics — the axes a Pareto front is drawn on.

Given a :class:`~glue.analysis.batch_plan.BatchPlan`, compute a **flat dict** of
metrics: the diversity of the whole selected set, the concurrency (number of lanes),
the reward distribution, and — if a priced :class:`~glue.chemistry.library.ChemLibrary`
is supplied — the synthesis cost under the hub-batched vs. independent regimes. One such
dict per strategy configuration per trial is exactly one row of the sweep's tidy table,
and the raw material for ``glue.analysis.pareto``.

Diversity/medchem reuse ``glue.metrics.dataset_metrics.batch_metrics`` verbatim, so the
numbers match every other place the project reports diversity.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Optional

from glue.analysis import cost as cost_model
from glue.analysis.batch_plan import BatchPlan
from glue.chemistry.library import ChemLibrary
from glue.metrics.dataset_metrics import batch_metrics, internal_diversity


def plan_metrics(
    plan: BatchPlan,
    higher_is_better: bool,
    library: Optional[ChemLibrary] = None,
    per_reaction_cost: float = 1.0,
    yield_adjusted: bool = False,
    reference_smiles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute the Pareto-axis metrics for one plan.

    Args:
        plan: the batch plan (m lanes).
        higher_is_better: proxy orientation (for reward summaries + threshold tests).
        library: priced library; if given, the cost axis is computed. If None, cost
            columns are omitted.
        per_reaction_cost / yield_adjusted: passed to the cost model.
        reference_smiles: optional set (e.g. seed D_0) for a novelty column.

    Returns a flat ``{metric: value}`` dict.
    """
    smiles = plan.all_smiles()
    scores = plan.all_scores()
    valid_scores = [s for s in scores if s is not None and s == s]

    # --- diversity / medchem / reward distribution (reuse the standard metrics) ------
    m: Dict[str, Any] = batch_metrics(
        smiles,
        labels=scores,
        reference_smiles=reference_smiles,
        oracle_higher_is_better=higher_is_better,
    )

    # --- concurrency ------------------------------------------------------------------
    per_lane_counts = [lane.n for lane in plan.lanes]
    m["concurrency"] = plan.concurrency
    m["n_molecules"] = len(smiles)
    m["molecules_per_lane_mean"] = mean(per_lane_counts) if per_lane_counts else 0.0
    m["molecules_per_lane_min"] = min(per_lane_counts) if per_lane_counts else 0
    m["molecules_per_lane_max"] = max(per_lane_counts) if per_lane_counts else 0

    # mean within-lane diversity: how varied each parallel batch is on its own.
    lane_divs = [
        internal_diversity([c.smiles for c in lane.children]) for lane in plan.lanes if lane.n >= 2
    ]
    lane_divs = [d for d in lane_divs if d == d]  # drop NaN
    m["within_lane_diversity_mean"] = mean(lane_divs) if lane_divs else float("nan")

    # --- reward summary (oriented "best") --------------------------------------------
    if valid_scores:
        m["reward_mean"] = mean(valid_scores)
        m["reward_best"] = max(valid_scores) if higher_is_better else min(valid_scores)

    # --- cost axis (optional) --------------------------------------------------------
    if library is not None:
        hub_batched = 0.0
        independent = 0.0
        for lane in plan.lanes:
            if lane.hub.route_to_hub is not None:
                hub_batched += cost_model.route_cost(
                    lane.hub.route_to_hub, library, per_reaction_cost, yield_adjusted
                )
            for child in lane.children:
                hub_batched += cost_model.final_step_cost(
                    child.route, library, per_reaction_cost, yield_adjusted
                )
                independent += cost_model.route_cost(
                    child.route, library, per_reaction_cost, yield_adjusted
                )
        n = max(1, len(smiles))
        m["cost_hub_batched"] = hub_batched
        m["cost_independent"] = independent
        m["cost_saving"] = independent - hub_batched
        m["cost_saving_ratio"] = (independent - hub_batched) / independent if independent else 0.0
        m["cost_per_molecule"] = hub_batched / n
    return m
