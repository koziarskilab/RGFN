"""``glue.analysis`` — post-hoc analysis of a trained reaction-GFN.

The first capability is **hub-based late-stage diversification**: read a trained GFN,
find the high-flow *pre-terminal* intermediates (hubs), and select ``m`` of them to
diversify in parallel — each hub's shared scaffold split into ``k`` products by a single
diverging final reaction. Swappable strategies choose the hubs and the per-hub products;
a sweep runs many strategies over many trials into a tidy table; plan-level metrics give
the diversity / concurrency / cost axes of a Pareto front. See ``README.md``.

Pipeline (each stage is one pluggable piece):

    TrainedGFN.load(cfg, ckpt)            # loader.py  — model handle
      → sample_trajectories()             #             — forward samples (+ rewards)
      → build_hub_graph(traj)             # hub_graph.py — intermediates, flow, children
      → HubSelector.select(graph, m)      # hub_selectors.py — pick m hubs
      → ChildExpander.expand(hub)         # expanders.py — observed | enumerative
      → MoleculeSelector.select(., k)     # mol_selectors.py — pick k per hub
      → BatchPlan → plan_metrics(...)     # batch_plan.py / metrics.py
      → SweepRunner(...).run()            # sweep.py — trials × strategies → results.csv
      → pareto_front(rows, objectives)    # pareto.py — the frontier

This subpackage imports from ``glue`` and ``rgfn`` (never the reverse); it is analysis,
not part of the shipped in-loop pipeline, and is not gin-configurable (driven by
``scripts/analyze_gfn.py`` / a Python or JSON sweep spec), so it is intentionally NOT
imported by ``glue.registry``.
"""

from glue.analysis.batch_plan import BatchPlan, Lane
from glue.analysis.expanders import ChildExpander, EnumerativeExpander, ObservedExpander
from glue.analysis.hub_graph import Child, Hub, HubGraph, build_hub_graph
from glue.analysis.hub_selectors import (
    DiverseHubSelector,
    HighestChildRewardHubSelector,
    HighestExpectedRewardHubSelector,
    HighestFlowHubSelector,
    HighestTBFlowHubSelector,
    HubSelector,
    MostModesHubSelector,
)
from glue.analysis.loader import TrainedGFN
from glue.analysis.metrics import plan_metrics
from glue.analysis.mol_selectors import (
    MoleculeSelector,
    RandomKSelector,
    ScaffoldDiverseKSelector,
    TopKRewardDiverseSelector,
    TopKRewardSelector,
)
from glue.analysis.pareto import aggregate_trials, pareto_front
from glue.analysis.plot import plot_flow_agreement, plot_pareto, plot_sweep_fronts
from glue.analysis.select import run_selection
from glue.analysis.sweep import SweepRunner, SweepSpec
from glue.analysis.tb_flow import annotate_tb_flow, flow_agreement

__all__ = [
    "TrainedGFN",
    "build_hub_graph",
    "Hub",
    "HubGraph",
    "Child",
    "ChildExpander",
    "ObservedExpander",
    "EnumerativeExpander",
    "HubSelector",
    "HighestFlowHubSelector",
    "HighestTBFlowHubSelector",
    "MostModesHubSelector",
    "HighestExpectedRewardHubSelector",
    "HighestChildRewardHubSelector",
    "DiverseHubSelector",
    "MoleculeSelector",
    "TopKRewardSelector",
    "TopKRewardDiverseSelector",
    "ScaffoldDiverseKSelector",
    "RandomKSelector",
    "run_selection",
    "plan_metrics",
    "BatchPlan",
    "Lane",
    "SweepRunner",
    "SweepSpec",
    "pareto_front",
    "aggregate_trials",
    "plot_pareto",
    "plot_sweep_fronts",
    "plot_flow_agreement",
    "annotate_tb_flow",
    "flow_agreement",
]
