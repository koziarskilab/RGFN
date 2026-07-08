"""Synthesis-cost model for a batch plan — the ``cost`` axis of the Pareto front.

This turns a molecule's **route** (the ``glue.active_learning.route`` dict every
reaction-GFN candidate carries) into a number, using the per-block prices and
per-reaction yields already annotated on a ``glue.chemistry.ChemLibrary`` (see
``docs/CHEM_LIBRARY_FORMAT.md``). It is deliberately small and swappable: the whole
model is two functions, so a different cost story (real catalog prices, step-count
only, labour, …) is a drop-in replacement.

The point of hub-based late-stage diversification is that a shared intermediate is
synthesized **once** and split into many parallel final steps. So the plan-level cost
has two regimes we can compare directly:

- **independent** — every selected molecule is made from scratch (its full route).
- **hub-batched** — each lane pays its route-to-hub *once*, then only the incremental
  cost of each product's *final* diversifying step.

The gap between them is the saving hub-batching buys, and it is what makes "diversity
vs. cost" a meaningful trade rather than "diversity vs. molecule count".

Cost model (documented, first cut — tune via the kwargs):
    materials  = Σ price(building block) + Σ price(fragment added at each step)
    per_reaction_cost = a flat effort charge added per committed reaction step
    yield_adjusted    = if True, divide the running material cost by each step's yield
                        (a low-yield step wastes upstream material, so it costs more)

All prices/yields fall back to the library's ``default_cost`` / ``default_yield`` when
a block/reaction is unpriced, so an unpriced library still produces consistent
*relative* numbers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from glue.chemistry.library import ChemLibrary


def _bb_smiles(route: Optional[Dict[str, Any]]) -> Optional[str]:
    bb = (route or {}).get("building_block")
    return bb.get("smiles") if bb else None


def route_cost(
    route: Optional[Dict[str, Any]],
    library: ChemLibrary,
    per_reaction_cost: float = 1.0,
    yield_adjusted: bool = False,
) -> float:
    """Total cost to make one molecule from scratch, following its full route.

    Sums the seed building block plus every fragment added in every step, adds a flat
    ``per_reaction_cost`` per step, and (optionally) inflates by the compounded inverse
    yield. Returns ``0.0`` for an empty/None route (nothing to price).
    """
    if not route:
        return 0.0
    cost = 0.0
    bb = _bb_smiles(route)
    if bb:
        cost += library.cost_of(bb)
    running = cost
    for step in route.get("steps", []):
        for frag in step.get("fragments", []):
            add = library.cost_of(frag)
            cost += add
            running += add
        cost += per_reaction_cost
        running += per_reaction_cost
        if yield_adjusted:
            y = library.yield_of(step.get("reaction_smarts", ""))
            if y and y > 0:
                extra = running / y - running
                cost += extra
                running += extra
    return cost


def final_step_cost(
    route: Optional[Dict[str, Any]],
    library: ChemLibrary,
    per_reaction_cost: float = 1.0,
    yield_adjusted: bool = False,
) -> float:
    """Incremental cost of a molecule's *last* route step only.

    This is what a product costs *beyond* its shared intermediate in the hub-batched
    regime: the fragments consumed in the diverging final reaction plus one reaction's
    effort charge. A molecule whose route has no steps (a bare building block) costs
    ``0.0`` incrementally.
    """
    if not route:
        return 0.0
    steps = route.get("steps", [])
    if not steps:
        return 0.0
    last = steps[-1]
    cost = per_reaction_cost
    for frag in last.get("fragments", []):
        cost += library.cost_of(frag)
    if yield_adjusted:
        y = library.yield_of(last.get("reaction_smarts", ""))
        if y and y > 0:
            cost = cost / y
    return cost
