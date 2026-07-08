"""Reconstruct a molecule's synthesis route from its SCENT trajectory.

A self-contained copy of ``glue.active_learning.route`` for the ``scent`` conda env
(importing ``glue`` there would pull in our ``rgfn``/oracle stack, which isn't
installed in that env). SCENT is a fork of RGFN, so its reaction state machine emits
the *same* actions at the *same* import path — the logic below is identical to
``glue.active_learning.route``; only the package the symbols resolve from differs
(SCENT's installed ``rgfn`` here vs. our repo-local ``rgfn/`` there).

SCENT, like RGFN, is synthesizable by construction: every molecule carries a route
(a building block + an ordered list of committed reactions). We record it at
sampling time so the standard candidate dataset gets ``has_route=1`` + ``routes.jsonl``
— the differentiator vs. the non-synthesizable FragGFN entrant.
"""

from typing import Any, Dict, List

from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionAction0,
    ReactionActionC,
    ReactionStateTerminal,
)


def extract_route(states: List[Any], actions: List[Any]) -> Dict[str, Any]:
    """Build a structured synthesis route from one trajectory's states/actions.

    Returns a JSON-serialisable dict with ``product_smiles``, ``num_reactions``,
    ``building_block`` and an ordered ``steps`` list (each step: reaction idx +
    SMARTS, reactant, fragments, product). Reconstructed defensively: unrecognised
    action types are skipped, so an upstream layout change degrades to a shorter
    route rather than crashing the loop.
    """
    building_block = None
    steps: List[Dict[str, Any]] = []
    for action in actions:
        if isinstance(action, ReactionAction0):
            building_block = {"smiles": action.fragment.smiles, "idx": action.fragment.idx}
        elif isinstance(action, ReactionActionC):
            steps.append(
                {
                    "step": len(steps) + 1,
                    "reaction_idx": action.input_reaction.idx,
                    "reaction_smarts": action.input_reaction.reaction,
                    "reactant": action.input_molecule.smiles,
                    "fragments": [f.smiles for f in action.input_fragments],
                    "product": action.output_molecule.smiles,
                }
            )

    last = states[-1] if states else None
    product_smiles = last.molecule.smiles if isinstance(last, ReactionStateTerminal) else None
    return {
        "product_smiles": product_smiles,
        "num_reactions": len(steps),
        "building_block": building_block,
        "steps": steps,
    }


def route_to_str(route: Dict[str, Any]) -> str:
    """One-line human-readable rendering of a route (for CSV provenance)."""
    bb = route.get("building_block")
    head = (
        f"F{bb['idx']}:{bb['smiles']}"
        if bb and bb.get("idx") is not None
        else (bb["smiles"] if bb else "?")
    )
    parts = [head]
    for s in route.get("steps", []):
        parts.append(f"r{s['reaction_idx']} -> {s['product']}")
    return " | ".join(parts)
