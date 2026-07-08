"""Reconstruct a molecule's synthesis route from its RGFN trajectory.

The active-learning loop samples *trajectories* (``rgfn.api.trajectories.Trajectories``)
and, until now, kept only the terminal SMILES — throwing away **how** RGFN built
each molecule. But the entire value of a *reaction*-GFlowNet is that every molecule
comes with a synthesizable route by construction (``[koziarski2024rgfn]`` §3): a
starting building block plus an ordered list of reactions. To later verify that a
suggested glue's route is chemically plausible, we have to record that route at
sampling time. This module turns one trajectory's ``(states, actions)`` into a
structured, JSON-serialisable route.

The reaction-GFN state machine (see ``rgfn/gfns/reaction_gfn/api/reaction_api.py``)
emits these actions along a forward trajectory::

    ReactionAction0(fragment, idx)         pick the initial building block
    ReactionActionA(anchored_reaction,..)  choose a reaction (or None == stop)
    ReactionActionB(fragment, idx)         choose a fragment to react in
    ReactionActionC(input_molecule,         COMMIT: apply input_reaction to
                    input_reaction,         input_molecule + input_fragments,
                    input_fragments,        giving output_molecule
                    output_molecule)

The ``ReactionActionC`` is the only one that records a *committed* reaction with
its reactants and product, so a route is fully described by the single
``ReactionAction0`` (the seed building block) plus every ``ReactionActionC`` in
order. The A/B actions are intermediate selections subsumed by the C they lead to.
"""

from typing import Any, Dict, List

from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionAction0,
    ReactionActionC,
    ReactionStateTerminal,
)


def extract_route(states: List[Any], actions: List[Any]) -> Dict[str, Any]:
    """Build a structured synthesis route from one trajectory's states/actions.

    Args:
        states: the trajectory's state sequence (``trajectories._states_list[i]``).
        actions: the trajectory's action sequence (``trajectories._actions_list[i]``);
            ``len(actions) == len(states) - 1``.

    Returns:
        A JSON-serialisable dict::

            {
              "product_smiles": <terminal molecule SMILES, or None>,
              "num_reactions": <int, count of committed reaction steps>,
              "building_block": {"smiles": ..., "idx": ...} | None,
              "steps": [
                {
                  "step": 1,
                  "reaction_idx": <int>,
                  "reaction_smarts": "<left >> right>",
                  "reactant": "<input_molecule SMILES>",
                  "fragments": ["<fragment SMILES>", ...],
                  "product": "<output_molecule SMILES>"
                },
                ...
              ]
            }

        The route is reconstructed defensively: unrecognised action types are
        skipped, so a layout change upstream degrades to a shorter route rather
        than crashing the loop.
    """
    building_block = None
    steps: List[Dict[str, Any]] = []
    for action in actions:
        if isinstance(action, ReactionAction0):
            building_block = {
                "smiles": action.fragment.smiles,
                "idx": action.fragment.idx,
            }
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
    """One-line human-readable rendering of a route (for CSV provenance).

    Example::

        F(123) | r17: CCO.NC=O>>... -> CCOC=O | r4: ... -> <product>

    The structured JSONL written alongside is authoritative; this is a glanceable
    summary so a CSV row is legible without parsing JSON.
    """
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
