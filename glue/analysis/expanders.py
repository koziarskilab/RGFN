"""Child expanders — how a hub's one-reaction-away terminal children are obtained.

This is the biggest cost/completeness knob in the pipeline:

* :class:`ObservedExpander` (cheap, default) returns the children the policy actually
  *sampled* through the hub. No model or proxy calls — it just reads what
  ``build_hub_graph`` already collected. Flow-faithful (it *is* the realized
  diversification), but limited to what the policy explored.

* :class:`EnumerativeExpander` (exhaustive, opt-in) drives the ``ReactionEnv``'s own
  forward transitions to enumerate **every** terminal product reachable from the hub in
  one reaction — the full late-stage-diversification library — then scores each with the
  proxy. This can find high-reward analogs the policy under-sampled, at the price of
  many proxy calls per hub. Best for cheap proxies (sEH MPNN) and/or a handful of
  selected hubs; avoid it with an expensive in-loop oracle.

Because it reuses the env's action spaces, the enumerated set is exactly the reachable
one — same chemistry the GFN samples from, no hand-rolled reaction application.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional

from glue.analysis.hub_graph import Child, Hub
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionActionSpaceA,
    ReactionActionSpaceEarlyTerminate,
    ReactionStateB,
    ReactionStateC,
    ReactionStateTerminal,
)


class ChildExpander(abc.ABC):
    """Given a hub (and a loaded model), return its 1-reaction-away terminal children."""

    name: str = "base"

    @abc.abstractmethod
    def expand(self, hub: Hub, gfn) -> List[Child]:  # gfn: TrainedGFN
        ...


class ObservedExpander(ChildExpander):
    """Return the children observed during sampling (no extra computation)."""

    name = "observed"

    def expand(self, hub: Hub, gfn) -> List[Child]:
        return hub.children_list()


class EnumerativeExpander(ChildExpander):
    """Enumerate + proxy-score all 1-reaction products reachable from the hub."""

    name = "enumerative"

    def __init__(
        self,
        max_fragments_per_pattern: Optional[int] = None,
        max_children: Optional[int] = None,
        include_observed: bool = True,
        score_batch_size: int = 256,
        verbose: bool = True,
    ):
        """
        Args:
            max_fragments_per_pattern: cap the fragment choices explored at each
                ``ReactionStateB`` branch (bounds combinatorial blow-up for
                multi-fragment reactions). ``None`` = no cap.
            max_children: stop after this many distinct products (across all reactions).
            include_observed: union the enumerated set with the sampled children so
                nothing the policy found is lost.
            score_batch_size: proxy-scoring batch size.
            verbose: print a one-line warning whenever a cap actually truncated the set
                (no silent caps — a bounded library must announce itself).
        """
        self.max_fragments_per_pattern = max_fragments_per_pattern
        self.max_children = max_children
        self.include_observed = include_observed
        self.score_batch_size = score_batch_size
        self.verbose = verbose
        # Enumeration is deterministic in (molecule, depth) under a frozen proxy, so the
        # enumerated set for a hub key can be cached and reused across sweep combos/trials
        # (the same hub selected by different strategies is enumerated once). Observed
        # children are merged fresh per call, so per-trial sampling is not cached away.
        self._enum_cache: Dict[Any, List[Child]] = {}

    def expand(self, hub: Hub, gfn) -> List[Child]:
        if hub.state is None:
            return hub.children_list()  # cannot enumerate without the real state object
        enumerated = self._enum_cache.get(hub.key)
        if enumerated is None:
            enumerated = self._enumerate(hub, gfn)
            self._enum_cache[hub.key] = enumerated
        if not self.include_observed:
            return list(enumerated)
        merged: Dict[str, Child] = dict(hub.observed_children)
        for child in enumerated:
            merged[child.smiles] = child  # enumerated score/route take precedence
        return list(merged.values())

    def _enumerate(self, hub: Hub, gfn) -> List[Child]:
        """Enumerate + proxy-score every 1-reaction product reachable from the hub."""
        env = gfn.env
        state = hub.state
        depth = hub.num_reactions
        hub_steps = (hub.route_to_hub or {}).get("steps", [])
        bb = (hub.route_to_hub or {}).get("building_block")

        fas = env.get_forward_action_spaces([state])[0]
        if not isinstance(fas, ReactionActionSpaceA):
            return []  # hub cannot take another reaction; expand() still returns observed

        # smiles -> (Molecule, final_step dict)
        products: Dict[str, Any] = {}
        truncated = False

        for a_idx in fas.get_possible_actions_indices():
            action_a = fas.get_action_at_idx(a_idx)
            if action_a.anchored_reaction is None:
                continue  # the STOP action; not a diversifying reaction
            reaction = action_a.anchored_reaction
            start = env.apply_forward_actions([state], [action_a])[0]

            # DFS over fragment choices (B) down to committed products (C).
            stack = [(start, [])]  # (state, fragment_smiles_so_far)
            while stack:
                s, frags = stack.pop()
                if isinstance(s, ReactionStateB):
                    fb = env.get_forward_action_spaces([s])[0]
                    if isinstance(fb, ReactionActionSpaceEarlyTerminate):
                        continue
                    idxs = fb.get_possible_actions_indices()
                    if (
                        self.max_fragments_per_pattern is not None
                        and len(idxs) > self.max_fragments_per_pattern
                    ):
                        idxs = idxs[: self.max_fragments_per_pattern]
                        truncated = True
                    for b_idx in idxs:
                        action_b = fb.get_action_at_idx(b_idx)
                        s2 = env.apply_forward_actions([s], [action_b])[0]
                        stack.append((s2, frags + [action_b.fragment.smiles]))
                elif isinstance(s, ReactionStateC):
                    fc = env.get_forward_action_spaces([s])[0]
                    if isinstance(fc, ReactionActionSpaceEarlyTerminate):
                        continue
                    for c_idx in fc.get_possible_actions_indices():
                        action_c = fc.get_action_at_idx(c_idx)
                        sa_next = env.apply_forward_actions([s], [action_c])[0]
                        mol = sa_next.molecule
                        if not mol.valid or mol.smiles in products:
                            continue
                        step = {
                            "step": len(hub_steps) + 1,
                            "reaction_idx": reaction.idx,
                            "reaction_smarts": reaction.reaction,
                            "reactant": state.molecule.smiles,
                            "fragments": frags,
                            "product": mol.smiles,
                        }
                        products[mol.smiles] = (mol, step)
                        if self.max_children is not None and len(products) >= self.max_children:
                            truncated = True
                            stack = []
                            break
            if self.max_children is not None and len(products) >= self.max_children:
                break

        # Score all enumerated products with the proxy (in batches).
        smiles_order = list(products.keys())
        terminals = [
            ReactionStateTerminal(molecule=products[smi][0], num_reactions=depth + 1)
            for smi in smiles_order
        ]
        scores: List[Optional[float]] = []
        for i in range(0, len(terminals), self.score_batch_size):
            scores.extend(gfn.score_states(terminals[i : i + self.score_batch_size]))

        children: List[Child] = []
        for smi, score in zip(smiles_order, scores):
            _, step = products[smi]
            route = {
                "product_smiles": smi,
                "num_reactions": depth + 1,
                "building_block": bb,
                "steps": list(hub_steps) + [step],
            }
            children.append(
                Child(
                    smiles=smi,
                    score=score,
                    num_reactions=depth + 1,
                    route=route,
                    source="enumerated",
                )
            )

        if truncated and self.verbose:
            print(
                f"[expander] hub {hub.smiles[:32]}… (depth {depth}): enumeration capped "
                f"(max_fragments_per_pattern={self.max_fragments_per_pattern}, "
                f"max_children={self.max_children}); kept {len(products)} products.",
                flush=True,
            )
        return children
