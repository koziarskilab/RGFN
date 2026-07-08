"""The hub graph — the central data structure this whole analysis is built on.

Background (see ``rgfn/gfns/reaction_gfn/api/reaction_api.py`` +
``reaction_env.py``). A reaction-GFN builds a molecule as a chain of
``ReactionStateA`` intermediates::

    S0 -> SA(bb,0) -> [SB..SC] -> SA(m1,1) -> ... -> SA(m_{k-1},k-1)
                                                  -> [SB..SC] -> SA(x,k) -> Terminal(x)

At every ``ReactionStateA`` the forward policy either **STOPs** (emits the current
molecule as terminal) or applies **one more reaction**. That makes each ``SA``
intermediate a natural **hub**: a shared scaffold from which a single *diverging final
reaction* yields many terminal analogs — exactly late-stage diversification, and the
thing a chemist would batch (make the intermediate once, split into parallel final
steps).

This module reduces a batch of sampled ``Trajectories`` to a :class:`HubGraph`:

* one :class:`Hub` per distinct intermediate ``(smiles, num_reactions)``,
* its **flow** = how many sampled trajectories passed through it (an unbiased Monte-
  Carlo estimate of the marginal state flow ``F(hub)/Z``; the configs train with
  **Trajectory Balance** (``[malkin2022trajectorybalance]``), which does *not*
  parameterize per-state flows, so visit frequency is the principled, model-agnostic
  flow signal — ``glue.analysis.tb_flow`` adds the complementary balance-based estimate),
* its **observed children** = the terminal molecules seen one reaction downstream
  (the flow-realized diversification set), each with the proxy score the sampler
  attached and the full synthesis route.

Everything here is a pure reduction over already-sampled trajectories: **no model
calls, no proxy calls**. Exhaustively enumerating a hub's children (beyond what the
policy happened to sample) is the separate, opt-in job of
``glue.analysis.expanders.EnumerativeExpander``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from glue.active_learning.route import extract_route
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionStateA,
    ReactionStateTerminal,
)

HubKey = Tuple[str, int]  # (intermediate SMILES, num_reactions)


@dataclass
class Child:
    """A terminal molecule reachable one reaction downstream of a hub."""

    smiles: str
    score: Optional[float]  # proxy value (raw; orient with HubGraph.higher_is_better)
    num_reactions: int  # depth of this terminal (== hub depth + 1 for a 1-reaction child)
    route: Optional[Dict[str, Any]] = None  # full synthesis route to this molecule
    source: str = "observed"  # "observed" (sampled) | "enumerated" (env-expanded)


@dataclass
class Hub:
    """One reaction intermediate (a ``ReactionStateA``) and its diversification set."""

    smiles: str
    num_reactions: int  # depth of the intermediate (>=1 hubs are terminatable molecules)
    visit_count: int = 0  # number of sampled trajectories passing through this hub == flow
    route_to_hub: Optional[Dict[str, Any]] = None  # shared route to build the intermediate
    state: Optional[ReactionStateA] = None  # the real state object (used by EnumerativeExpander)
    # observed 1-reaction-away terminal children, de-duplicated by SMILES (best score kept).
    observed_children: Dict[str, Child] = field(default_factory=dict)
    # Trajectory-balance flow estimate F(h) = mean over trajectories through h of
    # R(x)*P_B(h|x)/P_F(x|h) (see glue.analysis.tb_flow). Populated only if annotated;
    # ``tb_flow_log`` is the log (the safe field to rank on — F(h) can overflow float).
    tb_flow: Optional[float] = None
    tb_flow_log: Optional[float] = None
    tb_flow_n: int = 0  # number of trajectory estimates averaged

    @property
    def key(self) -> HubKey:
        return (self.smiles, self.num_reactions)

    @property
    def flow(self) -> int:
        """Empirical (unnormalized) flow through this hub == trajectory visit count."""
        return self.visit_count

    @property
    def n_observed_children(self) -> int:
        return len(self.observed_children)

    def children_list(self) -> List[Child]:
        return list(self.observed_children.values())


class HubGraph:
    """All hubs found in one batch of sampled trajectories, plus book-keeping."""

    def __init__(self, higher_is_better: bool, n_trajectories: int):
        self.hubs: Dict[HubKey, Hub] = {}
        self.higher_is_better = higher_is_better
        self.n_trajectories = n_trajectories
        self.n_terminal = 0  # trajectories that reached a real terminal molecule

    # -------------------------------------------------------------- construction
    def _get_or_add(self, state: ReactionStateA) -> Hub:
        key: HubKey = (state.molecule.smiles, state.num_reactions)
        hub = self.hubs.get(key)
        if hub is None:
            hub = Hub(
                smiles=state.molecule.smiles,
                num_reactions=state.num_reactions,
                state=state,
            )
            self.hubs[key] = hub
        return hub

    # -------------------------------------------------------------------- access
    def all_hubs(self) -> List[Hub]:
        return list(self.hubs.values())

    def candidate_hubs(
        self, min_depth: int = 1, max_depth: Optional[int] = None, min_children: int = 1
    ) -> List[Hub]:
        """Hubs eligible for selection: within a depth band and with enough observed
        children to actually diversify. ``min_depth=1`` excludes bare building blocks
        by default (depth-0 intermediates cannot be terminated on their own)."""
        out = []
        for hub in self.hubs.values():
            if hub.num_reactions < min_depth:
                continue
            if max_depth is not None and hub.num_reactions > max_depth:
                continue
            if hub.n_observed_children < min_children:
                continue
            out.append(hub)
        return out

    def normalized_flow(self, hub: Hub) -> float:
        """``F(hub)/Z`` estimate: fraction of sampled trajectories through the hub."""
        return hub.visit_count / self.n_trajectories if self.n_trajectories else 0.0

    def summary(self) -> Dict[str, Any]:
        hubs = self.all_hubs()
        n_children = [h.n_observed_children for h in hubs]
        return {
            "n_trajectories": self.n_trajectories,
            "n_terminal": self.n_terminal,
            "n_hubs": len(hubs),
            "n_hubs_with_children": sum(1 for c in n_children if c > 0),
            "max_children_one_hub": max(n_children) if n_children else 0,
            "total_observed_children": sum(n_children),
        }


def build_hub_graph(
    trajectories,
    higher_is_better: bool,
    count_early_terminal_visits: bool = True,
) -> HubGraph:
    """Reduce a batch of sampled ``Trajectories`` to a :class:`HubGraph`.

    For each trajectory we:
      1. increment the visit count of every ``ReactionStateA`` it passes through
         (this is the flow signal), recording the first-seen route to each hub;
      2. if it reached a real terminal molecule ``x`` at depth ``k``, register ``x`` as
         a one-reaction-away child of the **penultimate** intermediate ``SA`` (depth
         ``k-1``) — i.e. the scaffold from which the final diverging reaction was taken.

    Args:
        trajectories: an ``rgfn.api.trajectories.Trajectories`` with reward outputs set
            (the valid/forward sampler attaches ``proxy`` scores at sampling time).
        higher_is_better: orientation of the proxy score (from the loaded model's proxy).
        count_early_terminal_visits: also count hub visits along trajectories that died
            in an early-terminal state (they still traversed those intermediates). Their
            dead terminal is never registered as a child either way.

    Returns:
        A populated :class:`HubGraph`.
    """
    graph = HubGraph(higher_is_better=higher_is_better, n_trajectories=len(trajectories))

    reward_outputs = getattr(trajectories, "_reward_outputs", None)
    scores: Optional[List[float]] = None
    if reward_outputs is not None and reward_outputs.proxy is not None:
        scores = reward_outputs.proxy.detach().cpu().reshape(-1).tolist()

    states_list = trajectories._states_list
    actions_list = trajectories._actions_list

    for i, (states, actions) in enumerate(zip(states_list, actions_list)):
        last = states[-1] if states else None
        is_terminal = isinstance(last, ReactionStateTerminal)
        if not is_terminal and not count_early_terminal_visits:
            continue

        # Positions of the ReactionStateA intermediates along the trajectory.
        sa_positions = [pos for pos, s in enumerate(states) if isinstance(s, ReactionStateA)]

        # 1. flow: count a visit for every intermediate; stash its route (first seen).
        for pos in sa_positions:
            hub = graph._get_or_add(states[pos])
            hub.visit_count += 1
            if hub.route_to_hub is None:
                hub.route_to_hub = extract_route(states[: pos + 1], actions[:pos])

        # 2. children: the terminal molecule is a 1-reaction child of the penultimate SA.
        if not is_terminal:
            continue
        graph.n_terminal += 1
        if len(sa_positions) < 2:
            # k == 0 (unreachable: a terminal needs >=1 reaction) — nothing to hang it on.
            continue
        hub_state = states[sa_positions[-2]]  # depth k-1 intermediate
        hub = graph._get_or_add(hub_state)
        score = scores[i] if scores is not None and i < len(scores) else None
        child = Child(
            smiles=last.molecule.smiles,
            score=score,
            num_reactions=last.num_reactions,
            route=extract_route(states, actions),
            source="observed",
        )
        _merge_child(hub, child, higher_is_better)

    return graph


def _merge_child(hub: Hub, child: Child, higher_is_better: bool) -> None:
    """Add a child to a hub, keeping the better-scored copy on a SMILES collision."""
    existing = hub.observed_children.get(child.smiles)
    if existing is None:
        hub.observed_children[child.smiles] = child
        return
    if child.score is None:
        return
    if existing.score is None or (
        child.score > existing.score if higher_is_better else child.score < existing.score
    ):
        hub.observed_children[child.smiles] = child
