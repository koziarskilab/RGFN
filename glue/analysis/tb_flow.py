"""Balance-based flow estimate — read ``F(h)`` off the Trajectory Balance condition.

The GFN is trained with **Trajectory Balance** (``[malkin2022trajectorybalance]``), which
learns ``P_F``, ``P_B`` and a single global ``Z`` but *no* per-state flow head (``hub_graph``
therefore estimates flow by counting visits). But the TB constraint still lets us
**recover** the flow through a hub.

For a complete trajectory ``s0→…→h→…→x`` the paper's Eq. (13) is
``Z·∏ P_F = R(x)·∏ P_B`` with ``F(x)=R(x)`` (Eq. 8). Splitting the products at ``h`` and
using sub-trajectory balance on the prefix (the ``s0→h`` part cancels) leaves the
detailed-/sub-trajectory-balance relation (Eq. 7 generalized):

    F(h) = R(x) · P_B(h|x) / P_F(x|h)

where ``P_F(x|h)`` / ``P_B(h|x)`` are the **path-products** of the per-step forward /
backward probabilities along the ``h→x`` suffix (``h`` and ``x`` are several reaction
steps apart in RGFN). By Proposition 1 this is *exact* when the model perfectly satisfies
TB, and gives the same value for any terminal ``x`` / any path; off-optimum it is an
estimator, so we **average over all sampled trajectories through ``h``**. That arithmetic
mean is unbiased for ``F(h)`` under forward sampling (the ``R(x)·P_B`` mass telescopes to
``F(h)``).

Two views, one loss: visit-counting estimates the ``Z·∏P_F`` (forward/sampling) side of
the TB loss; this estimates the ``R(x)·∏P_B`` (reward/backward) side. TB training drives
them together, so their per-hub disagreement (:func:`flow_agreement`) is a direct
training-quality diagnostic.

Everything is computed in log-space from the per-step ``log P_F`` / ``log P_B`` that
``ObjectiveBase.assign_log_probs`` already produces, plus the shaped ``log R(x)`` the GFN
balanced against (``reward_outputs.log_reward`` — the training reward, *not* the raw proxy).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

from glue.analysis.hub_graph import HubGraph
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionStateA,
    ReactionStateTerminal,
)


@torch.no_grad()
def annotate_tb_flow(graph: HubGraph, trajectories, gfn) -> HubGraph:
    """Annotate each hub in ``graph`` with a Trajectory-Balance flow estimate ``F(h)``.

    Runs ``objective.assign_log_probs`` on ``trajectories`` (one forward + one backward
    policy pass over the batch), then for every ``ReactionStateA`` position on every
    *terminated* trajectory computes ``log F = log R(x) + Σ logP_B(suffix) − Σ logP_F(suffix)``
    and stores, per hub, the (log of the) arithmetic mean of ``F`` over its trajectories.

    Mutates and returns ``graph`` (sets ``hub.tb_flow`` / ``tb_flow_log`` / ``tb_flow_n``).
    No-op with a warning if the trajectories carry no rewards.
    """
    objective = getattr(gfn, "objective", None)
    if objective is None:
        print("[tb_flow] no objective on the loaded model — skipping", flush=True)
        return graph
    reward_outputs = getattr(trajectories, "_reward_outputs", None)
    if reward_outputs is None or reward_outputs.log_reward is None:
        print("[tb_flow] trajectories carry no rewards — skipping", flush=True)
        return graph

    # Both policies must have their device set (forward is set at load; backward is not
    # touched by inference, so set it here) or index tensors land on the wrong device.
    for pol in (objective.forward_policy, objective.backward_policy):
        if hasattr(pol, "set_device"):
            pol.set_device(gfn.device)

    objective.assign_log_probs(trajectories)  # sets _forward/_backward_log_probs_flat
    fwd = trajectories.get_forward_log_probs_flat().detach().cpu()
    bwd = trajectories.get_backward_log_probs_flat().detach().cpu()
    log_reward = reward_outputs.log_reward.detach().cpu().reshape(-1)

    states_list = trajectories._states_list
    action_counts = [len(a) for a in trajectories._actions_list]

    log_estimates: Dict[Any, List[float]] = defaultdict(list)
    offset = 0
    for i, (states, m) in enumerate(zip(states_list, action_counts)):
        fwd_i, bwd_i = fwd[offset : offset + m], bwd[offset : offset + m]
        offset += m
        if m == 0 or not isinstance(states[-1], ReactionStateTerminal):
            continue
        lr = float(log_reward[i]) if i < len(log_reward) else float("nan")
        if not math.isfinite(lr):
            continue
        # suffix sums: suf[p] = Σ_{j>=p} logP(action_j) = log path-prob of states[p]→terminal.
        suf_fwd = torch.flip(torch.cumsum(torch.flip(fwd_i, [0]), 0), [0])
        suf_bwd = torch.flip(torch.cumsum(torch.flip(bwd_i, [0]), 0), [0])
        for p, s in enumerate(states):
            if p >= m or not isinstance(s, ReactionStateA):
                continue  # p==m is the terminal; only intermediates (p<m) are hubs
            log_f = lr + float(suf_bwd[p]) - float(suf_fwd[p])
            log_estimates[(s.molecule.smiles, s.num_reactions)].append(log_f)

    for key, logs in log_estimates.items():
        hub = graph.hubs.get(key)
        if hub is None:
            continue
        t = torch.tensor(logs, dtype=torch.float64)
        # log of the arithmetic mean of F = logsumexp(logF) - log(n)  (the unbiased mean).
        hub.tb_flow_log = float(torch.logsumexp(t, 0) - math.log(len(logs)))
        hub.tb_flow = math.exp(hub.tb_flow_log) if hub.tb_flow_log < 700 else float("inf")
        hub.tb_flow_n = len(logs)

    n_annot = sum(1 for h in graph.all_hubs() if h.tb_flow_log is not None)
    print(f"[tb_flow] annotated {n_annot}/{len(graph.hubs)} hubs with TB flow", flush=True)
    return graph


def log_Z(gfn) -> Optional[float]:
    """The model's global ``log Z`` (sum of the ``logZ`` parameter), or None if absent.

    Visit-count estimates ``F(h)/Z``; TB flow estimates absolute ``F(h)``. Subtracting
    ``log Z`` puts TB flow on the same ``F/Z`` scale for a like-for-like comparison."""
    obj = getattr(gfn, "objective", None)
    logz = getattr(obj, "logZ", None)
    if logz is None:
        return None
    try:
        return float(logz.detach().sum())
    except Exception:  # noqa: BLE001
        return None


def flow_agreement(graph: HubGraph, gfn=None, min_visits: int = 2) -> Dict[str, Any]:
    """Compare the two flow estimates across hubs — a training-quality diagnostic.

    For hubs that have both a TB flow and at least ``min_visits`` visits, correlate the
    (log) sampling flow ``visit_count/N`` against the (log) TB flow (shifted by ``log Z``
    to the same ``F/Z`` scale if available). Well-trained ⇒ the two agree (high
    correlation, points near ``y=x``). Returns correlation stats plus the point arrays for
    :func:`glue.analysis.plot.plot_flow_agreement`.
    """
    logz = log_Z(gfn) if gfn is not None else None
    xs: List[float] = []  # log sampling flow  (log F/Z)
    ys: List[float] = []  # log TB flow        (log F, minus logZ if available)
    labels: List[str] = []
    for h in graph.all_hubs():
        if h.tb_flow_log is None or h.visit_count < min_visits:
            continue
        vf = graph.normalized_flow(h)
        if vf <= 0:
            continue
        xs.append(math.log(vf))
        ys.append(h.tb_flow_log - (logz if logz is not None else 0.0))
        labels.append(h.smiles)

    return {
        "n_hubs": len(xs),
        "pearson_log": _pearson(xs, ys),
        "spearman": _spearman(xs, ys),
        "log_Z": logz,
        "x_log_visit_flow": xs,
        "y_log_tb_flow": ys,
        "labels": labels,
        "aligned_to_fz": logz is not None,
    }


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    import numpy as np

    a, b = np.asarray(x), np.asarray(y)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    import numpy as np

    def ranks(v):
        order = np.argsort(np.asarray(v), kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        return r

    return _pearson(list(ranks(x)), list(ranks(y)))
