"""Persist / restore SCENT's backward-policy **guidance-model** weights.

Why this exists
---------------
SCENT's cost-guided backward policy shapes ``P_B`` with two *learned* MLPs — the
Synthesis-Cost model ``ĉ_B^S`` (``SimpleCostModel.mlp_c``) and the Decomposability
model ``ĉ_B^D`` (``BinaryDecomposableModel.mlp_c``), ~263k params each. They are
trained online (each has its own Adam, stepped in ``on_start_computing_objective``).

But ``JointlyGuidedBackwardPolicy`` stores its two sub-policies in a **plain Python
list** (``self.policies = [...]``), not an ``nn.ModuleList``. ``nn.Module`` only
registers ``nn.Module``/``Parameter`` *attributes* in ``_modules``; a list of modules
is invisible to it. So ``objective.state_dict()`` never recurses into the guidance
models, and the trained ``P_B`` weights are silently dropped from ``last_gfn.pt``
(verified: Logs entry on the P_B acid test — 0 of 131 checkpoint keys are guidance).

Reloading a checkpoint therefore gives **freshly-initialised random** guidance MLPs,
so the recovered ``P_B`` is not the trained distribution — which breaks any post-hoc
Trajectory-Balance flow analysis that relies on ``P_B`` (``F(h) = R·∏P_B / ∏P_F``).

The clone (``external/scent``) is a pinned, git-ignored dependency, so rather than
patch it we capture the missing weights from our own adapter: after training, write a
``guidance_models.pt`` sidecar next to ``last_gfn.pt``; on reload, load it back into
the freshly-built guidance models. ``P_B`` is then exactly the trained distribution.

Sidecar format::

    {"guidance": {"policies.<i>.<attr>": <module state_dict>, ...}}

keyed by the sub-policy's position in ``JointlyGuidedBackwardPolicy.policies`` and the
guidance-model attribute name, so it round-trips regardless of policy ordering.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import torch

# Attribute names under which a guided backward policy holds its learned model.
_GUIDANCE_ATTRS = ("cost_prediction_model", "decomposable_prediction_model")


def _iter_guidance(backward_policy) -> Iterator[Tuple[str, torch.nn.Module]]:
    """Yield ``(key, module)`` for every learned guidance model reachable from a
    backward policy — whatever its shape (a ``JointlyGuidedBackwardPolicy`` with a
    ``.policies`` list, a single guided policy, or none). ``key`` is stable across
    save/load (``policies.<i>.<attr>``)."""
    policies = getattr(backward_policy, "policies", None)
    if policies is None:  # a single guided policy, not the jointly-guided wrapper
        policies = [backward_policy]
    for i, pol in enumerate(policies):
        for attr in _GUIDANCE_ATTRS:
            module = getattr(pol, attr, None)
            if module is not None and hasattr(module, "state_dict"):
                yield f"policies.{i}.{attr}", module


def save_guidance_models(objective, path) -> List[str]:
    """Save every guidance model on ``objective.backward_policy`` to ``path``.

    Returns the list of keys written (empty if the backward policy has no learned
    guidance models, e.g. cost guidance disabled — in which case ``P_B`` is fully
    determined by static data and needs no sidecar)."""
    blob = {key: module.state_dict() for key, module in _iter_guidance(objective.backward_policy)}
    torch.save({"guidance": blob}, str(path))
    return list(blob.keys())


def load_guidance_models(
    objective, path, map_location="cpu", strict: bool = True
) -> Tuple[List[str], List[str]]:
    """Load a ``guidance_models.pt`` sidecar into ``objective``'s guidance models.

    Returns ``(loaded_keys, unmatched_keys)`` — ``unmatched_keys`` are entries in the
    sidecar with no counterpart on the live backward policy (should be empty for a
    config that matches the one the sidecar was saved from)."""
    blob = torch.load(str(path), map_location=map_location)["guidance"]
    live = dict(_iter_guidance(objective.backward_policy))
    loaded: List[str] = []
    for key, state_dict in blob.items():
        module = live.get(key)
        if module is not None:
            module.load_state_dict(state_dict, strict=strict)
            loaded.append(key)
    unmatched = [key for key in blob if key not in live]
    return loaded, unmatched
