"""``OracleRewardProxy`` — turn any :class:`~glue.oracles.base.GlueOracle` into an
in-loop RGFN reward generator.

The missing symmetric piece of ``glue/proxies``. We already have
``LearnedGlueProxy`` (a *learned* surrogate ``M`` refit on oracle labels, for the
active-learning loop) and ``ExampleGlueProxy`` (the QED template). This adapter is
the third case the **fixed-reward** pipeline needs: expose an oracle *directly* as
the reward generator, called every training step (``[koziarski2024rgfn]`` §4.1's
"docking directly in the training loop"). It is deliberately generic — ANY oracle
(the 6TD3 differential today, an MD-stability oracle later) becomes a fixed reward
by wrapping it here, with no new proxy code.

Sign / normalization — matched to upstream ``DockingMoleculeProxy``:
    An oracle maps SMILES → a raw score whose orientation is declared by
    ``oracle.higher_is_better`` (e.g. the 6TD3 ``dvina`` differential is
    *lower-is-better*). A GFlowNet reward must be **non-negative** and
    **higher-is-better**. We apply exactly upstream's docking convention
    (``docking_proxy.py`` line 188: ``clip(-x / norm, 0, inf)``), generalized to the
    oracle's declared orientation::

        reward = max(sign · raw / norm, 0),   sign = +1 if higher_is_better else −1

    so a lower-is-better oracle is flipped to higher-is-better, negatives are clipped
    to 0 (as upstream does), and ``nan`` (an oracle failure per the ``GlueOracle``
    contract) maps to ``failed_score`` (default 0.0, matching the docking proxy).

Provenance: the raw oracle score is preserved as a ``raw_score`` component, so the
fixed-reward pipeline records both the reward the GFN trained on (``value``) and the
underlying scientific quantity (e.g. ``dvina``) as a candidate column.
"""

from typing import Dict, List

import gin

from glue.oracles.base import GlueOracle
from rgfn.api.type_variables import TState
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase


@gin.configurable()
class OracleRewardProxy(CachedProxyBase[ReactionState]):
    """Wrap a :class:`GlueOracle` as a non-negative, higher-is-better GFN reward.

    Args:
        oracle: the scorer to call every step (a ``glue.oracles.GlueOracle``). Its
            ``higher_is_better`` sets the sign flip; its ``score`` is called on the
            batch's SMILES.
        norm: divisor applied to the (sign-corrected) raw score before clipping,
            matching ``DockingMoleculeProxy.norm``. Scales the reward magnitude that
            feeds ``r(x) = reward(x)^beta``.
        failed_score: reward assigned to molecules the oracle fails to score
            (``nan``/``None``) and to early-terminal (invalid) states. Default 0.0,
            as upstream docking.
        free_gpu_cache: when True (default), call ``torch.cuda.empty_cache()`` before
            each oracle call. Essential when the oracle docks on the GPU (QuickVina2-
            GPU) *while* the GFN trains: torch's caching allocator otherwise holds the
            device (40 GB → ~1 GB free) and the docking subprocess gets 0 poses
            (Logs/014). A no-op / harmless for a pure-CPU oracle.
    """

    def __init__(
        self,
        oracle: GlueOracle,
        norm: float = 1.0,
        failed_score: float = 0.0,
        free_gpu_cache: bool = True,
    ):
        super().__init__()
        self.oracle = oracle
        self.norm = float(norm)
        self.failed_score = float(failed_score)
        self.free_gpu_cache = free_gpu_cache
        self._sign = 1.0 if oracle.higher_is_better else -1.0
        # Early-terminal (invalid) states get the worst reward. Stored as a dict so
        # every cache entry has the same shape (CachedProxyBase decides float-vs-dict
        # from the first entry; keeping all entries dicts keeps the components path).
        self.cache = {
            ReactionStateEarlyTerminal(None): {
                "value": self.failed_score,
                "raw_score": float("nan"),
            }
        }

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def _compute_proxy_output(self, states: List[TState]) -> List[Dict[str, float]]:
        smiles = [state.molecule.smiles for state in states]
        if self.free_gpu_cache:
            self._free_gpu_cache()
        raw_scores = self.oracle.score(smiles)
        out: List[Dict[str, float]] = []
        for raw in raw_scores:
            if raw is None or raw != raw:  # None or nan -> oracle failure
                out.append({"value": self.failed_score, "raw_score": float("nan")})
            else:
                reward = self._sign * float(raw) / self.norm
                out.append({"value": max(reward, 0.0), "raw_score": float(raw)})
        return out

    @staticmethod
    def _free_gpu_cache() -> None:
        """Return torch's reserved-but-unused GPU memory so a GPU docking oracle's
        subprocess can allocate (Logs/014). Best-effort; never fatal, no-op off-CUDA."""
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - freeing cache must never break the reward
            pass
