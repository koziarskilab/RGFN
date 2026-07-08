"""RxnFlow generator — RxnFlow's synthesis GFlowNet wired to our learned proxy ``M``
as its reward (the *synthesizable* baseline; the peer to RGFN, opposite FragGFN).

RxnFlow (``[seo2024rxnflow]``) assembles molecules along a synthetic pathway — pick a
building block, then apply a reaction template to a chosen reactant — over a large
action space (~1.2M blocks × 71 templates) tamed by action-space subsampling. It is
built on a bundled Recursion ``gflownet``, so the task/trainer wiring mirrors our
FragGFN adapter: swap the built-in reward for **our** in-loop proxy ``M`` so RxnFlow
trains against the *same* proxy (refit on the *same* docking oracle, *same* seed and
budget) as the RGFN entrant — only the generator's action space differs.

Reward / β faithfulness
-----------------------
:meth:`RxnFlowTask.compute_obj_properties` returns ``M``'s positive reward
``exp(signed_value)`` (see :meth:`AtomMPNNProxy.reward`). A **constant** temperature β
(set on the config by :func:`build_constant_temperature`) makes the gflownet TB target
``reward^β = exp(signed_value·β)`` — the same target RGFN and FragGFN train against
(β=8). A constant (not sampled) temperature matches RGFN's single fixed β.

The proxy ``M`` is held by the task; the active-learning loop refits it in place each
round (``al_loop.py``). We run with ``num_workers=0`` so a refit is visible to sampling
immediately (no multiprocessing model copies to resync).

API note (flagged for Balam validation, ``docs/REFACTOR_LOG.md``): the exact RxnFlow
base-class names/signatures (``RxnFlowTrainer``, ``BaseTask``, ``setup_task``,
``set_default_hps``) and the env-context wiring (``--env_dir`` building blocks +
templates) are taken from the RxnFlow docs/examples; confirm against the cloned repo in
the ``rxnflow`` env. This module is written to mirror the FragGFN adapter so the diff
is small and localized.
"""

from typing import List, Tuple

import torch
from gflownet import ObjectProperties
from rdkit.Chem.rdchem import Mol as RDMol

# RxnFlow's online-training base + task base (bundled gflownet underneath). Aliased so
# our subclass name doesn't collide with the upstream ``RxnFlowTrainer``.
from rxnflow.base import BaseTask
from rxnflow.base import RxnFlowTrainer as RxnFlowTrainerBase
from torch import Tensor

from validation.generators.rxnflow.proxy import AtomMPNNProxy


class RxnFlowTask(BaseTask):
    """A RxnFlow ``BaseTask`` whose reward is our learned proxy ``M`` (atom-graph MPNN).

    Follows RxnFlow's documented single-objective task contract: implement
    ``compute_obj_properties(mols) -> (ObjectProperties, is_valid)`` returning a
    non-negative reward per valid molecule. The β-tempering is handled by the base
    task's temperature conditional, configured constant via
    :func:`build_constant_temperature`.
    """

    is_moo = False

    def __init__(self, cfg, proxy: AtomMPNNProxy):
        # Store the proxy before base init (base setup must never need it, but be safe).
        self.proxy = proxy
        super().__init__(cfg)

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        from rdkit import Chem

        smis = [Chem.MolToSmiles(m) if m is not None else None for m in mols]
        is_valid_list = [s is not None for s in smis]
        is_valid = torch.tensor(is_valid_list).bool()
        valid_smis = [s for s in smis if s is not None]
        if not valid_smis:
            return ObjectProperties(torch.zeros((0, 1))), is_valid
        rewards = self.proxy.reward(valid_smis)  # positive scalar per valid mol
        preds = torch.tensor(rewards, dtype=torch.float).reshape((-1, 1))
        return ObjectProperties(preds), is_valid


class RxnFlowGlueTrainer(RxnFlowTrainerBase):
    """RxnFlow synthesis-GFlowNet trainer with our proxy as the task reward.

    Mirrors :class:`validation.generators.fraggfn.task.FragGFNTrainer`: the reward is
    our refit-able :class:`AtomMPNNProxy` (passed in, held by the task), and we force
    ``num_workers=0`` so per-round proxy refits take effect immediately. All
    synthesis-specific defaults (env dir, action subsampling, TB) come from RxnFlow's
    own ``set_default_hps`` (called via ``super()``); we override only what the
    active-learning loop requires.
    """

    task: RxnFlowTask

    def __init__(self, config, proxy: AtomMPNNProxy, print_config: bool = False):
        self._proxy = proxy
        super().__init__(config, print_config=print_config)

    # NB: we deliberately do NOT override ``set_default_hps`` — RxnFlow's own defaults
    # (``num_workers=0``, ``method="TB"``, ``train_random_action_prob=0.1``) are the
    # validated-stable regime. Forcing ``train_random_action_prob=0.0`` (as the
    # RGFN/FragGFN entrants do) makes RxnFlow's TB loss go non-finite on this synthesis
    # action space — RxnFlow explicitly recommends a positive exploration rate. So we
    # keep RxnFlow's exploration knob at its native value; the only shared-fairness
    # levers (proxy ``M``, oracle, seed, budget, constant β) are wired elsewhere.

    def setup_task(self) -> None:
        self.task = RxnFlowTask(self.cfg, proxy=self._proxy)


def build_constant_temperature(cfg, beta: float) -> None:
    """Configure a CONSTANT temperature β on a gflownet/RxnFlow Config (matches RGFN's
    single fixed β). Sets the conditional's sampling distribution to a constant so every
    trajectory is tempered identically and the TB target is ``R^β``. Same mapping as
    ``validation.generators.fraggfn.task.build_constant_temperature``."""
    cfg.cond.temperature.sample_dist = "constant"
    cfg.cond.temperature.dist_params = [float(beta)]
