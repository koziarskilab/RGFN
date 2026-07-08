"""FragGFN generator — Recursion's fragment-based GFlowNet wired to our learned
proxy ``M`` as its reward (the non-synthesizable baseline from the RGFN paper).

RGFN's pitch is *synthesizability by construction* (a reaction DAG over building
blocks). FGFN is the foil: ``FragMolBuildingEnvContext`` builds a molecule by
joining fragments at arbitrary attachment points — valid graphs, but **no
synthesis route** (``has_route=0``). This module is a thin adapter over Recursion's
``gflownet`` that swaps the built-in pretrained sEH proxy for **our** in-loop
reward, so FGFN is trained against the *same* proxy ``M`` (refit on the *same*
docking oracle) as the RGFN entrant — only the generator's action space differs.

Reward / β faithfulness
-----------------------
``FragGFNTask.compute_obj_properties`` returns ``M``'s positive reward
``exp(signed_value)`` (see :meth:`AtomMPNNProxy.reward`). ``cond_info_to_logreward``
applies a **constant** temperature β via gflownet's ``TemperatureConditional``, so
the TB target becomes ``reward^β = exp(signed_value·β)`` — bit-for-bit the same
target RGFN trains against (``LearnedGlueProxy`` + exponential-boosted reward,
β=8). A constant (not sampled) temperature matches RGFN's single fixed β.

The proxy ``M`` is held by the task; the active-learning loop refits it in place
each round (``al_loop.py``). We run with ``num_workers=0`` so the refit is visible
to sampling immediately (no multiprocessing model copies to resync).
"""

import socket
from typing import Dict, List, Tuple

import torch
from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.transforms import to_logreward
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor

from validation.generators.fraggfn.proxy import AtomMPNNProxy


class FragGFNTask(GFNTask):
    """A ``GFNTask`` whose reward is our learned proxy ``M`` (atom-graph MPNN)."""

    def __init__(self, cfg: Config, proxy: AtomMPNNProxy):
        self.cfg = cfg
        self.proxy = proxy
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(
        self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties
    ) -> LogScalar:
        # transform multiplies log-reward by the (constant) temperature β →
        # logreward = β·log(R) = log(R^β) = signed_value·β. Matches RGFN exactly.
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(obj_props)))

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


class FragGFNTrainer(StandardOnlineTrainer):
    """Fragment-GFlowNet trainer (TB) with our proxy as the task reward.

    Defaults mirror ``SEHFragTrainer`` (the canonical fragment task) except:
      * the reward is our refit-able ``AtomMPNNProxy`` (set via ``attach_proxy``
        before ``setup`` runs), not the pretrained sEH proxy;
      * ``num_workers=0`` (single-process) so per-round proxy refits take effect;
      * a **constant** temperature β (the active-learning loop sets it from the
        run config) instead of sampled tempering.
    """

    task: FragGFNTask

    def __init__(self, config: Config, proxy: AtomMPNNProxy, print_config: bool = False):
        self._proxy = proxy
        super().__init__(config, print_config=print_config)

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 0  # single-process: proxy refits are visible to sampling
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.num_from_policy = 64
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False
        cfg.algo.tb.do_sample_p_b = True

        cfg.replay.use = False

    def setup_task(self):
        self.task = FragGFNTask(cfg=self.cfg, proxy=self._proxy)

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes,
            num_cond_dim=self.task.num_cond_dim,
            fragments=bengio2021flow.FRAGMENTS,
        )


def build_constant_temperature(cfg: Config, beta: float) -> None:
    """Configure a CONSTANT temperature β on a gflownet Config (matches RGFN's
    single fixed β). Sets the conditional's sampling distribution to a constant so
    every trajectory is tempered identically and the TB target is ``R^β``."""
    cfg.cond.temperature.sample_dist = "constant"
    cfg.cond.temperature.dist_params = [float(beta)]
