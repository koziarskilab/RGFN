"""``TrainedGFN`` — load a trained reaction-GFN checkpoint and expose the few handles
the analysis needs: sample trajectories, score states, reach the env + policy.

This is the *only* place the analysis package touches gin / checkpoints. It reproduces
exactly the load sequence in the repo's ``infer.py`` (parse the same gin config the run
used, build the ``objective`` singleton, load the ``"model"`` state dict, build the
``valid_sampler`` whose ``.policy`` is the trained forward policy), then surfaces:

    .env             the ReactionEnv (transitions, forward action spaces) — for enumeration
    .forward_policy  the trained forward policy (action log-probs, optional state flows)
    .proxy           the reward's proxy (scores arbitrary terminal states)
    .sampler         the valid RandomSampler (forward sampling from S0, rewards attached)
    .higher_is_better / .is_non_negative   the proxy's score orientation

Gin uses process-wide singletons, so **one process loads one model**. Loading a second
different config in the same process calls ``gin.clear_config()`` first; re-parsing is
cheap but rebuilding the env (reads the building-block library) is not, so a sweep loads
the model once and reuses this handle across all strategies/trials.
"""

from __future__ import annotations

from typing import List, Optional

import gin
import torch

import glue  # noqa: F401  (side effect: registers our gin configurables before config parse)
from rgfn.api.trajectories import Trajectories

# The training configs bind ``Trainer.*`` (rgfn_base.gin); gin must know ``Trainer`` to
# parse those bindings even though we never instantiate it. ``rgfn/__init__`` does not
# register it (root ``train.py`` imports it explicitly), so we mirror that here.
from rgfn.trainer.trainer import Trainer  # noqa: F401
from rgfn.utils.helpers import seed_everything


class TrainedGFN:
    def __init__(
        self,
        env,
        forward_policy,
        proxy,
        sampler,
        objective,
        device: str,
        config_path: str,
        checkpoint_path: str,
    ):
        self.env = env
        self.forward_policy = forward_policy
        self.proxy = proxy
        self.sampler = sampler
        self.objective = objective
        self.device = device
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

    # --------------------------------------------------------------------- load
    @classmethod
    def load(
        cls,
        config_path: str,
        checkpoint_path: str,
        *,
        checkpoint_key: str = "model",
        bindings: Optional[List[str]] = None,
        device: Optional[str] = None,
        run_name: str = "analysis",
    ) -> "TrainedGFN":
        """Build a ``TrainedGFN`` from a gin config + a checkpoint ``.pt``.

        Args:
            config_path: the gin config the run was trained with (e.g.
                ``configs/glue/fixed_reward_seh_proxy_stdlib.gin``).
            checkpoint_path: a ``train/checkpoints/{best,last}_gfn.pt`` file.
            checkpoint_key: key in the checkpoint dict holding the objective state dict.
            bindings: extra gin bindings to append after parsing the config.
            device: ``"cuda"``/``"cpu"``; defaults to cuda if available.
            run_name: bound to gin ``run_name`` (configs reference ``%run_name``).
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        gin.clear_config()
        all_bindings = [f'run_name="{run_name}"'] + list(bindings or [])
        gin.parse_config_files_and_bindings([config_path], bindings=all_bindings)

        # Objective holds the trained forward/backward policies (mirrors infer.py).
        objective = gin.get_configurable("objective/gin.singleton")()
        objective.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # strict=False: trained checkpoints carry recomputed-at-sampling cache buffers
        # (e.g. ``forward_policy.b_action_embedding_fn._cache``) absent from a fresh model.
        # Those are repopulated on the first sample; any *other* mismatch is surfaced.
        result = objective.load_state_dict(checkpoint[checkpoint_key], strict=False)
        real_missing = [k for k in result.missing_keys if "_cache" not in k]
        real_unexpected = [k for k in result.unexpected_keys if "_cache" not in k]
        if real_missing or real_unexpected:
            print(
                f"[TrainedGFN] state_dict mismatch beyond caches — "
                f"missing={real_missing} unexpected={real_unexpected}",
                flush=True,
            )
        objective.eval()

        # The valid sampler forward-samples from S0 with the learned policy + reward.
        sampler = gin.get_configurable("valid_sampler/gin.singleton")()
        sampler.policy.set_device(device)

        proxy = sampler.reward.proxy if sampler.reward is not None else None
        forward_policy = getattr(objective, "forward_policy", sampler.policy)

        return cls(
            env=sampler.env,
            forward_policy=forward_policy,
            proxy=proxy,
            sampler=sampler,
            objective=objective,
            device=device,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )

    # ------------------------------------------------------------------ proxy
    @property
    def higher_is_better(self) -> bool:
        return bool(self.proxy.higher_is_better) if self.proxy is not None else True

    @property
    def is_non_negative(self) -> bool:
        return bool(self.proxy.is_non_negative) if self.proxy is not None else True

    @torch.no_grad()
    def score_states(self, states: List) -> List[Optional[float]]:
        """Proxy value for each state (the raw score; orient with ``higher_is_better``).

        Used to score enumerated children the sampler never rewarded. Returns ``None``
        entries only if there is no proxy.
        """
        if not states:
            return []
        if self.proxy is None:
            return [None] * len(states)
        out = self.proxy.compute_proxy_output(states)
        return [float(v) for v in out.value.detach().cpu().reshape(-1).tolist()]

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample_trajectories(
        self, n_trajectories: int, batch_size: int = 200, seed: Optional[int] = None
    ) -> Trajectories:
        """Forward-sample ``n_trajectories`` and return them as one ``Trajectories``.

        Rewards (``proxy`` scores) are attached by the sampler. Iterates the sampler's
        batched iterator and concatenates, so large samples stay within memory.
        """
        if seed is not None:
            seed_everything(seed)
        batches = list(self.sampler.get_trajectories_iterator(n_trajectories, batch_size))
        return Trajectories.from_trajectories(batches) if len(batches) > 1 else batches[0]
