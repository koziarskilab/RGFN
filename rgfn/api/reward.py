import math
from typing import Generic, List, Literal

import gin
import torch

from rgfn.api.proxy_base import ProxyBase
from rgfn.api.reward_output import RewardOutput
from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.api.type_variables import TState


@gin.configurable()
class Reward(Generic[TState], TrainingHooksMixin):
    """
    A class representing the reward function. The reward function is a function that takes a batch of states and
        computes rewards that are used to train the policy.

    Type parameters:
        TState: The type of the states.

    Attributes:
        proxy: The proxy that is used to compute the rewards.
        reward_boosting: The type of reward boosting. It can be either "linear" or "exponential".
        min_reward: The minimum reward value. If the reward boosting is "linear", the rewards are clamped to be at least
            `min_reward`. If the reward boosting is "exponential", the log rewards are clamped to be at least
            `math.log(min_reward)`.
        beta: The coefficient that multiplies the proxy values to compute the rewards.
    """

    def __init__(
        self,
        proxy: ProxyBase[TState],
        reward_boosting: Literal["linear", "exponential"] = "linear",
        min_reward: float = 0.0,
        beta: float = 1.0,
    ):
        super().__init__()
        assert reward_boosting in ["linear", "exponential"]
        if reward_boosting == "linear" and not proxy.is_non_negative:
            raise ValueError("Reward boosting is linear but proxy is not non-negative!")
        self.proxy = proxy
        self.reward_boosting = reward_boosting
        self.min_reward = min_reward
        self.min_log_reward = math.log(min_reward) if min_reward > 0 else -float("inf")
        self.beta = beta

    @property
    def hook_objects(self) -> List["TrainingHooksMixin"]:
        return [self.proxy]

    @torch.no_grad()
    def compute_reward_output(self, states: List[TState]) -> RewardOutput:
        """
        Compute the reward output on a batch of states.

        Args:
            states: a list of states of length `n_states`.

        Returns:
            a `RewardOutput` object. The reward output contains the log rewards, rewards, proxy values,
            and possibly some components of the proxy values that may be used to compute metrics.
        """
        proxy_output = self.proxy.compute_proxy_output(states)
        value = proxy_output.value
        signed_value = value if self.proxy.higher_is_better else -value
        if self.reward_boosting == "linear":
            reward = signed_value * self.beta
            reward = torch.clamp(reward, min=self.min_reward)
            log_reward = reward.log()
        else:
            log_reward = signed_value * self.beta
            log_reward = torch.clamp(log_reward, min=self.min_log_reward)
            reward = log_reward.exp()
        return RewardOutput(
            log_reward=log_reward,
            reward=reward,
            proxy=value,
            proxy_components=proxy_output.components,
        )
