from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import Tensor


@dataclass
class RewardOutput:
    """
    The class to store the output obtained by calculating the reward on a batch of states. It contains the log rewards,
        rewards, proxy values, and possibly some components of the proxy values that may be used to compute metrics.

    Attributes:
        log_reward: The log rewards.
        reward: The rewards.
        proxy: The proxy values.
        proxy_components: A dictionary of components of the proxy values. It is used to compute metrics. If None, the
            proxy values have no components.
    """

    log_reward: Tensor
    reward: Tensor
    proxy: Tensor
    proxy_components: Dict[str, Tensor] | None = None

    def set_device(self, device: str, recursive: bool = True):
        """
        Set the device on which to perform the computations.

        Args:
            device: a string representing the device.

        Returns:
            None
        """
        self.log_reward = self.log_reward.to(device)
        self.reward = self.reward.to(device)
        self.proxy = self.proxy.to(device)
        if self.proxy_components is not None:
            for key, value in self.proxy_components.items():
                self.proxy_components[key] = value.to(device)

    @classmethod
    def from_list(cls, items: List["RewardOutput"]) -> "RewardOutput":
        """
        Concatenate a list of RewardOutput objects into a single RewardOutput object.
        Used in `Trajectories.from_trajectories` method.

        Args:`
            items: a list of RewardOutput objects.

        Returns:
            a new RewardOutput object that is the concatenation of the input items.
        """
        proxy_components = [
            item.proxy_components for item in items if item.proxy_components is not None
        ]
        if len(proxy_components) == 0:
            new_proxy_components = None
        elif len(proxy_components) == len(items):
            new_proxy_components = {}
            for key in proxy_components[0].keys():
                new_proxy_components[key] = torch.cat([item[key] for item in proxy_components])
        else:
            raise ValueError("Some items have proxy components and some don't!")
        return cls(
            log_reward=torch.cat([item.log_reward for item in items]),
            reward=torch.cat([item.reward for item in items]),
            proxy=torch.cat([item.proxy for item in items]),
            proxy_components=new_proxy_components,
        )

    def masked_select(self, mask: Tensor) -> "RewardOutput":
        """
        Select a subset of the RewardOutput object using a boolean mask. Used in `Trajectories.masked_select` method.

        Args:
            mask: a boolean mask of shape `(n,)` where `n` is the number of elements in the RewardOutput object.

        Returns:
            a new RewardOutput object that contains only the elements selected by the mask.
        """
        if self.proxy_components is not None:
            proxy_components = {
                key: value[mask].clone() for key, value in self.proxy_components.items()
            }
        else:
            proxy_components = None
        return RewardOutput(
            log_reward=self.log_reward[mask].clone(),
            reward=self.reward[mask].clone(),
            proxy=self.proxy[mask].clone(),
            proxy_components=proxy_components,
        )
