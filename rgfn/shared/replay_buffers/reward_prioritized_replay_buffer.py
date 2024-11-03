from typing import Hashable, Iterator, List, Set, TypeVar

import gin
import numpy as np
import torch
from more_itertools import chunked

from rgfn.api.replay_buffer_base import ReplayBufferBase
from rgfn.api.sampler_base import SamplerBase
from rgfn.api.trajectories import Trajectories
from rgfn.api.type_variables import TAction, TActionSpace

THashableState = TypeVar("THashableState", bound=Hashable)


@gin.configurable()
class RewardPrioritizedReplayBuffer(ReplayBufferBase[THashableState, TActionSpace, TAction]):
    """
    A replay buffer that stores terminal states and their proxy values. The proxy values are used to weight the
    probability of sampling a backward trajectory starting from a terminal state. The proxy values are multiplied by a
    temperature coefficient before applying the softmax function to get the probabilities. It is inspired by the
    "An Empirical Study of the Effectiveness of Using a Replay Buffer on Mode Discovery in GFlowNets" paper.

    Args:
        sampler: a sampler that samples trajectories from the environment. The environment within the sampler should
            be reversed.
        max_size: the maximum number of terminal states to store.
        temperature: the temperature coefficient that is multiplied by the proxy values before applying the softmax
            function.
    """

    def __init__(
        self,
        sampler: SamplerBase[THashableState, TActionSpace, TAction],
        max_size: int = int(1e6),
        temperature: float = 1.0,
        proxy_term: str | None = None,
    ):
        super().__init__(sampler)
        self.max_size = int(max_size)
        self.states_list: List[THashableState] = []
        self.states_set: Set[THashableState] = set()
        self.proxy_value_array = np.zeros((self.max_size + 1,), dtype=np.float32)
        self.temperature = temperature
        self.proxy_term = proxy_term

    def get_trajectories_iterator(
        self, n_total_trajectories: int, batch_size: int
    ) -> Iterator[Trajectories[THashableState, TActionSpace, TAction]]:
        """
        Get an iterator that samples trajectories from the environment. It can be used to sampled trajectories in
            batched manner.
        Args:
            n_total_trajectories: total number of trajectories to sample.
            batch_size: the size of the batch. If -1, the batch size is equal to the number of n_total_trajectories.

        Returns:
            an iterator that samples trajectories.
        """
        n_total_trajectories = min(n_total_trajectories, self.size)
        batch_size = n_total_trajectories if batch_size == -1 else batch_size

        if n_total_trajectories == 0:
            yield Trajectories()
            return

        if n_total_trajectories >= self.size:
            sampled_states = self.states_list
        else:
            logits = torch.tensor(self.proxy_value_array[: self.size] * self.temperature)
            probs = torch.nn.functional.softmax(logits, dim=0).numpy()
            n_total_trajectories = min(n_total_trajectories, np.sum(probs > 0))
            sampled_indices = np.random.choice(
                self.size, size=n_total_trajectories, replace=False, p=probs
            )
            sampled_states = [self.states_list[i] for i in sampled_indices]

        for sampled_states_chunk in chunked(sampled_states, batch_size):
            yield self.sampler.sample_trajectories_from_sources(sampled_states_chunk)

    def add_trajectories(self, trajectories: Trajectories[THashableState, TActionSpace, TAction]):
        """
        Add the terminal states from the trajectories to the replay buffer that are not already in the replay buffer.

        Args:
            trajectories: trajectories to get the terminal states from.

        Returns:
            None
        """
        terminal_states = trajectories.get_last_states_flat()
        reward_output = trajectories.get_reward_outputs()
        proxy_value = (
            reward_output.proxy
            if self.proxy_term is None
            else reward_output.proxy_components[self.proxy_term]
        )

        for state, proxy_value in zip(terminal_states, proxy_value):
            if state not in self.states_set:
                self._add_state(state, proxy_value.item())

    def state_dict(self) -> dict:
        return {
            "states_list": self.states_list,
            "proxy_value_array": self.proxy_value_array,
        }

    def load_state_dict(self, state_dict: dict):
        self.states_list = state_dict["states_list"]
        self.proxy_value_array = state_dict["proxy_value_array"]
        self.states_set = set(self.states_list)

    def _add_state(self, state: THashableState, proxy_value: float):
        self.proxy_value_array[self.size] = proxy_value
        self.states_list.append(state)
        self.states_set.add(state)
        if self.size > self.max_size:
            el = self.states_list.pop(0)
            self.states_set.remove(el)
            self.proxy_value_array[:-1] = self.proxy_value_array[1:]

    @property
    def size(self):
        return len(self.states_list)
