import random
from typing import List

import pytest
import torch
from shared.objectives.test_subtrajectory_balance_gfn import MockProxy
from shared.policies.test_uniform_policy import IndexedList
from torch import Tensor

from rgfn.api.env_base import EnvBase
from rgfn.api.reward import Reward
from rgfn.api.trajectories import Trajectories
from rgfn.shared.policies.uniform_policy import UniformPolicy
from rgfn.shared.samplers.sequential_sampler import SequentialSampler


class MockEnv(EnvBase[int, IndexedList, int]):
    def __init__(self, max_component: int = 3, max_sum: int = 5, size: int = 20):
        super().__init__()
        self.max_component = max_component
        self.max_sum = max_sum
        self.size = size
        self.source_states = [0] * size
        self.terminal_states = [max_sum + random.randint(0, max_sum) for _ in range(self.size)]

    def get_forward_action_spaces(self, states: List[int]) -> List[IndexedList]:
        return [IndexedList(list(range(self.max_component + 1))) for _ in states]

    def get_backward_action_spaces(self, states: List[int]) -> List[IndexedList]:
        component_list = list(range(self.max_component + 1))
        return [IndexedList(component_list[: state + 1]) for state in states]

    def apply_forward_actions(self, states: List[int], actions: List[int]) -> List[int]:
        return [state + action for state, action in zip(states, actions)]

    def apply_backward_actions(self, states: List[int], actions: List[IndexedList]) -> List[int]:
        return [state - action for state, action in zip(states, actions)]

    def get_terminal_mask(self, states: List[int]) -> Tensor:
        return torch.tensor([state >= self.max_sum for state in states]).bool()

    def get_source_mask(self, states: List[int]) -> List[bool]:
        return [state == 0 for state in states]

    def sample_terminal_states(self, n_states: int) -> List[int]:
        indices = random.choices(range(self.size), k=n_states)
        return [self.terminal_states[i] for i in indices]

    def sample_source_states(self, n_states: int) -> List[int]:
        indices = random.choices(range(self.size), k=n_states)
        return [self.source_states[i] for i in indices]

    def get_num_source_states(self) -> int:
        return self.size

    def get_source_states_at_index(self, index: List[int]) -> List[int]:
        return [self.source_states[i] for i in index]

    def get_num_terminal_states(self) -> int:
        return self.size

    def get_terminal_states_at_index(self, index: List[int]) -> List[int]:
        return [self.terminal_states[i] for i in index]


@pytest.mark.parametrize("size", [10, 20])
@pytest.mark.parametrize("batch_size", [4, 16])
@pytest.mark.parametrize("n_repeats", [1, 3])
def test__sequential_sampler__forward_env(size: int, batch_size: int, n_repeats: int):
    env = MockEnv(size=size)
    sampler = SequentialSampler(
        policy=UniformPolicy(), env=env, reward=Reward(proxy=MockProxy()), n_repeats=n_repeats
    )

    trajectories_list = []
    for trajectories in sampler.get_trajectories_iterator(-1, batch_size):
        trajectories_list.append(trajectories)

    trajectories = Trajectories.from_trajectories(trajectories_list)
    last_states = trajectories.get_last_states_flat()
    assert env.get_terminal_mask(last_states).all()
    assert len(last_states) == size * n_repeats


@pytest.mark.parametrize("size", [10, 20])
@pytest.mark.parametrize("batch_size", [4, 16])
@pytest.mark.parametrize("n_repeats", [1, 3])
def test__sequential_sampler__backward_env(size: int, batch_size: int, n_repeats: int):
    env = MockEnv(size=size)
    sampler = SequentialSampler(
        policy=UniformPolicy(),
        env=env.reversed(),
        reward=Reward(proxy=MockProxy()),
        n_repeats=n_repeats,
    )

    trajectories_list = []
    for trajectories in sampler.get_trajectories_iterator(-1, batch_size):
        trajectories_list.append(trajectories)

    trajectories = Trajectories.from_trajectories(trajectories_list)

    last_states = trajectories.get_last_states_flat()
    assert env.get_terminal_mask(last_states).all()
    assert len(last_states) == size * n_repeats
