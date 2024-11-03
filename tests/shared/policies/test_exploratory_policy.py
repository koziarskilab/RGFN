from typing import List

import pytest
from torchtyping import TensorType

from rgfn.api.policy_base import PolicyBase
from rgfn.shared.policies.exploratory_policy import ExploratoryPolicy
from rgfn.utils.helpers import seed_everything


class DeterministicPolicy(PolicyBase[int, int, int]):
    def __init__(self, value: int):
        self.value = value

    def sample_actions(self, states: List[int], action_spaces: List[int]) -> List[int]:
        return [self.value] * len(states)

    def compute_action_log_probs(
        self, states: List[int], action_spaces: List[int], actions: List[int]
    ) -> TensorType[float]:
        pass

    def compute_states_log_flow(self, states: List[int]) -> TensorType[float]:
        pass

    def set_device(self, device: str):
        pass

    def clear_sampling_cache(self) -> None:
        pass

    def clear_action_embedding_cache(self) -> None:
        pass


@pytest.mark.repeat(10)
def test__exploration_policy__returns_only_possible_actions():
    policy = ExploratoryPolicy(
        first_policy=DeterministicPolicy(0),
        second_policy=DeterministicPolicy(1),
        first_policy_weight=0.5,
    )
    states = [0, 1, 2, 3]
    action_spaces = [0] * len(states)
    actions = policy.sample_actions(states, action_spaces)
    for action, action_space in zip(actions, action_spaces):
        assert action in [0, 1]


def test_exploration_policy__covers_all_possible_actions():
    seed_everything(42)
    policy = ExploratoryPolicy(
        first_policy=DeterministicPolicy(0),
        second_policy=DeterministicPolicy(1),
        first_policy_weight=0.5,
    )
    states = [0, 1, 2, 3]
    action_spaces = [0] * len(states)
    all_sampled_actions = set()
    for _ in range(10):
        actions = policy.sample_actions(states, action_spaces)
        all_sampled_actions.update(actions)
    assert all_sampled_actions == {0, 1}


def test__exploration_policy__samples_in_proportion():
    seed_everything(42)
    n_repeats = 1000
    n_repeats_per_action = n_repeats // 2
    policy = ExploratoryPolicy(
        first_policy=DeterministicPolicy(0),
        second_policy=DeterministicPolicy(1),
        first_policy_weight=0.5,
    )
    states = [0]
    action_spaces = [0] * len(states)
    actions_count = {0: 0, 1: 0}
    for _ in range(n_repeats):
        actions = policy.sample_actions(states, action_spaces)
        for action in actions:
            actions_count[action] += 1
    assert abs(actions_count[0] - n_repeats_per_action) < 0.1 * n_repeats_per_action
    assert abs(actions_count[1] - n_repeats_per_action) < 0.1 * n_repeats_per_action
