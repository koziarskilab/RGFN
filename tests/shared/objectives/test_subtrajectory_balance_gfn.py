import random
from typing import List

import pytest
import torch
from torch import Tensor

from rgfn.api.policy_base import PolicyBase
from rgfn.api.proxy_base import ProxyBase, ProxyOutput
from rgfn.api.reward import Reward
from rgfn.api.trajectories import Trajectories
from rgfn.shared.objectives import SubTrajectoryBalanceObjective


class MockProxy(ProxyBase[int]):
    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def set_device(self, device: str):
        pass

    def compute_proxy_output(self, states: List[int]) -> ProxyOutput:
        value = torch.ones(len(states)).float()
        return ProxyOutput(value=value)


class MockPolicy(PolicyBase[int, List[int], int]):
    def sample_actions(self, states: List[int], action_spaces: List[List[int]]) -> List[int]:
        return [random.choice(action_space) for action_space in action_spaces]

    def compute_action_log_probs(
        self, states: List[int], action_spaces: List[List[int]], actions: List[int]
    ) -> Tensor:
        return torch.tensor([1 / len(action_space) for action_space in action_spaces]).float()

    def compute_states_log_flow(self, states: List[int]) -> Tensor:
        return torch.tensor(states).float()

    def clear_sampling_cache(self) -> None:
        pass

    def clear_action_embedding_cache(self) -> None:
        pass

    def set_device(self, device: str):
        pass


@pytest.fixture()
def reward() -> Reward:
    return Reward(proxy=MockProxy(), beta=1.0, reward_boosting="exponential", min_reward=0.0)


@pytest.mark.parametrize(
    "lambda_coeff, expected_loss",
    [
        (1.0, 1.3519),
        (0.5, 1.3154),
    ],
)
def test__sub_trajectory_balance_gfn__single_trajectory(
    reward: Reward, lambda_coeff: float, expected_loss: float
):
    policy = MockPolicy()
    gfn = SubTrajectoryBalanceObjective(
        forward_policy=policy,
        backward_policy=policy,
        lambda_coeff=lambda_coeff,
    )

    trajectories = Trajectories()
    trajectories._states_list = [[0, 1, 2, 3]]
    trajectories._forward_action_spaces_list = [[[0, 1], [0, 1, 2], [0, 1]]]
    trajectories._backward_action_spaces_list = [[[0, 1, 2], [0, 1], [0, 1, 2]]]
    trajectories._actions_list = [[1, 2, 1]]
    trajectories._reward_outputs = reward.compute_reward_output(trajectories.get_last_states_flat())

    loss = gfn.compute_objective_output(trajectories).loss
    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-4)


@pytest.mark.parametrize("lambda_coeff", [1.0, 0.5])
@pytest.mark.parametrize("n_trajectories", [2, 5])
def test__sub_trajectory_balance_gfn__many_trajectories(
    reward: Reward, n_trajectories: int, lambda_coeff: float
):
    policy = MockPolicy()
    gfn = SubTrajectoryBalanceObjective(
        forward_policy=policy,
        backward_policy=policy,
        lambda_coeff=lambda_coeff,
    )

    def _get_random_trajectory():
        trajectories = Trajectories()
        n_actions = random.randint(1, 10)
        trajectories._actions_list = [[random.randint(0, 2) for _ in range(n_actions)]]
        trajectories._states_list = [[random.randint(0, 2) for _ in range(n_actions + 1)]]
        trajectories._forward_action_spaces_list = [
            [[0] * random.randint(1, 10) for _ in range(n_actions)]
        ]
        trajectories._backward_action_spaces_list = [
            [[0] * random.randint(1, 10) for _ in range(n_actions)]
        ]
        trajectories._reward_outputs = reward.compute_reward_output(
            trajectories.get_last_states_flat()
        )
        return trajectories

    trajectories_list = [_get_random_trajectory() for _ in range(n_trajectories)]
    trajectories_losses = [
        gfn.compute_objective_output(trajectories).loss for trajectories in trajectories_list
    ]
    expected_loss = torch.mean(torch.stack(trajectories_losses))

    trajectories = Trajectories.from_trajectories(trajectories_list)
    loss = gfn.compute_objective_output(trajectories).loss

    assert torch.isclose(loss, expected_loss)
