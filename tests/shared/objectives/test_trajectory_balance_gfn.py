import pytest
import torch
from shared.objectives.test_subtrajectory_balance_gfn import MockPolicy, MockProxy

from rgfn.api.reward import Reward
from rgfn.api.trajectories import Trajectories, TrajectoriesContainer
from rgfn.shared.objectives import TrajectoryBalanceObjective


@pytest.fixture()
def reward() -> Reward:
    return Reward(proxy=MockProxy(), beta=1.0, reward_boosting="exponential", min_reward=0.0)


@pytest.fixture()
def objective():
    policy = MockPolicy()
    return TrajectoryBalanceObjective(
        forward_policy=policy,
        backward_policy=policy,
    )


def test__trajectory_balance_gfn__single_trajectory(
    reward: Reward, objective: TrajectoryBalanceObjective
):
    trajectories = Trajectories()
    trajectories._states_list = [[0, 2, 4, 6]]
    trajectories._forward_action_spaces_list = [[[0, 1, 2], [0, 1, 2], [0, 1, 2]]]
    trajectories._backward_action_spaces_list = [[[0, 1, 2], [0, 1, 2], [0, 1, 2]]]
    trajectories._actions_list = [[2, 2, 2]]
    trajectories._reward_outputs = reward.compute_reward_output(trajectories.get_last_states_flat())
    trajectories_container = TrajectoriesContainer(
        forward_trajectories=trajectories,
    )
    loss = objective.compute_objective_output(trajectories_container).loss
    expected_loss = torch.tensor(1.8057)

    assert torch.isclose(loss, expected_loss, rtol=1e-4)


def test__trajectory_balance_gfn__many_trajectories(
    reward: Reward, objective: TrajectoryBalanceObjective
):
    trajectories = Trajectories()
    trajectories._states_list = [[0, 2, 4, 6], [0, 3, 6]]
    trajectories._forward_action_spaces_list = [
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        [[0, 1, 2], [0, 1, 2]],
    ]
    trajectories._backward_action_spaces_list = [
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        [[0, 1, 2], [0, 1, 2]],
    ]
    trajectories._actions_list = [[2, 2, 2], [3, 3]]
    trajectories._reward_outputs = reward.compute_reward_output(trajectories.get_last_states_flat())
    trajectories_container = TrajectoriesContainer(
        forward_trajectories=trajectories,
    )
    loss = objective.compute_objective_output(trajectories_container).loss
    expected_loss = torch.tensor(1.8057)

    assert torch.isclose(loss, expected_loss, rtol=1e-4)
