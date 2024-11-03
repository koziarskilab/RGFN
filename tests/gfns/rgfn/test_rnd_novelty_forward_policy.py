# can return sensible log probs for any trajectory
import torch
from gfns.helpers.policy_test_helpers import (
    helper__test_forward_policy__samples_only_allowed_actions,
)

from rgfn import RandomSampler, UniformPolicy
from rgfn.gfns.reaction_gfn.policies.rnd_novelty_forward_policy import (
    RNDNoveltyForwardPolicy,
)
from rgfn.utils.helpers import seed_everything

from .fixtures import *


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__rnd_novelty_forward_policy__samples_only_allowed_actions(
    rgfn_env: ReactionEnv, rgfn_data_factory: ReactionDataFactory, n_trajectories: int
):
    helper__test_forward_policy__samples_only_allowed_actions(
        RNDNoveltyForwardPolicy(data_factory=rgfn_data_factory), rgfn_env, n_trajectories
    )


@pytest.mark.parametrize("n_trajectories", [1, 10, 100])
@pytest.mark.parametrize("num_layers", [3, 5])
def test__rnd_novelty_policy__updates_properly(
    rgfn_env: ReactionEnv,
    rgfn_data_factory: ReactionDataFactory,
    n_trajectories: int,
    num_layers: int,
):
    """
    Test whether the ReinventPriorProxy updates properly.
    """
    seed_everything(42)
    policy = RNDNoveltyForwardPolicy(data_factory=rgfn_data_factory, num_layers=num_layers, lr=1e-3)
    sampler = RandomSampler(
        policy=UniformPolicy(),
        env=rgfn_env,
        reward=None,
    )

    trajectories = sampler.sample_trajectories(n_trajectories=n_trajectories)
    states = trajectories.get_non_last_states_flat()
    action_spaces = trajectories.get_forward_action_spaces_flat()
    actions = trajectories.get_actions_flat()

    novelty_scores_old = policy.compute_state_action_novelty(states, action_spaces, actions)

    for i in range(3):
        policy.on_end_computing_objective(iteration_idx=i, trajectories=trajectories)

    novelty_scores_new = policy.compute_state_action_novelty(states, action_spaces, actions)
    assert torch.all(novelty_scores_old >= novelty_scores_new)
