import torch
from gfns.helpers.proxy_test_helpers import (
    helper__test_proxy__is_deterministic,
    helper__test_proxy__returns_sensible_values,
)

from rgfn import RandomSampler, UniformPolicy
from rgfn.gfns.reaction_gfn.proxies.rnd_novelty_proxy import RNDNoveltyProxy
from rgfn.utils.helpers import seed_everything

from ..fixtures import *


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__novelty_proxy__returns_sensible_values(rgfn_env: ReactionEnv, n_trajectories: int):
    """
    Test whether the ReinventPriorProxy returns sensible values for the sampled trajectories.
    """
    helper__test_proxy__returns_sensible_values(
        env=rgfn_env, proxy=RNDNoveltyProxy(), n_trajectories=n_trajectories
    )


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__novelty_proxy__is_deterministic(rgfn_env: ReactionEnv, n_trajectories: int):
    """
    Test whether the ReinventPriorProxy is deterministic.
    """
    helper__test_proxy__is_deterministic(env=rgfn_env, proxy=RNDNoveltyProxy(), n_trajectories=10)


@pytest.mark.parametrize("n_trajectories", [10])
@pytest.mark.parametrize("num_layers", [3, 5])
def test__novelty_proxy__updates_properly(
    rgfn_env: ReactionEnv, n_trajectories: int, num_layers: int
):
    """
    Test whether the ReinventPriorProxy updates properly.
    """
    seed_everything(42)
    proxy = RNDNoveltyProxy(lr=1e-3)
    sampler = RandomSampler(
        policy=UniformPolicy(),
        env=rgfn_env,
        reward=None,
    )

    trajectories = sampler.sample_trajectories(n_trajectories=n_trajectories)
    states = trajectories.get_last_states_flat()
    novelty_scores_old = proxy.compute_proxy_output(states).value

    for i in range(3):
        proxy.on_end_computing_objective(trajectories=trajectories, iteration_idx=i)

    novelty_scores_new = proxy.compute_proxy_output(states).value
    assert torch.all(novelty_scores_old >= novelty_scores_new)
