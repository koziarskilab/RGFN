# can return sensible log probs for any trajectory
from gfns.helpers.policy_test_helpers import (
    helper__test_backward_policy__returns_sensible_log_probs,
    helper__test_backward_policy__samples_only_allowed_actions,
)

from rgfn.gfns.reaction_gfn.policies.reaction_backward_policy import (
    ReactionBackwardPolicy,
)

from .fixtures import *


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__rgfn_backward_policy__samples_only_allowed_actions(
    rgfn_backward_policy: ReactionBackwardPolicy, rgfn_env: ReactionEnv, n_trajectories: int
):
    helper__test_backward_policy__samples_only_allowed_actions(
        rgfn_backward_policy, rgfn_env, n_trajectories, sample_directly_from_reversed_env=False
    )


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__rgfn_backward_policy__returns_sensible_log_probs(
    rgfn_backward_policy: ReactionBackwardPolicy, rgfn_env: ReactionEnv, n_trajectories: int
):
    helper__test_backward_policy__returns_sensible_log_probs(
        rgfn_backward_policy, rgfn_env, n_trajectories
    )
