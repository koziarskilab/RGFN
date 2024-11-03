from itertools import compress
from typing import List

import gin
import numpy as np
from torch import Tensor

from rgfn.api.policy_base import PolicyBase
from rgfn.api.type_variables import TAction, TActionSpace, TState


@gin.configurable()
class ExploratoryPolicy(PolicyBase[TState, TActionSpace, TAction]):
    """
    A policy that samples actions from two different policies with a given probability.

    Type parameters:
        TState: The type of the states.
        TActionSpace: The type of the action spaces.
        TAction: The type of the actions.

    Attributes:
        first_policy: The first policy.
        second_policy: The second policy.
        first_policy_weight: The probability of sampling actions from the first policy.
    """

    def __init__(
        self,
        first_policy: PolicyBase[TState, TActionSpace, TAction],
        second_policy: PolicyBase[TState, TActionSpace, TAction],
        first_policy_weight: float,
    ):
        super().__init__()
        self.first_policy = first_policy
        self.second_policy = second_policy
        self.first_policy_weight = first_policy_weight

    def sample_actions(
        self, states: List[TState], action_spaces: List[TActionSpace]
    ) -> List[TAction]:
        """
        Sample actions from two different policies with a given probability.

        Args:
            states: a list of states of length `n_states`.
            action_spaces: a list of corresponding action spaces of length `n_states`.

        Returns:
            a list of sampled actions of length `n_states`.
        """
        first_policy_mask = np.random.binomial(1, self.first_policy_weight, len(states)).astype(
            bool
        )
        second_policy_mask = ~first_policy_mask

        first_policy_states = list(compress(states, first_policy_mask))
        first_policy_action_spaces = list(compress(action_spaces, first_policy_mask))
        second_policy_states = list(compress(states, second_policy_mask))
        second_policy_action_spaces = list(compress(action_spaces, second_policy_mask))

        if len(first_policy_states) > 0:
            first_policy_actions = self.first_policy.sample_actions(
                first_policy_states, first_policy_action_spaces
            )
        else:
            first_policy_actions = []

        if len(second_policy_states) > 0:
            second_policy_actions = self.second_policy.sample_actions(
                second_policy_states, second_policy_action_spaces
            )
        else:
            second_policy_actions = []

        actions = []
        first_policy_actions_index = 0
        second_policy_actions_index = 0
        for sample in first_policy_mask:
            if sample:
                actions.append(first_policy_actions[first_policy_actions_index])
                first_policy_actions_index += 1
            else:
                actions.append(second_policy_actions[second_policy_actions_index])
                second_policy_actions_index += 1

        return actions

    def compute_action_log_probs(
        self, states: List[TState], action_spaces: List[TActionSpace], actions: List[TAction]
    ) -> Tensor:
        raise NotImplementedError()

    def compute_states_log_flow(self, states: List[TState]) -> Tensor:
        raise NotImplementedError()
