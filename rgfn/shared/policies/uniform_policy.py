import abc
import random
from typing import Generic, List, TypeVar

import gin
import torch
from torchtyping import TensorType

from rgfn.api.policy_base import PolicyBase
from rgfn.api.type_variables import TAction, TState


class IndexedActionSpaceBase(abc.ABC, Generic[TAction]):
    """
    An abstract class for an indexed action space. It should be used as a base class for all action spaces for
    convenience. It enables easy use of the `UniformPolicy` policy.
    """

    @abc.abstractmethod
    def get_action_at_idx(self, idx: int) -> TAction:
        pass

    @abc.abstractmethod
    def get_idx_of_action(self, action: TAction) -> int:
        pass

    @abc.abstractmethod
    def get_possible_actions_indices(self) -> List[int]:
        pass

    def is_empty(self) -> bool:
        return len(self) == 0

    def __len__(self) -> int:
        return len(self.get_possible_actions_indices())

    def is_action_allowed(self, action: TAction) -> bool:
        return self.get_idx_of_action(action) in self.get_possible_actions_indices()


TIndexedActionSpace = TypeVar("TIndexedActionSpace", bound=IndexedActionSpaceBase)


@gin.configurable()
class UniformPolicy(PolicyBase[TState, TIndexedActionSpace, TAction]):
    """
    A uniform policy that samples actions uniformly from the action space.
    """

    def __init__(self):
        self.device = "cpu"

    def sample_actions(
        self, states: List[TState], action_spaces: List[TIndexedActionSpace]
    ) -> List[TAction]:
        """
        Sample actions uniformly for the given states and action spaces.

        Args:
            states: a list of states of length `n_states`.
            action_spaces: a list of indexed action spaces of length `n_states`.

        Returns:
            a list of actions of length `n_states`.
        """
        actions = []
        for action_space in action_spaces:
            possible_indices = action_space.get_possible_actions_indices()
            sampled_idx = random.choice(possible_indices)
            actions.append(action_space.get_action_at_idx(sampled_idx))
        return actions

    def compute_action_log_probs(
        self,
        states: List[TState],
        action_spaces: List[TIndexedActionSpace],
        actions: List[TAction],
    ) -> TensorType[float]:
        """
        Compute the log probabilities of the given actions take in the given states (and corresponding action spaces).

        Args:
            states: a list of states of length `n_states`.
            action_spaces: a list of action spaces of length `n_states`. An action space describes the possible actions
                that can be taken in a given state.
            actions: a list of actions chosen in the given states of length `n_states`.

        Returns:
            a tensor of log probabilities of shape `(n_states,)`.
        """
        action_space_lengths = torch.tensor(
            [len(action_space) for action_space in action_spaces],
            dtype=torch.float,
            device=self.device,
        )
        action_space_lengths = torch.clamp(action_space_lengths, min=1.0)
        return torch.log(1.0 / action_space_lengths)

    def compute_states_log_flow(self, states: List[TState]) -> TensorType[float]:
        raise NotImplementedError()
