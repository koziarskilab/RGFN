from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Generic, Iterator, List

from torch import Tensor, nn
from torch.nn import Parameter

from rgfn.api.policy_base import PolicyBase
from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.api.trajectories import Trajectories
from rgfn.api.type_variables import TAction, TActionSpace, TState


@dataclass
class ObjectiveOutput:
    """
    The class to store the output obtained by calculating the GFN objective on a batch of trajectories.

    Attributes:
        loss: The loss value.
        metrics: A dictionary of metrics.
    """

    loss: Tensor
    metrics: Dict[str, float] = field(default_factory=dict)


class ObjectiveBase(nn.Module, ABC, Generic[TState, TActionSpace, TAction], TrainingHooksMixin):
    """
    A base class for GFN objectives. An objective is a function that takes a batch of
        trajectories and computes the loss (objective) and possibly some metrics.

    Type parameters:
        TState: The type of the states.
        TActionSpace: The type of the action spaces.
        TAction: The type of the actions.

    Attributes:
        forward_policy: The forward policy that estimates the probabilities of actions taken in forward direction.
        backward_policy: The backward policy that estimates the probabilities of actions taken in backward direction.
        device: The device on which to perform the computations. It is set automatically with the set_device method.
    """

    def __init__(
        self,
        forward_policy: PolicyBase[TState, TActionSpace, TAction],
        backward_policy: PolicyBase[TState, TActionSpace, TAction],
    ):
        super().__init__()
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.device = "cpu"

    @property
    def hook_objects(self) -> List["TrainingHooksMixin"]:
        return [self.forward_policy, self.backward_policy]

    @abstractmethod
    def compute_objective_output(
        self, trajectories: Trajectories[TState, TActionSpace, TAction]
    ) -> ObjectiveOutput:
        """
        Compute the objective output on a batch of trajectories.

        Args:
            trajectories: the batch of trajectories obtained in the sampling process. It contains the states, actions,
                action spaces in forward and backward directions, and rewards. Other important quantities (e.g. log
                probabilities of taking actions in forward and backward directions) should be assigned in this method
                using appropriate methods (e.g. assign_log_probs).

        Returns:
            The output of the objective function, containing the loss and possibly some metrics.
        """
        ...

    def assign_log_probs(self, trajectories: Trajectories[TState, TActionSpace, TAction]) -> None:
        """
        Assign the log probabilities of taking actions in forward and backward directions to the trajectories.

        Args:
            trajectories: trajectories obtained in the sampling process. They will be modified in place.

        Returns:
            None
        """
        actions = trajectories.get_actions_flat()  # [n_actions]

        forward_states = trajectories.get_non_last_states_flat()  # [n_actions]
        forward_action_spaces = trajectories.get_forward_action_spaces_flat()  # [n_actions]
        forward_log_prob = self.forward_policy.compute_action_log_probs(
            states=forward_states, action_spaces=forward_action_spaces, actions=actions
        )  # [n_actions]

        backward_states = trajectories.get_non_source_states_flat()  # [n_actions]
        backward_action_spaces = trajectories.get_backward_action_spaces_flat()  # [n_actions]
        backward_log_prob = self.backward_policy.compute_action_log_probs(
            states=backward_states, action_spaces=backward_action_spaces, actions=actions
        )  # [n_actions]

        trajectories.set_forward_log_probs_flat(forward_log_prob)
        trajectories.set_backward_log_probs_flat(backward_log_prob)

    def assign_log_flows(self, trajectories: Trajectories[TState, TActionSpace, TAction]) -> None:
        """
        Assign the log flows of the states to the trajectories.

        Args:
            trajectories: trajectories obtained in the sampling process. They will be modified in place.

        Returns:
            None
        """
        states = trajectories.get_non_last_states_flat()  # [n_states]
        log_flow = self.forward_policy.compute_states_log_flow(states)  # [n_states]
        trajectories.set_log_flows_flat(log_flow)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Get the parameters of the objective function.

        Args:
            recurse: whether to recursively get the parameters of the submodules.

        Returns:
            An iterator over the parameters.
        """
        if isinstance(self.forward_policy, nn.Module):
            yield from self.forward_policy.parameters(recurse)
        if isinstance(self.backward_policy, nn.Module):
            yield from self.backward_policy.parameters(recurse)
