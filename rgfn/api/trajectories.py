import itertools
from typing import Any, Generic, List

import torch
from torch import Tensor

from rgfn.api.reward_output import RewardOutput
from rgfn.api.type_variables import TAction, TActionSpace, TState


class Trajectories(Generic[TState, TActionSpace, TAction]):
    """
    Class that stores a batch GFN trajectories. Each trajectory is a sequence of states and actions. Every state has a
    corresponding forward and backward action space which describe the possible actions that can be taken from that
    state. The class can also store the rewards for each trajectory and the forward and backward log probabilities
    of the actions taken in the trajectories (plus the log flow log(F(s)) of the states). The number of trajectories
    in the batch, denoted `n_trajectories` is equal to the number of source states added with `add_source_states` method.
    We denote the total number of states in the batch as `n_total_states` and the total number of actions as `n_actions`.
    Note that `n_total_actions` = `n_total_states` - `n_trajectories`. The total number of forward and backward action
    spaces is equal to `n_total_actions`.

    Type parameters:
        TState: The type of the states.
        TActionSpace: The type of the action spaces.
        TAction: The type of the actions.
    """

    def __init__(self) -> None:
        self._states_list: List[List[TState]] = []
        self._forward_action_spaces_list: List[List[TActionSpace]] = []
        self._backward_action_spaces_list: List[List[TActionSpace]] = []
        self._actions_list: List[List[TAction]] = []
        self._reward_outputs: RewardOutput | None = None
        self._forward_log_probs_flat: Tensor | None = None
        self._backward_log_probs_flat: Tensor | None = None
        self._log_flows_flat: Tensor | None = None

    def __len__(self):
        """
        Return the number of trajectories in the batch.

        Returns:
            number of trajectories in the batch.
        """
        return len(self._states_list)

    def add_source_states(self, source_states: List[TState]) -> None:
        """
        Add source states to the trajectories. The source states are the states from which the trajectories start.

        Args:
            source_states: a list of source states of length `n_states`.

        Returns:
            None
        """
        self._states_list = [[source_state] for source_state in source_states]
        self._forward_action_spaces_list = [[] for _ in range(len(source_states))]
        self._backward_action_spaces_list = [[] for _ in range(len(source_states))]
        self._actions_list = [[] for _ in range(len(source_states))]

    def add_actions_states(
        self,
        actions: List[TAction],
        states: List[TState],
        forward_action_spaces: List[TActionSpace],
        backward_action_spaces: List[TActionSpace],
        not_terminated_mask: List[bool] | None = None,
    ) -> None:
        """
        It extends the not-terminated trajectories with the new actions leading to the new states.

        Args:
            actions: a list of actions of length `n_new_states`.
            states: a list of states of length `n_new_states`.
            forward_action_spaces: a list of forward action spaces of length `n_new_states` from which the actions were
                chosen.
            backward_action_spaces: a list of backward action spaces of length `n_new_states` from which the actions
                may be chosen in the backward direction.
            not_terminated_mask: a list of booleans of length `n_trajectories` indicating which trajectories we want to
                extend. If None, all trajectories are extended. It sums to `n_new_states`.

        Returns:
            None
        """
        if not_terminated_mask is None:
            indices = list(range(len(states)))
        else:
            indices = [i for i, mask in enumerate(not_terminated_mask) if mask]
        for out_idx, in_idx in enumerate(indices):
            self._states_list[in_idx].append(states[out_idx])
            self._forward_action_spaces_list[in_idx].append(forward_action_spaces[out_idx])
            self._backward_action_spaces_list[in_idx].append(backward_action_spaces[out_idx])
            self._actions_list[in_idx].append(actions[out_idx])

    # def get_all_states_flat(self) -> List[TState]:
    #     return [state for states in self._states_list for state in states]
    #
    def get_source_states_flat(self) -> List[TState]:
        return [states[0] for states in self._states_list]

    def get_last_states_flat(self) -> List[TState]:
        """
        Return the last state (possibly a terminal one) of each trajectory in the batch.

        Returns:
            a list of the last states of each trajectory in the batch of length `n_trajectories`.
        """
        return [states[-1] for states in self._states_list]

    def get_non_last_states_flat(self) -> List[TState]:
        """
        Return all the states except the last one of each trajectory in the batch. The states are flattened.

        Returns:
            a list of all the states except the last one of each trajectory in the batch. The length of the list is
            equal to `n_total_states` - `n_trajectories` or `n_total_actions`.
        """
        return [state for states in self._states_list for state in states[:-1]]

    def get_non_source_states_flat(self) -> List[TState]:
        """
        Return all the states except the source states of each trajectory in the batch. The states are flattened.

        Returns:
            a list of all the states except the source states of each trajectory in the batch. The length of the list
            is equal to `n_total_states` - `n_trajectories` or `n_total_actions`.
        """
        return [state for states in self._states_list for state in states[1:]]

    def get_actions_flat(self) -> List[TAction]:
        """
        Return all the actions taken in the trajectories. The actions are flattened.

        Returns:
            a list of all the actions taken in the trajectories. The length of the list is equal to `n_total_actions`.
        """
        return [action for actions in self._actions_list for action in actions]

    def get_forward_action_spaces_flat(self) -> List[TActionSpace]:
        """
        Return all the forward action spaces of the trajectories. The action spaces are flattened.

        Returns:
            a list of all the forward action spaces of the trajectories. The length of the list is equal to
            `n_total_actions`.
        """
        return [
            action_space
            for action_spaces in self._forward_action_spaces_list
            for action_space in action_spaces
        ]

    def get_backward_action_spaces_flat(self) -> List[TActionSpace]:
        """
        Return all the backward action spaces of the trajectories. The action spaces are flattened.

        Returns:
            a list of all the backward action spaces of the trajectories. The length of the list is equal to
            `n_total_actions`.
        """
        return [
            action_space
            for action_spaces in self._backward_action_spaces_list
            for action_space in action_spaces
        ]

    def get_index_flat(self) -> Tensor:
        """
        Return a tensor of indices. i-th element of the tensor is the index of the trajectory to which the i-th action
            belongs.

        Returns:
            a tensor of indices. The length of the tensor is equal to `n_total_actions`.
        """
        actions_count = [len(actions) for actions in self._actions_list]
        sizes = torch.tensor(actions_count)
        indices = torch.arange(len(self._actions_list))
        return torch.repeat_interleave(indices, sizes).long()

    def set_reward_outputs(self, reward_outputs: RewardOutput) -> None:
        """
        Set the reward outputs of the trajectories.

        Args:
            reward_outputs: object containing the rewards of the trajectories. The length of the reward tensor should
                be equal to `n_trajectories`.

        Returns:
            None
        """
        self._reward_outputs = reward_outputs

    def get_reward_outputs(self) -> RewardOutput:
        """
        Return the reward outputs of the trajectories.

        Returns:
            object containing the rewards of the trajectories. The length of the reward tensor is equal to
            `n_trajectories`.
        """
        if self._reward_outputs is None:
            raise ValueError("Trajectories have no rewards")
        return self._reward_outputs

    def set_forward_log_probs_flat(self, forward_log_probs_flat: Tensor) -> None:
        """
        Set the forward log probabilities of the actions taken in the trajectories.

        Args:
            forward_log_probs_flat: a tensor of length `n_total_actions` containing the forward log probabilities

        Returns:
            None
        """
        self._forward_log_probs_flat = forward_log_probs_flat

    def get_forward_log_probs_flat(self) -> Tensor:
        """
        Return the forward log probabilities of the actions taken in the trajectories.

        Returns:
            a tensor of length `n_total_actions` containing the forward log probabilities
        """
        if self._forward_log_probs_flat is None:
            raise ValueError("Trajectories have no forward log probs")
        return self._forward_log_probs_flat

    def set_backward_log_probs_flat(self, backward_log_probs_flat: Tensor) -> None:
        """
        Set the backward log probabilities of the actions taken in the trajectories.

        Args:
            backward_log_probs_flat: a tensor of length `n_total_actions` containing the backward log probabilities

        Returns:
            None
        """
        self._backward_log_probs_flat = backward_log_probs_flat

    def get_backward_log_probs_flat(self) -> Tensor:
        """
        Return the backward log probabilities of the actions taken in the trajectories.

        Returns:
            a tensor of length `n_total_actions` containing the backward log probabilities
        """
        if self._backward_log_probs_flat is None:
            raise ValueError("Trajectories have no backward log probs")
        return self._backward_log_probs_flat

    def set_log_flows_flat(self, log_flows_flat: Tensor) -> None:
        """
        Set the log flows of the states in the trajectories.

        Args:
            log_flows_flat: a tensor of length `n_total_states` containing the log flows

        Returns:
            None
        """
        self._log_flows_flat = log_flows_flat

    def get_log_flows_flat(self) -> Tensor:
        """
        Return the log flows of the states in the trajectories.

        Returns:
            a tensor of length `n_total_states` containing the log flows
        """
        if self._log_flows_flat is None:
            raise ValueError("Trajectories have no log flows")
        return self._log_flows_flat

    def reversed(self) -> "Trajectories[TState, TActionSpace, TAction]":
        """
        Return the reversed trajectories. The states, actions and action spaces are properly reversed. If the original
        trajectories were obtained by backward sampling, the reversed trajectories can be treated as trajectories
        obtained by forward sampling.

        Returns:
            reversed trajectories
        """
        if self._reward_outputs is not None:
            raise ValueError("Cannot reverse trajectories with rewards")
        if self._backward_log_probs_flat is not None or self._forward_log_probs_flat is not None:
            raise ValueError("Cannot reverse trajectories with log probs")
        if self._log_flows_flat is not None:
            raise ValueError("Cannot reverse trajectories with log flows")

        trajectories: Trajectories[TState, TActionSpace, TAction] = Trajectories()
        trajectories._states_list = [list(reversed(states)) for states in self._states_list]
        trajectories._actions_list = [list(reversed(actions)) for actions in self._actions_list]
        _forward_action_spaces_list = [
            list(reversed(action_spaces)) for action_spaces in self._backward_action_spaces_list
        ]
        _backward_action_spaces_list = [
            list(reversed(action_spaces)) for action_spaces in self._forward_action_spaces_list
        ]
        trajectories._forward_action_spaces_list = _forward_action_spaces_list
        trajectories._backward_action_spaces_list = _backward_action_spaces_list
        return trajectories

    def masked_select(
        self, mask: Tensor | List[bool]
    ) -> "Trajectories[TState, TActionSpace, TAction]":
        """
        Select trajectories from the batch using the mask. It is used to select only the trajectories that are not
        terminated during the sampling.

        Args:
            mask: a boolean tensor of length `n_trajectories` indicating which trajectories to select.

        Returns:
            a new Trajectories object containing only the selected trajectories.
        """
        if all(mask):
            return self
        mask = torch.tensor(mask) if isinstance(mask, list) else mask
        trajectories: Trajectories[TState, TActionSpace, TAction] = Trajectories()
        trajectories._states_list = list(itertools.compress(self._states_list, mask))
        trajectories._forward_action_spaces_list = list(
            itertools.compress(self._forward_action_spaces_list, mask)
        )
        trajectories._backward_action_spaces_list = list(
            itertools.compress(self._backward_action_spaces_list, mask)
        )
        trajectories._actions_list = list(itertools.compress(self._actions_list, mask))

        if self._reward_outputs is not None:
            trajectories._reward_outputs = self._reward_outputs.masked_select(mask)

        sizes = torch.tensor([len(actions) for actions in self._actions_list])
        flat_mask = torch.repeat_interleave(mask, sizes).bool()

        if self._forward_log_probs_flat is not None:
            flat_mask = flat_mask.to(self._forward_log_probs_flat.device)
            trajectories._forward_log_probs_flat = torch.masked_select(
                self._forward_log_probs_flat, flat_mask
            )
        if self._backward_log_probs_flat is not None:
            flat_mask = flat_mask.to(self._forward_log_probs_flat.device)
            trajectories._backward_log_probs_flat = torch.masked_select(
                self._backward_log_probs_flat, flat_mask
            )
        if self._log_flows_flat is not None:
            flat_mask = flat_mask.to(self._forward_log_probs_flat.device)
            trajectories._log_flows_flat = torch.masked_select(self._log_flows_flat, flat_mask)
        return trajectories

    @classmethod
    def from_trajectories(
        cls, trajectories_list: List["Trajectories[TState, TActionSpace, TAction]"]
    ) -> "Trajectories[TState, TActionSpace, TAction]":
        """
        Concatenate a list of Trajectories objects into a single Trajectories object.

        Args:
            trajectories_list: a list of Trajectories objects.

        Returns:
            a new Trajectories object that is the concatenation of the input trajectories.
        """
        if len(trajectories_list) == 1:
            return trajectories_list[0]

        trajectories: Trajectories[TState, TActionSpace, TAction] = Trajectories()

        for trajectory in trajectories_list:
            trajectories._states_list.extend(trajectory._states_list)
            trajectories._forward_action_spaces_list.extend(trajectory._forward_action_spaces_list)
            trajectories._backward_action_spaces_list.extend(
                trajectory._backward_action_spaces_list
            )
            trajectories._actions_list.extend(trajectory._actions_list)

        rewards_list = []
        forward_log_probs_list = []
        backward_log_probs_list = []
        for trajectory in trajectories_list:
            if trajectory._reward_outputs is not None:
                rewards_list.append(trajectory._reward_outputs)
            if trajectory._forward_log_probs_flat is not None:
                forward_log_probs_list.append(trajectory._forward_log_probs_flat)
            if trajectory._backward_log_probs_flat is not None:
                backward_log_probs_list.append(trajectory._backward_log_probs_flat)

        if rewards_list:
            trajectories._reward_outputs = RewardOutput.from_list(rewards_list)
        if forward_log_probs_list:
            trajectories._forward_log_probs_flat = torch.cat(forward_log_probs_list)
        if backward_log_probs_list:
            trajectories._backward_log_probs_flat = torch.cat(backward_log_probs_list)
        return trajectories

    def set_device(self, device: str, recursive: bool = True):
        """
        Set the device of the trajectories.

        Args:
            device: a string representing the device to use.

        Returns:
            None
        """
        self._forward_log_probs_flat = (
            self._forward_log_probs_flat.to(device)
            if self._forward_log_probs_flat is not None
            else None
        )
        self._backward_log_probs_flat = (
            self._backward_log_probs_flat.to(device)
            if self._backward_log_probs_flat is not None
            else None
        )
        self._log_flows_flat = (
            self._log_flows_flat.to(device) if self._log_flows_flat is not None else None
        )
        if self._reward_outputs is not None:
            self._reward_outputs.set_device(device)

    def __repr__(self):
        def _single_trajectory_repr(states: List[TState], actions: List[TAction]) -> str:
            rep = str(states[0])
            if not actions:
                return rep
            for state, action in zip(states[1:], actions):
                rep = f"{rep} ={action}=> {state}"
            return rep

        header = f"TrajectoriesList(n_trajectories={len(self._states_list)}):"
        content = "\n ".join(
            [
                _single_trajectory_repr(states, actions)
                for states, actions in itertools.zip_longest(
                    self._states_list[:10], self._actions_list[:10]
                )
            ]
        )
        return f"{header}\n {content}"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Trajectories):
            return (
                self._states_list == other._states_list
                and self._forward_action_spaces_list == other._forward_action_spaces_list
                and self._backward_action_spaces_list == other._backward_action_spaces_list
                and self._actions_list == other._actions_list
            )
        else:
            return False
