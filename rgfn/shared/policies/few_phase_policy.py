import abc
from functools import singledispatch
from typing import Callable, Dict, Generic, List, Sequence, Tuple, Type, TypeVar

import torch
from torch.distributions import Categorical
from torchtyping import TensorType

from rgfn.api.env_base import TAction, TState
from rgfn.api.policy_base import PolicyBase
from rgfn.shared.policies.uniform_policy import TIndexedActionSpace

TSharedEmbeddings = TypeVar("TSharedEmbeddings")


class FewPhasePolicyBase(
    PolicyBase[TState, TIndexedActionSpace, TAction],
    Generic[TState, TIndexedActionSpace, TAction, TSharedEmbeddings],
    abc.ABC,
):
    @abc.abstractmethod
    def get_shared_embeddings(
        self, states: List[TState], action_spaces: List[TIndexedActionSpace]
    ) -> TSharedEmbeddings:
        ...

    @property
    @abc.abstractmethod
    def action_space_to_forward_fn(
        self,
    ) -> Dict[
        Type[TIndexedActionSpace],
        Callable[[List[TState], List[TIndexedActionSpace], TSharedEmbeddings], TensorType[float]],
    ]:
        ...

    @property
    @abc.abstractmethod
    def device(self) -> str:
        ...

    def sample_actions(
        self, states: List[TState], action_spaces: List[TIndexedActionSpace]
    ) -> List[TAction]:
        shared_embeddings = self.get_shared_embeddings(states, action_spaces)

        actions = []
        action_to_state_idx = []
        for action_space_type, forward_fn in self.action_space_to_forward_fn.items():
            phase_indices = [
                idx
                for idx, action_space in enumerate(action_spaces)
                if isinstance(action_space, action_space_type)
            ]
            if len(phase_indices) == 0:
                continue
            phase_states = [states[idx] for idx in phase_indices]
            phase_action_spaces = [action_spaces[idx] for idx in phase_indices]

            log_probs = forward_fn(phase_states, phase_action_spaces, shared_embeddings)
            phase_actions = self._sample_actions_from_log_probs(log_probs, phase_action_spaces)
            actions.extend(phase_actions)
            action_to_state_idx.extend(phase_indices)

        state_to_action_idx = [0] * len(states)
        for action_idx, state_idx in enumerate(action_to_state_idx):
            state_to_action_idx[state_idx] = action_idx

        return [actions[state_to_action_idx[state_idx]] for state_idx in range(len(states))]

    def _sample_actions_from_log_probs(
        self, log_probs: TensorType[float], action_spaces: List[TIndexedActionSpace]
    ) -> List[TAction]:
        """
        A helper function to sample actions from the log probabilities.

        Args:
            log_probs: log probabilities of the shape (N, max_num_actions)
            action_spaces: the list of action spaces of the length N.

        Returns:
            the list of sampled actions.
        """
        action_indices = Categorical(probs=torch.exp(log_probs)).sample()
        return [
            action_space.get_action_at_idx(idx.item())
            for action_space, idx in zip(action_spaces, action_indices)
        ]

    def compute_action_log_probs(
        self, states: List[TState], action_spaces: List[TIndexedActionSpace], actions: List[TAction]
    ) -> TensorType[float]:
        shared_embeddings = self.get_shared_embeddings(states, action_spaces)

        log_probs_list = []
        log_probs_to_state_idx = []
        for action_space_type, forward_fn in self.action_space_to_forward_fn.items():
            phase_indices = [
                idx
                for idx, action_space in enumerate(action_spaces)
                if isinstance(action_space, action_space_type)
            ]
            if len(phase_indices) == 0:
                continue

            phase_states = [states[idx] for idx in phase_indices]
            phase_action_spaces = [action_spaces[idx] for idx in phase_indices]
            phase_actions = [actions[idx] for idx in phase_indices]
            log_probs = forward_fn(phase_states, phase_action_spaces, shared_embeddings)
            phase_log_probs = self._select_actions_log_probs(
                log_probs, phase_action_spaces, phase_actions
            )
            log_probs_list.append(phase_log_probs)
            log_probs_to_state_idx.extend(phase_indices)

        log_probs = torch.cat(log_probs_list, dim=0)
        state_to_action_idx = torch.empty(len(states), dtype=torch.long)
        for action_idx, state_idx in enumerate(log_probs_to_state_idx):
            state_to_action_idx[state_idx] = action_idx
        state_to_action_idx = state_to_action_idx.to(self.device)

        return torch.index_select(log_probs, index=state_to_action_idx, dim=0).to(self.device)

    def _select_actions_log_probs(
        self,
        log_probs: TensorType[float],
        action_spaces: Sequence[TIndexedActionSpace],
        actions: Sequence[TAction],
    ) -> TensorType[float]:
        """
        A helper function to select the log probabilities of the actions.

        Args:
            log_probs: log probabilities of the shape (N, max_num_actions)
            action_spaces: the list of action spaces of the length N.
            actions: the list of chosen actions of the length N.

        Returns:
            the log probabilities of the chosen actions of the shape (N,).
        """
        action_indices = [
            action_space.get_idx_of_action(action)  # type: ignore
            for action_space, action in zip(action_spaces, actions)
        ]
        max_num_actions = log_probs.shape[1]
        action_indices = [
            idx * max_num_actions + action_idx for idx, action_idx in enumerate(action_indices)
        ]
        action_tensor_indices = torch.tensor(action_indices).long().to(self.device)
        log_probs = torch.index_select(log_probs.view(-1), index=action_tensor_indices, dim=0)
        return log_probs