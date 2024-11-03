from typing import Dict, Hashable, List, TypeVar

import gin
import torch
from torch.distributions import Categorical
from torchtyping import TensorType

from rgfn.api.policy_base import PolicyBase
from rgfn.api.trajectories import Trajectories
from rgfn.api.type_variables import TState
from rgfn.shared.policies.uniform_policy import TIndexedActionSpace

THashableAction = TypeVar("THashableAction", bound=Hashable)


@gin.configurable()
class ActionCountPolicy(PolicyBase[TState, TIndexedActionSpace, THashableAction]):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.actions_count: Dict[THashableAction, int] = {}
        self.temperature = temperature
        self.device = "cpu"

    def _forward(self, action_spaces: List[TIndexedActionSpace]) -> TensorType[float]:
        max_action_space_size = max(
            action_space.get_possible_actions_indices()[-1] for action_space in action_spaces
        )
        action_scores_list = []
        for action_space in action_spaces:
            possible_actions_indices = action_space.get_possible_actions_indices()
            possible_actions_counts = [
                self.actions_count.get(action_space.get_action_at_idx(idx), 0)
                for idx in possible_actions_indices
            ]
            total_count = max(1, sum(possible_actions_counts))

            actions_scores = [-float("inf")] * (max_action_space_size + 1)
            for idx, count in zip(possible_actions_indices, possible_actions_counts):
                actions_scores[idx] = -count / total_count
            action_scores_list.append(actions_scores)

        logits = torch.tensor(action_scores_list, dtype=torch.float32, device=self.device)
        log_probs = torch.log_softmax(logits * self.temperature, dim=1)
        return log_probs

    def sample_actions(
        self, states: List[TState], action_spaces: List[TIndexedActionSpace]
    ) -> List[THashableAction]:
        log_probs = self._forward(action_spaces)
        action_indices = Categorical(probs=torch.exp(log_probs)).sample()
        return [
            action_space.get_action_at_idx(idx.item())
            for action_space, idx in zip(action_spaces, action_indices)
        ]

    def compute_action_log_probs(
        self,
        states: List[TState],
        action_spaces: List[TIndexedActionSpace],
        actions: List[THashableAction],
    ) -> TensorType[float]:
        log_probs = self._forward(action_spaces)
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

    def compute_states_log_flow(self, states: List[TState]) -> TensorType[float]:
        pass

    def on_end_computing_objective(
        self, iteration_idx: int, trajectories: Trajectories, recursive: bool = True
    ) -> Dict[str, float]:
        for action in trajectories.get_actions_flat():
            self.actions_count[action] = self.actions_count.get(action, 0) + 1
        return {}
