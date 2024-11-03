import abc
from abc import ABC
from itertools import compress
from typing import Generic, Iterator, List

import torch

from rgfn.api.env_base import EnvBase
from rgfn.api.policy_base import PolicyBase
from rgfn.api.reward import Reward
from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.api.trajectories import Trajectories
from rgfn.api.type_variables import TAction, TActionSpace, TState


class SamplerBase(ABC, Generic[TState, TActionSpace, TAction], TrainingHooksMixin):
    """
    A base class for samplers. A sampler samples trajectories from the environment using a policy and assigns rewards to
    the trajectories.

    Type parameters:
        TState: The type of the states.
        TActionSpace: The type of the action spaces.
        TAction: The type of the actions.

    Attributes:
        policy: A policy that will be used sample actions.
        env: An environment that describes the transitions between states.
        reward: A reward function that assigns rewards to the terminal states.
    """

    def __init__(
        self,
        policy: PolicyBase[TState, TActionSpace, TAction],
        env: EnvBase[TState, TActionSpace, TAction],
        reward: Reward[TState] | None,
    ):
        super().__init__()
        self.policy = policy
        self.env = env
        self.reward = reward

    @property
    def hook_objects(self) -> List[TrainingHooksMixin]:
        hooks = [self.policy, self.env]
        if self.reward is not None:
            hooks.append(self.reward)
        return hooks

    @abc.abstractmethod
    def get_trajectories_iterator(
        self, n_total_trajectories: int, batch_size: int
    ) -> Iterator[Trajectories[TState, TActionSpace, TAction]]:
        """
        Get an iterator that samples trajectories from the environment. It can be used to sampled trajectories in
            batched manner.
        Args:
            n_total_trajectories: total number of trajectories to sample. If set to -1, the sampler should iterate over
                all source states (used in `SequentialSampler`).
            batch_size: the size of the batch. If -1, the batch size is equal to the number of n_total_trajectories.

        Returns:
            an iterator that samples trajectories.
        """
        ...

    @torch.no_grad()
    def sample_trajectories_from_sources(
        self, source_states: List[TState]
    ) -> Trajectories[TState, TActionSpace, TAction]:
        """
        Sample trajectories from the source states using the policy.

        Args:
            source_states: a list of source states of length `n`.

        Returns:
            a `Trajectories` object containing the sampled trajectories starting from source_states. The trajectories
             contain the visited states, forward and backward action spaces, actions, and rewards.
        """
        trajectories: Trajectories[TState, TActionSpace, TAction] = Trajectories()
        trajectories.add_source_states(source_states)
        while True:
            current_states = trajectories.get_last_states_flat()
            terminal_mask = self.env.get_terminal_mask(current_states)
            if all(terminal_mask):
                break
            non_terminal_mask = [not t for t in terminal_mask]
            non_terminal_states = list(compress(current_states, non_terminal_mask))

            forward_action_spaces = self.env.get_forward_action_spaces(non_terminal_states)
            new_actions = self.policy.sample_actions(non_terminal_states, forward_action_spaces)
            new_states = self.env.apply_forward_actions(non_terminal_states, new_actions)
            backward_action_spaces = self.env.get_backward_action_spaces(new_states)

            trajectories.add_actions_states(
                forward_action_spaces=forward_action_spaces,
                backward_action_spaces=backward_action_spaces,
                actions=new_actions,
                states=new_states,
                not_terminated_mask=non_terminal_mask,
            )
        if self.env.is_reversed:
            trajectories = trajectories.reversed()
        if self.reward is not None:
            reward_outputs = self.reward.compute_reward_output(trajectories.get_last_states_flat())
            trajectories.set_reward_outputs(reward_outputs)
        return trajectories
