import abc
from abc import ABC
from typing import Generic, Iterator, List

from rgfn.api.sampler_base import SamplerBase
from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.api.trajectories import Trajectories
from rgfn.api.type_variables import TAction, TActionSpace, TState


class ReplayBufferBase(ABC, Generic[TState, TActionSpace, TAction], TrainingHooksMixin):
    """
    A base class for replay buffers. A replay buffer stores terminal states or trajectories and can sample them
    in backward direction using the provided sampler.

    Type parameters:
        TState: The type of the states.
        TActionSpace: The type of the action spaces.
        TAction: The type of the actions.
    """

    def __init__(self, sampler: SamplerBase[TState, TActionSpace, TAction]):
        super().__init__()
        assert sampler.env.is_reversed, "The environment should be reversed."
        self.sampler = sampler

    @property
    def hook_objects(self) -> List["TrainingHooksMixin"]:
        return [self.sampler]

    @abc.abstractmethod
    def add_trajectories(self, trajectories: Trajectories[TState, TActionSpace, TAction]):
        """
        Add trajectories to the replay buffer.

        Args:
            trajectories: trajectories to add.

        Returns:
            None
        """
        ...

    @abc.abstractmethod
    def get_trajectories_iterator(
        self, n_total_trajectories: int, batch_size: int
    ) -> Iterator[Trajectories]:
        """
        Get an iterator that samples trajectories from the environment. It can be used to sampled trajectories in
            batched manner.
        Args:
            n_total_trajectories: total number of trajectories to sample.
            batch_size: the size of the batch. If -1, the batch size is equal to the number of n_total_trajectories.

        Returns:
            an iterator that samples trajectories.
        """
        ...

    @abc.abstractmethod
    def state_dict(self) -> dict:
        """
        Return the state of the replay buffer as a dictionary.

        Returns:
            a dictionary containing the state of the replay buffer.
        """
        ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the state of the replay buffer from a dictionary.

        Args:
            state_dict: a dictionary containing the state of the replay buffer.

        Returns:
            None
        """
        ...
