from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, List

from torch import Tensor

from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.api.type_variables import TState


@dataclass
class ProxyOutput(Generic[TState]):
    """
    The class to store the output obtained by calculating the proxy on a batch of states. It contains the proxy values
        that are used directly to compute the GFN reward and possibly some components of the proxy values that
        may be used to compute metrics.
    """

    value: Tensor
    components: Dict[str, Tensor] | None = None


class ProxyBase(Generic[TState], ABC, TrainingHooksMixin):
    """
    A base class for proxies. A proxy is a function that takes a batch of states and computes
        values that are then used to compute the GFN reward.

    Type parameters:
        TState: The type of the states.
    """

    @abstractmethod
    def compute_proxy_output(self, states: List[TState]) -> ProxyOutput:
        """
        Compute the proxy output on a batch of states.

        Args:
            states: a list of states of length `n_states`.

        Returns:
            a `ProxyOutput` object.

        """
        ...

    @property
    @abstractmethod
    def is_non_negative(self) -> bool:
        """
        Whether the proxy values are non-negative.
        """
        ...

    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        """
        Whether higher proxy values are "better".
        """
        ...
