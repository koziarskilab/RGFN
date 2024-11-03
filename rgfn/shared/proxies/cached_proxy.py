import abc
from typing import Dict, Hashable, List, TypeVar, cast

import torch

from rgfn.api.proxy_base import ProxyBase, ProxyOutput

THashableState = TypeVar("THashableState", bound=Hashable)


class CachedProxyBase(ProxyBase[THashableState], abc.ABC):
    """
    A base class for cached proxies. It caches the results of the proxy computation to avoid redundant computations.
    """

    def __init__(self):
        super().__init__()
        self.cache: Dict[THashableState, Dict[str, float] | List[float]] = {}
        self.total_calls = 0
        self.device = "cpu"

    @property
    def n_proxy_calls(self) -> int:
        return len(self.cache)

    def clear_cache(self) -> None:
        self.cache = {}

    @abc.abstractmethod
    def _compute_proxy_output(
        self, states: List[THashableState]
    ) -> List[Dict[str, float]] | List[float]:
        """
        A method that computes the proxy output for the given states.

        Args:
            states: a list of states for which the proxy output should be computed

        Returns:
            a list of dictionaries containing the proxy output (along with components) for each state. The main proxy
            output should be stored under the key "value".
        """
        ...

    def compute_proxy_output(self, states: List[THashableState]) -> ProxyOutput:
        uncached_indices = [idx for idx, state in enumerate(states) if state not in self.cache]
        uncached_states = [states[idx] for idx in uncached_indices]
        if len(uncached_states) > 0:
            uncached_states = list(set(uncached_states))
            scores_dict_or_values_list = self._compute_proxy_output(uncached_states)
            for score_dict_or_value, state in zip(scores_dict_or_values_list, uncached_states):
                self.cache[state] = score_dict_or_value  # type: ignore
        scores_dict_or_values_list = [self.cache[state] for state in states]  # type: ignore

        if isinstance(scores_dict_or_values_list[0], float):
            cast(List[float], scores_dict_or_values_list)
            value = torch.tensor(scores_dict_or_values_list, dtype=torch.float, device=self.device)
            components = None
        else:
            cast(List[Dict[str, float]], scores_dict_or_values_list)
            component_dict = {}
            for key in scores_dict_or_values_list[0].keys():
                component_dict[key] = torch.tensor(
                    [d[key] for d in scores_dict_or_values_list], dtype=torch.float  # type: ignore
                )
            value = component_dict["value"].to(self.device)
            components = {key: component_dict[key] for key in component_dict if key != "value"}
        return ProxyOutput(value=value, components=components)

    def set_device(self, device: str, recursive: bool = True):
        self.device = device
