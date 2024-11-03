from typing import Dict, List

import gin
import torch
from torch import Tensor

from rgfn.api.proxy_base import ProxyBase
from rgfn.shared.proxies.cached_proxy import CachedProxyBase, THashableState


@gin.configurable()
class ComposedProxy(CachedProxyBase[THashableState]):
    """
    A proxy that is a composition of other proxies. It computes the output of each proxy and then aggregates them
    using a specified aggregation method.
    """

    def __init__(
        self,
        proxies_dict: Dict[str, ProxyBase[THashableState]],
        weight_dict: Dict[str, float],
        aggregation: str = "weighted_mean",
    ):
        super().__init__()
        assert aggregation in ["sum", "weighted_mean", "min", "prod"]
        assert set(proxies_dict.keys()) == set(weight_dict.keys())
        assert all(weight >= 0 for weight in weight_dict.values())
        assert all(proxy.is_non_negative for proxy in proxies_dict.values())
        assert all(proxy.higher_is_better for proxy in proxies_dict.values())

        self.proxies_dict = proxies_dict
        self.weight_dict = weight_dict
        self.aggregation = aggregation
        if self.aggregation == "weighted_mean":
            weight_sum = sum(weight_dict.values())
            self.weight_dict = {key: weight / weight_sum for key, weight in weight_dict.items()}

    @property
    def hook_objects(self) -> List["TrainingHooksMixin"]:
        return [proxy for proxy in self.proxies_dict.values()]

    @property
    def is_non_negative(self) -> bool:
        """
        Whether the proxy values are non-negative.
        """
        return True

    @property
    def higher_is_better(self) -> bool:
        """
        Whether higher proxy values are "better".
        """
        return True

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
        proxy_values_dict: Dict[str, Tensor] = {}
        for proxy_name, proxy in self.proxies_dict.items():
            proxy_values_dict[proxy_name] = proxy.compute_proxy_output(states).value

        values_tensor = torch.stack(
            [
                proxy_values_dict[proxy_name] * weight
                for proxy_name, weight in self.weight_dict.items()
            ],
            dim=-1,
        )
        if self.aggregation in ["sum", "weighted_mean"]:
            values = values_tensor.sum(dim=-1)
        elif self.aggregation == "min":
            values = values_tensor.min(dim=-1).values
        elif self.aggregation == "prod":
            values = values_tensor.prod(dim=-1)
        else:
            raise ValueError(f"Aggregation method {self.aggregation} not supported.")

        proxy_values_dict["value"] = values
        proxy_values_list_dict = {
            proxy_name: proxy_values_dict[proxy_name].tolist() for proxy_name in proxy_values_dict
        }

        return [
            {key: proxy_values_list_dict[key][i] for key in proxy_values_list_dict}
            for i in range(len(states))
        ]
