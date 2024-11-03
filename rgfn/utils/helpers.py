import heapq
import random
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterator, List, Literal, Set

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ComparableTuple:
    value: float
    item: Hashable

    def __lt__(self, other: "ComparableTuple") -> bool:
        return self.value < other.value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ComparableTuple):
            return False
        return self.value == other.value


class ContentHeap:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.heap: List[ComparableTuple] = []
        self.items: Set[Any] = set()

    def __len__(self) -> int:
        return len(self.heap)

    def push(self, value: float, item: Hashable) -> None:
        if item in self.items:
            return None
        self.items.add(item)
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, ComparableTuple(value, item))
        else:
            t = heapq.heappushpop(self.heap, ComparableTuple(value, item))
            self.items.remove(t.item)

    def __iter__(self) -> Iterator[ComparableTuple]:
        return iter(self.heap)


def to_indices(counts: Tensor) -> Tensor:
    indices = torch.arange(len(counts), device=counts.device)
    return torch.repeat_interleave(indices, counts).long()


def dict_mean(dict_list: List[Dict[str, float]]) -> Dict[str, float]:
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def infer_metric_direction(metric_name: str) -> Literal["min", "max"]:
    if metric_name.startswith("loss"):
        return "min"
    elif "acc" in metric_name:
        return "max"
    elif "auroc" in metric_name:
        return "max"
    elif "mrr" in metric_name:
        return "max"
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
