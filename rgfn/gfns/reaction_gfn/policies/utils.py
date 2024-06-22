from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch_geometric.utils import to_dense_batch
from torchtyping import TensorType

from rgfn.gfns.reaction_gfn.api.data_structures import Molecule, Reaction


def to_dense_embeddings(
    embeddings: torch.Tensor,
    counts: Sequence[int],
    fill_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts sparse node embeddings to dense node embeddings with padding.
    Arguments:
        embeddings: embeddings in a sparse format, i.e. [total_num_nodes, hidden_size]
        counts: the number of nodes in each graph, i.e. [batch_size]
        fill_value: a value to fill the padding with

    Returns:
        node_embeddings: embeddings in a dense format, i.e. [batch_size, max_num_nodes or max_num_edges, hidden_size]
        mask: a mask indicating which nodes are real and which are padding, i.e. [batch_size, max_num_nodes]
    """
    counts = (
        torch.tensor(counts, device=embeddings.device)
        if not isinstance(counts, torch.Tensor)
        else counts
    )
    indices = torch.arange(len(counts), device=embeddings.device)
    batch = torch.repeat_interleave(indices, counts).long()  # e.g. [0, 0, 1, 1, 1, 2, 2, 2]
    return to_dense_batch(
        embeddings, batch, fill_value=fill_value
    )  # that's the only reason we have torch_geometric in the requirements


def one_hot(idx: int, num_classes: int) -> List[int]:
    x = [0] * num_classes
    if idx == -1:
        return x
    x[idx] = 1
    return x