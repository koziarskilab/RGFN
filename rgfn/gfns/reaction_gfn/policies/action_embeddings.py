import abc
import math
from typing import Any, Dict, List

import dgl
import gin
import numpy as np
import torch
from dgllife.model import GAT, GCN
from dgllife.utils import (
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    SMILESToBigraph,
)
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch import Tensor, nn
from torch.nn import init

from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.api.trajectories import Trajectories
from rgfn.gfns.reaction_gfn.api.reaction_data_factory import ReactionDataFactory


class ActionEmbeddingBase(abc.ABC, nn.Module, TrainingHooksMixin):
    def __init__(self, data_factory: ReactionDataFactory, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.data_factory = data_factory
        self._cache: Tensor | None = None

    def get_embeddings(self) -> Tensor:
        if self._cache is None:
            self._cache = self._get_embeddings()
        return self._cache

    def clear_cache(self):
        self._cache = None

    def on_start_sampling(self, iteration_idx: int, recursive: bool = True) -> Dict[str, Any]:
        self._cache = None
        return {}

    def on_end_sampling(
        self, iteration_idx: int, trajectories: Trajectories, recursive: bool = True
    ) -> Dict[str, Any]:
        self._cache = None
        return {}

    @abc.abstractmethod
    def _get_embeddings(self) -> Tensor:
        pass


@gin.configurable()
class FragmentOneHotEmbedding(ActionEmbeddingBase):
    def __init__(self, data_factory: ReactionDataFactory, hidden_dim: int = 64):
        super().__init__(data_factory, hidden_dim)
        self.weights = nn.Parameter(
            torch.empty(len(data_factory.get_fragments()), hidden_dim), requires_grad=True
        )
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def _get_embeddings(self) -> Tensor:
        return self.weights


@gin.configurable()
class ReactionsOneHotEmbedding(ActionEmbeddingBase):
    def __init__(self, data_factory: ReactionDataFactory, hidden_dim: int = 64):
        super().__init__(data_factory, hidden_dim)
        self.weights = nn.Parameter(
            torch.empty(len(data_factory.get_anchored_reactions()) + 1, hidden_dim),
            requires_grad=True,
        )
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def _get_embeddings(self) -> Tensor:
        return self.weights


@gin.configurable()
class FragmentFingerprintEmbedding(ActionEmbeddingBase):
    def __init__(
        self,
        data_factory: ReactionDataFactory,
        fingerprint_list: List[str],
        random_linear_compression: bool,
        hidden_dim: int = 64,
        one_hot_weight: float = 1.0,
        linear_embedding: bool = True,
    ):
        super().__init__(data_factory, hidden_dim)
        self.fingerprint_list = fingerprint_list
        self.fragments = self.data_factory.get_fragments()
        if not self.fragments:
             print("Warning: FragmentFingerprintEmbedding received an empty fragment list from data_factory.")

        self.one_hot_weight = one_hot_weight
        self.one_hot = nn.Parameter(
            torch.empty(len(self.fragments), hidden_dim), requires_grad=True
        )
        init.kaiming_uniform_(self.one_hot, a=math.sqrt(5))

        self.all_fingerprints = self._get_fingerprints()
        if random_linear_compression:
            random_projection = torch.empty(self.all_fingerprints.shape[-1], hidden_dim)
            init.kaiming_uniform_(random_projection, a=math.sqrt(5))
            self.all_fingerprints = torch.matmul(
                self.all_fingerprints, random_projection
            )  # (num_fragments, hidden_dim)
        if linear_embedding:
            self.fp_embedding = nn.Linear(self.all_fingerprints.shape[-1], hidden_dim)
        else:
            self.fp_embedding = nn.Sequential(
                nn.Linear(self.all_fingerprints.shape[-1], hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def _get_fingerprints(self) -> Tensor:
        fps_list = []
        for molecule in self.fragments:
            mol = molecule.rdkit_mol
            for fp_type in self.fingerprint_list:
                fps = []
                if fp_type == "maccs":
                    fp = MACCSkeys.GenMACCSKeys(mol)
                    array = np.zeros((0,), dtype=np.float32)
                    DataStructs.ConvertToNumpyArray(fp, array)
                    fps.append(array)
                elif fp_type == "ecfp":
                    fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=20480)
                    array = np.zeros((0,), dtype=np.float32)
                    DataStructs.ConvertToNumpyArray(fp, array)
                    fps.append(array)
            fps = np.concatenate(fps, axis=0)
            fps_list.append(fps)
        fps_numpy = np.stack(fps_list, axis=0)
        # remove zero columns
        fps_numpy = fps_numpy[:, np.any(fps_numpy, axis=0)]
        return torch.tensor(fps_numpy).float()

    def _get_embeddings(self) -> Tensor:
        fingerprints = self.fp_embedding(self.all_fingerprints)
        if self.one_hot_weight > 0:
            return fingerprints + self.one_hot_weight * self.one_hot
        return fingerprints

    def set_device(self, device: str, recursive: bool = True):
        self.all_fingerprints = self.all_fingerprints.to(device)
        super().set_device(device, recursive=recursive)


@gin.configurable()
class FragmentGNNEmbedding(ActionEmbeddingBase):
    def __init__(
        self,
        data_factory: ReactionDataFactory,
        hidden_dim: int,
        gnn_type: str,
        num_layers: int,
        linear_embedding: bool,
        one_hot_weight: float = 1.0,
    ):
        super().__init__(data_factory, hidden_dim)
        self.node_featurizer = CanonicalAtomFeaturizer()
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.smiles_to_graph = SMILESToBigraph(
            node_featurizer=self.node_featurizer,
            edge_featurizer=self.edge_featurizer,
            add_self_loop=True,
        )
        self.gnn_type = gnn_type
        if gnn_type == "gat":
            self.gnn = GAT(
                in_feats=self.node_featurizer.feat_size(),
                hidden_feats=[hidden_dim] * num_layers,
            )
        elif gnn_type == "gcn":
            self.gnn = GCN(
                in_feats=self.node_featurizer.feat_size(),
                hidden_feats=[hidden_dim] * num_layers,
            )
        if linear_embedding:
            self.final_embedding = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.final_embedding = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.one_hot_weight = one_hot_weight
        self.one_hot = nn.Parameter(
            torch.empty(len(self.fragments), hidden_dim), requires_grad=True
        )
        init.kaiming_uniform_(self.one_hot, a=math.sqrt(5))

        self.batch = self._get_graph_batch()

    def _get_graph_batch(self) -> dgl.DGLGraph:
        graphs = []
        for molecule in self.fragments:
            graph = self.smiles_to_graph(molecule.smiles)
            graphs.append(graph)
        return dgl.batch(graphs)

    def _get_embeddings(self) -> Tensor:
        node_embeddings = self.gnn(self.batch, self.batch.ndata["h"])
        graph_embeddings = torch.zeros(
            size=(len(self.fragments), self.hidden_dim),
            dtype=torch.float32,
            device=node_embeddings.device,
        )
        num_nodes = self.batch.batch_num_nodes()
        indices = torch.arange(len(num_nodes), device=num_nodes.device)
        index = torch.repeat_interleave(indices, num_nodes).long()
        graph_embeddings = torch.index_add(
            input=graph_embeddings, index=index, dim=0, source=node_embeddings
        )
        graph_embeddings = self.final_embedding(graph_embeddings)
        if self.one_hot_weight > 0:
            return graph_embeddings + self.one_hot_weight * self.one_hot
        return graph_embeddings

    def set_device(self, device: str, recursive: bool = True):
        self.batch = self.batch.to(device)
        self.super().set_device(device, recursive=recursive)
