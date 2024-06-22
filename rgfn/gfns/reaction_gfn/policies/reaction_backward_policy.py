from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Sequence, Tuple, Type

import gin
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import Parameter
from torchtyping import TensorType

from rgfn.api.env_base import TState
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    Molecule,
    Reaction,
    ReactionAction,
    ReactionActionA,
    ReactionActionSpace,
    ReactionActionSpace0,
    ReactionActionSpaceA,
    ReactionActionSpaceB,
    ReactionActionSpaceC,
    ReactionActionSpaceEarlyTerminate,
    ReactionState,
    ReactionStateC,
)
from rgfn.gfns.reaction_gfn.api.reaction_data_factory import ReactionDataFactory
from rgfn.gfns.reaction_gfn.policies.graph_transformer import (
    GraphTransformer,
    mol2graph,
    mols2batch,
)
from rgfn.gfns.reaction_gfn.policies.reaction_forward_policy import (
    ReactionForwardPolicy,
)
from rgfn.gfns.reaction_gfn.policies.utils import one_hot, to_dense_embeddings
from rgfn.shared.policies.few_phase_policy import FewPhasePolicyBase, TSharedEmbeddings
from rgfn.shared.policies.uniform_policy import TIndexedActionSpace


@dataclass(frozen=True)
class SharedEmbeddings:
    molecule_reaction_to_idx: Dict[Tuple[Molecule, Reaction], int]
    all_embeddings: TensorType[float]


@gin.configurable()
class ReactionBackwardPolicy(
    FewPhasePolicyBase[ReactionState, ReactionActionSpace, ReactionAction, SharedEmbeddings],
    nn.Module,
):
    @property
    def action_space_to_forward_fn(
        self,
    ) -> Dict[
        Type[TIndexedActionSpace],
        Callable[[List[TState], List[TIndexedActionSpace], TSharedEmbeddings], TensorType[float]],
    ]:
        return self._action_space_type_to_forward_fn

    def __init__(
        self,
        data_factory: ReactionDataFactory,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 5,
        linear_output: bool = False,
        backbone_env: ReactionForwardPolicy | None = None,
    ):
        super().__init__()
        self.reactions = data_factory.get_reactions()
        self.reaction_to_idx = {reaction: idx for idx, reaction in enumerate(self.reactions)}
        self.fragments = data_factory.get_fragments()
        self.use_backbone = backbone_env is not None
        self.gnn = (
            GraphTransformer(
                x_dim=71,
                e_dim=4,
                g_dim=len(self.reactions),
                num_layers=num_layers,
                num_heads=num_heads,
                num_emb=hidden_dim,
            )
            if not self.use_backbone
            else backbone_env.gnn
        )

        self.mlp_c = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if not linear_output
            else nn.Linear(hidden_dim, 1)
        )
        self._action_space_type_to_forward_fn = {
            ReactionActionSpace0: self._forward_deterministic,
            ReactionActionSpaceB: self._forward_deterministic,
            ReactionActionSpaceEarlyTerminate: self._forward_deterministic,
            ReactionActionSpaceA: self._forward_deterministic,
            ReactionActionSpaceC: self._forward_c,
        }

        self._device = "cpu"

    def _forward_c(
        self,
        states: List[ReactionStateC],
        action_spaces: List[ReactionActionSpaceC],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        embedding_indices_list = []
        for action_space in action_spaces:
            embedding_indices = [
                shared_embeddings.molecule_reaction_to_idx[
                    (action.input_molecule, action.input_reaction.reaction)
                ]
                for action in action_space.possible_actions
            ]
            embedding_indices_list.append(embedding_indices)
        embedding_indices_flat = [idx for indices in embedding_indices_list for idx in indices]
        embedding_indices = torch.tensor(embedding_indices_flat).long().to(self.device)
        embeddings = torch.index_select(
            shared_embeddings.all_embeddings, index=embedding_indices, dim=0
        )
        logits = self.mlp_c(embeddings).squeeze(-1)
        logits, _ = to_dense_embeddings(
            logits, [len(indices) for indices in embedding_indices_list], fill_value=float("-inf")
        )
        return torch.log_softmax(logits, dim=1).float()

    def _forward_deterministic(
        self,
        states: List[ReactionState],
        action_spaces: List[ReactionActionSpace],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        max_num_actions = max(
            action_space.get_possible_actions_indices()[0] for action_space in action_spaces
        )
        log_probs_list = []
        for action_space in action_spaces:
            log_probs = [-float("inf")] * (max_num_actions + 1)
            log_probs[action_space.get_possible_actions_indices()[0]] = 0
            log_probs_list.append(log_probs)
        return torch.tensor(log_probs_list).float().to(self.device)

    def get_shared_embeddings(
        self, states: List[ReactionState], action_spaces: List[ReactionActionSpace]
    ) -> SharedEmbeddings:
        all_molecules_reactions = set()
        for state, action_space in zip(states, action_spaces):
            if isinstance(action_space, ReactionActionSpaceC):
                for action in action_space.possible_actions:
                    all_molecules_reactions.add(
                        (action.input_molecule, action.input_reaction.reaction)
                    )

        molecule_reaction_to_idx = {
            molecule_reaction: idx for idx, molecule_reaction in enumerate(all_molecules_reactions)
        }

        molecule_reaction_graphs = [
            mol2graph(mol.rdkit_mol) for mol, _ in molecule_reaction_to_idx.keys()
        ]
        molecule_reaction_cond = [
            one_hot(self.reaction_to_idx[r], len(self.reactions))
            for _, r in molecule_reaction_to_idx.keys()
        ]

        graphs = molecule_reaction_graphs
        conds = molecule_reaction_cond

        if len(graphs) == 0:
            return SharedEmbeddings(
                molecule_reaction_to_idx=molecule_reaction_to_idx,
                all_embeddings=torch.tensor([], dtype=torch.float).to(self.device),
            )
        graph_batch = mols2batch(graphs).to(self.device)
        cond_batch = torch.tensor(conds).float().to(self.device)

        embeddings = self.gnn(graph_batch, cond_batch)
        return SharedEmbeddings(
            molecule_reaction_to_idx=molecule_reaction_to_idx, all_embeddings=embeddings
        )

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if not self.use_backbone:
            return super().parameters(recurse)
        return self.mlp_c.parameters(recurse)

    def set_device(self, device: str):
        self.to(device)
        self._device = device

    @property
    def device(self) -> str:
        return self._device

    def compute_states_log_flow(self, states: List[ReactionState]) -> TensorType[float]:
        raise NotImplementedError()

    def clear_action_embedding_cache(self) -> None:
        pass

    def clear_sampling_cache(self) -> None:
        pass