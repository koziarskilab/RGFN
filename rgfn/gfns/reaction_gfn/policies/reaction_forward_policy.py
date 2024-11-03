import abc
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Type

import gin
import torch
from torch import nn
from torch.nn import init
from torchtyping import TensorType

from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.api.trajectories import Trajectories
from rgfn.api.type_variables import TState
from rgfn.gfns.reaction_gfn.api.data_structures import (
    AnchoredReaction,
    Molecule,
    Reaction,
)
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionAction,
    ReactionActionSpace,
    ReactionActionSpace0,
    ReactionActionSpace0Invalid,
    ReactionActionSpaceA,
    ReactionActionSpaceB,
    ReactionActionSpaceC,
    ReactionActionSpaceEarlyTerminate,
    ReactionState,
    ReactionState0,
    ReactionStateA,
    ReactionStateB,
    ReactionStateC,
)
from rgfn.gfns.reaction_gfn.api.reaction_data_factory import ReactionDataFactory
from rgfn.gfns.reaction_gfn.policies.action_embeddings import ActionEmbeddingBase
from rgfn.gfns.reaction_gfn.policies.graph_transformer import (
    GraphTransformer,
    mol2graph,
    mols2batch,
)
from rgfn.gfns.reaction_gfn.policies.utils import one_hot, to_dense_embeddings
from rgfn.shared.policies.few_phase_policy import FewPhasePolicyBase, TSharedEmbeddings
from rgfn.shared.policies.uniform_policy import TIndexedActionSpace


@dataclass(frozen=True)
class SharedEmbeddings:
    molecule_to_idx: Dict[Molecule, int]
    molecule_reaction_to_idx: Dict[Tuple[Molecule, Reaction], int]

    all_embeddings: TensorType[float]


class ReactantPositionalEncodingBase(abc.ABC, nn.Module, TrainingHooksMixin):
    def __init__(self, data_factory: ReactionDataFactory, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.data_factory = data_factory
        self._cache: TensorType[float] | None = None
        self.device = "cpu"

    @abc.abstractmethod
    def _get_all_embeddings(self) -> TensorType[float]:
        pass

    def get_all_embeddings(self) -> TensorType[float]:
        if self._cache is None:
            self._cache = self._get_all_embeddings()
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
    def select_embeddings(self, items: List[Tuple[AnchoredReaction, int]]) -> TensorType[float]:
        pass


class ReactantSimplePositionalEncoding(ReactantPositionalEncodingBase):
    def __init__(self, data_factory: ReactionDataFactory, hidden_dim: int):
        super().__init__(data_factory, hidden_dim)
        self.anchored_reaction = data_factory.get_anchored_reactions()
        self.reaction_idx_to_embedding_idx = []
        total_embeddings = 0
        for reaction in self.anchored_reaction:
            self.reaction_idx_to_embedding_idx.append(total_embeddings)
            total_embeddings += len(reaction.fragment_patterns)

        self.weights = nn.Parameter(torch.empty(total_embeddings, hidden_dim), requires_grad=True)
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def _get_all_embeddings(self) -> TensorType[float]:
        return self.weights

    def select_embeddings(self, items: List[Tuple[AnchoredReaction, int]]) -> TensorType[float]:
        indices = [self.reaction_idx_to_embedding_idx[reaction.idx] + i for reaction, i in items]
        return torch.index_select(
            self.weights, index=torch.tensor(indices).long().to(self.device), dim=0
        )


@gin.configurable()
class ReactionForwardPolicy(
    FewPhasePolicyBase[ReactionState, ReactionActionSpace, ReactionAction, SharedEmbeddings],
):
    def __init__(
        self,
        data_factory: ReactionDataFactory,
        action_embedding_fn: ActionEmbeddingBase,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 5,
        linear_output: bool = False,
        random_c: bool = False,
    ):
        super().__init__()
        self.anchored_reactions = data_factory.get_anchored_reactions()
        self.reaction_to_idx = {
            reaction: idx for idx, reaction in enumerate(self.anchored_reactions)
        }
        self.fragments = data_factory.get_fragments()
        self.random_c = random_c
        self.gnn = GraphTransformer(
            x_dim=71,
            e_dim=4,
            g_dim=len(self.anchored_reactions),
            num_layers=num_layers,
            num_heads=num_heads,
            num_emb=hidden_dim,
        )

        # State A -> Action A
        self.mlp_a = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(self.anchored_reactions) + 1),
            )
            if not linear_output
            else nn.Linear(hidden_dim, len(self.anchored_reactions) + 1)
        )

        # State B -> Action B
        self.mlp_b_fragments = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mlp_b = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            if not linear_output
            else nn.Linear(hidden_dim, hidden_dim)
        )
        self.b_reactant_pe = ReactantSimplePositionalEncoding(data_factory, hidden_dim)
        self.b_action_embedding_fn = action_embedding_fn
        self.b_action_embedding_cache = None

        # State C -> Action C
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
            ReactionActionSpace0: self._forward_0,
            ReactionActionSpaceA: self._forward_a,
            ReactionActionSpaceB: self._forward_b,
            ReactionActionSpaceC: self._forward_c,
            ReactionActionSpaceEarlyTerminate: self._forward_early_terminate,
            ReactionActionSpace0Invalid: self._forward_early_terminate,
        }

    @property
    def hook_objects(self) -> List["TrainingHooksMixin"]:
        return [self.b_action_embedding_fn, self.b_reactant_pe]

    @property
    def action_space_to_forward_fn(
        self,
    ) -> Dict[
        Type[TIndexedActionSpace],
        Callable[[List[TState], List[TIndexedActionSpace], TSharedEmbeddings], TensorType[float]],
    ]:
        return self._action_space_type_to_forward_fn

    def _forward_0(
        self,
        states: List[ReactionState0],
        action_spaces: List[ReactionActionSpace0],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        embedding_idx = shared_embeddings.molecule_to_idx[None]
        embedding = torch.index_select(
            shared_embeddings.all_embeddings,
            index=torch.tensor([embedding_idx]).long().to(self.device),
            dim=0,
        )
        embeddings = self.mlp_b(embedding)  # (1, hidden_dim)
        all_action_embeddings = (
            self.b_action_embedding_fn.get_embeddings()
        )  # (num_fragments, hidden_dim)
        logits = embeddings @ all_action_embeddings.T  # (1, num_fragments)
        return logits.repeat(len(states), 1)

    def _forward_a(
        self,
        states: List[ReactionStateA],
        action_spaces: List[ReactionActionSpaceA],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        embedding_indices = [shared_embeddings.molecule_to_idx[state.molecule] for state in states]
        embedding_indices = torch.tensor(embedding_indices).long().to(self.device)
        embeddings = torch.index_select(
            shared_embeddings.all_embeddings, index=embedding_indices, dim=0
        )
        logits = self.mlp_a(embeddings)
        mask = torch.tensor(
            [action_space.possible_actions_mask for action_space in action_spaces]
        ).to(self.device)
        return torch.masked_fill(logits, ~mask, float("-inf"))

    def _forward_b(
        self,
        states: List[ReactionStateB],
        action_spaces: List[ReactionActionSpaceB],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        mol_embedding_indices = [
            shared_embeddings.molecule_reaction_to_idx[(state.molecule, state.anchored_reaction)]
            for state in states
        ]
        mol_embedding_indices = torch.tensor(mol_embedding_indices).long().to(self.device)
        mol_embeddings = torch.index_select(
            shared_embeddings.all_embeddings, index=mol_embedding_indices, dim=0
        )

        actions_indices = [
            [action.idx for action in action_space.possible_actions]
            for action_space in action_spaces
        ]
        actions_indices_flat = [idx for indices in actions_indices for idx in indices]
        action_indices_flat = torch.tensor(actions_indices_flat).long().to(self.device)
        all_action_embeddings_flat = self.b_action_embedding_fn.get_embeddings()
        action_embeddings = torch.index_select(
            all_action_embeddings_flat, index=action_indices_flat, dim=0
        )
        actions_embeddings, mask = to_dense_embeddings(
            action_embeddings, [len(indices) for indices in actions_indices], fill_value=0
        )  # (batch_size, max_num_actions, hidden_dim)

        # Augmenting mol embedding with chosen fragments embeddings
        augmented_indices = []
        fragment_indices = []
        positional_encodings_indices = []
        for state_idx, state in enumerate(states):
            for positional_idx, fragment in enumerate(state.fragments):
                augmented_indices.append(state_idx)
                fragment_indices.append(fragment.idx)
                positional_encodings_indices.append((state.anchored_reaction, positional_idx))

        if len(augmented_indices) > 0:
            positional_encodings = self.b_reactant_pe.select_embeddings(
                positional_encodings_indices
            )
            fragment_indices = torch.tensor(fragment_indices).long().to(self.device)
            fragment_embeddings = torch.index_select(
                all_action_embeddings_flat, 0, fragment_indices
            )
            fragment_embeddings = fragment_embeddings + positional_encodings
            fragment_embeddings = self.mlp_b_fragments(fragment_embeddings)

            augmented_indices = torch.tensor(augmented_indices).long().to(self.device)
            mol_embeddings = torch.index_add(
                mol_embeddings, 0, augmented_indices, fragment_embeddings
            )

        mol_embeddings = self.mlp_b(mol_embeddings)  # (batch_size, hidden_dim)
        logits = torch.matmul(actions_embeddings, mol_embeddings.unsqueeze(2)).squeeze(
            2
        )  # (batch_size, max_num_actions)
        return torch.masked_fill(logits, ~mask, float("-inf"))

    def _forward_c(
        self,
        states: List[ReactionStateC],
        action_spaces: List[ReactionActionSpaceC],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        embedding_indices_list = []
        for action_space in action_spaces:
            embedding_indices = [
                shared_embeddings.molecule_to_idx[action.output_molecule]
                for action in action_space.possible_actions
            ]
            embedding_indices_list.append(embedding_indices)

        embedding_indices_flat = [idx for indices in embedding_indices_list for idx in indices]
        embedding_indices = torch.tensor(embedding_indices_flat).long().to(self.device)
        embeddings = torch.index_select(
            shared_embeddings.all_embeddings, index=embedding_indices, dim=0
        )
        logits = self.mlp_c(embeddings).squeeze(-1)
        if self.random_c:
            logits = torch.zeros_like(logits.detach())
        logits, _ = to_dense_embeddings(
            logits, [len(indices) for indices in embedding_indices_list], fill_value=float("-inf")
        )
        return logits

    def _forward_early_terminate(
        self,
        states: List[ReactionState],
        action_spaces: List[ReactionActionSpaceEarlyTerminate],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        return torch.zeros((len(states), 1), device=self.device, dtype=torch.float32)

    def get_shared_embeddings(
        self, states: List[ReactionState], action_spaces: List[ReactionActionSpace]
    ) -> SharedEmbeddings:
        all_molecules = set()
        all_molecules_reactions = set()
        for state, action_space in zip(states, action_spaces):
            if isinstance(action_space, ReactionActionSpace0):
                all_molecules.add(None)
            elif isinstance(action_space, ReactionActionSpaceA):
                all_molecules.add(state.molecule)
            elif isinstance(action_space, ReactionActionSpaceB):
                all_molecules_reactions.add((state.molecule, state.anchored_reaction))
            elif isinstance(action_space, ReactionActionSpaceC):
                for action in action_space.possible_actions:
                    all_molecules.add(action.output_molecule)

        molecule_to_idx = {molecule: idx for idx, molecule in enumerate(all_molecules)}
        molecule_and_reaction_to_idx = {
            molecule_reaction: idx + len(molecule_to_idx)
            for idx, molecule_reaction in enumerate(all_molecules_reactions)
        }

        molecule_graphs = [
            mol2graph(mol.rdkit_mol if mol else None) for mol in molecule_to_idx.keys()
        ]
        reaction_cond = [one_hot(0, len(self.anchored_reactions))] * len(molecule_to_idx)

        molecule_and_reaction_graphs = [
            mol2graph(mol.rdkit_mol) for mol, _ in molecule_and_reaction_to_idx.keys()
        ]
        molecule_and_reaction_cond = [
            one_hot(r.idx, len(self.anchored_reactions))
            for _, r in molecule_and_reaction_to_idx.keys()
        ]

        graphs = molecule_graphs + molecule_and_reaction_graphs
        conds = reaction_cond + molecule_and_reaction_cond
        if len(graphs) > 0:
            graph_batch = mols2batch(graphs).to(self.device)
            cond_batch = torch.tensor(conds).float().to(self.device)
            embeddings = self.gnn(graph_batch, cond_batch)
        else:
            embeddings = None
        return SharedEmbeddings(
            molecule_to_idx=molecule_to_idx,
            molecule_reaction_to_idx=molecule_and_reaction_to_idx,
            all_embeddings=embeddings,
        )
