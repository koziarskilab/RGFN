from dataclasses import dataclass, field
from typing import Any, List, Tuple

from rgfn.shared.policies.uniform_policy import IndexedActionSpaceBase

from .data_structures import AnchoredReaction, Molecule, Reaction


@dataclass(frozen=True, order=True)
class ReactionState0:
    def __repr__(self):
        return str(self)

    def __str__(self):
        return "S0"


@dataclass(frozen=True)
class ReactionAction0:
    fragment: Molecule
    idx: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"A0({self.idx})"


@dataclass(frozen=True)
class ReactionActionSpace0(IndexedActionSpaceBase[ReactionAction0]):
    all_actions: Tuple[ReactionAction0, ...]
    possible_actions_mask: List[bool]

    def get_action_at_idx(self, idx: int) -> ReactionAction0:
        return self.all_actions[idx]

    def get_idx_of_action(self, action: ReactionAction0) -> int:
        return action.idx

    def get_possible_actions_indices(self) -> List[int]:
        return [idx for idx, mask in enumerate(self.possible_actions_mask) if mask]

    def __repr__(self):
        return str(self)

    def __str__(self):
        possible_action_indices = self.get_possible_actions_indices()
        possible_actions = [self.all_actions[idx] for idx in possible_action_indices]
        return f"AS0({possible_actions})"


@dataclass(frozen=True, order=True)
class ReactionState0Invalid:
    """
    In our code there are some really rare cases of reactions (reactants, product, template) that cannot be reversed:
    they can be applied in one direction, but not in the other. The `ReactionEarlyTerminalState` is a hacky way of
    dealing with such cases in the forward sampling (a proper way is expensive). If we encounter a reaction that
    cannot be reversed, we terminate the generation process. The `ReactionState0Invalid` is a similar concept but
    for the backward sampling. In `ReactionEnv._is_decomposable` method we don't check if there are some
    irreversible reactions in the synthesis tree (it's expensive), so our backward trajectory could end up in a
    molecule that can only be decomposed by an irreversible reaction, which we cannot allow for the sake of the
    compatibility with forward sampling. At this point, we terminate the generation process and transition to
    `ReactionState0Invalid` state. All such invalid trajectories are then filtered from the trajectories batch.
    """

    previous_state: Any = field(hash=False, compare=False)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "S0Invalid"


@dataclass(frozen=True)
class ReactionAction0Invalid:
    def __repr__(self):
        return str(self)

    def __str__(self):
        return "A0Invalid"


@dataclass(frozen=True)
class ReactionActionSpace0Invalid(IndexedActionSpaceBase[ReactionAction0Invalid]):
    possible_action = ReactionAction0Invalid()

    def get_action_at_idx(self, idx: int) -> ReactionAction0Invalid:
        return self.possible_action

    def get_idx_of_action(self, action: ReactionAction0Invalid) -> int:
        return 0

    def get_possible_actions_indices(self) -> List[int]:
        return [0]

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "AS0Invalid"


@dataclass(frozen=True, order=True)
class ReactionStateA:
    molecule: Molecule
    num_reactions: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"SA({self.molecule}, {self.num_reactions})"


@dataclass(frozen=True)
class ReactionActionA:
    anchored_reaction: AnchoredReaction | None
    idx: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AA({self.idx})"


@dataclass(frozen=True)
class ReactionActionSpaceA(IndexedActionSpaceBase[ReactionActionA]):
    all_actions: Tuple[ReactionActionA, ...]
    possible_actions_mask: List[bool]

    def get_action_at_idx(self, idx: int) -> ReactionActionA:
        return self.all_actions[idx]

    def get_idx_of_action(self, action: ReactionActionA) -> int:
        return action.idx

    def get_possible_actions_indices(self) -> List[int]:
        return [idx for idx, mask in enumerate(self.possible_actions_mask) if mask]

    def __repr__(self):
        return str(self)

    def __str__(self):
        possible_action_indices = self.get_possible_actions_indices()
        possible_actions = [self.all_actions[idx] for idx in possible_action_indices]
        return f"ASA({possible_actions})"


@dataclass(frozen=True, order=True)
class ReactionStateB:
    molecule: Molecule
    anchored_reaction: AnchoredReaction
    fragments: Tuple[Molecule, ...]
    num_reactions: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"SB({self.molecule}, {self.anchored_reaction}, {self.num_reactions})"


@dataclass(frozen=True)
class ReactionActionB:
    fragment: Molecule
    idx: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AB({self.idx})"


@dataclass(frozen=True)
class ReactionActionSpaceB(IndexedActionSpaceBase[ReactionActionB]):
    possible_actions: Tuple[ReactionActionB, ...]

    def get_action_at_idx(self, idx: int) -> ReactionActionB:
        return self.possible_actions[idx]

    def get_idx_of_action(self, action: ReactionActionB) -> int:
        return list(self.possible_actions).index(action)

    def get_possible_actions_indices(self) -> List[int]:
        return list(range(len(self.possible_actions)))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"ASB({self.possible_actions})"


@dataclass(frozen=True, order=True)
class ReactionStateC:
    molecule: Molecule
    anchored_reaction: AnchoredReaction
    fragments: Tuple[Molecule, ...]
    num_reactions: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return (
            f"SC({self.molecule}, {self.anchored_reaction}, {self.fragments}, {self.num_reactions})"
        )


@dataclass(frozen=True)
class ReactionActionC:
    input_molecule: Molecule
    input_reaction: Reaction
    input_fragments: Tuple[Molecule, ...]
    output_molecule: Molecule

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AC({self.input_molecule}, {self.input_reaction}, {self.input_fragments}, {self.output_molecule})"


@dataclass(frozen=True)
class ReactionActionSpaceC(IndexedActionSpaceBase[ReactionActionC]):
    possible_actions: Tuple[ReactionActionC, ...]

    def get_action_at_idx(self, idx: int) -> ReactionActionC:
        return self.possible_actions[idx]

    def get_idx_of_action(self, action: ReactionActionC) -> int:
        return list(self.possible_actions).index(action)

    def get_possible_actions_indices(self) -> List[int]:
        return list(range(len(self.possible_actions)))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"ASC({self.possible_actions})"


@dataclass(frozen=True, order=True)
class ReactionStateEarlyTerminal:
    previous_state: Any = field(hash=False, compare=False)


@dataclass(frozen=True)
class ReactionActionEarlyTerminate:
    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"ETA()"


@dataclass(frozen=True)
class ReactionActionSpaceEarlyTerminate(IndexedActionSpaceBase[ReactionActionEarlyTerminate]):
    possible_action = ReactionActionEarlyTerminate()

    def get_action_at_idx(self, idx: int) -> ReactionActionEarlyTerminate:
        return self.possible_action

    def get_idx_of_action(self, action: ReactionActionEarlyTerminate) -> int:
        return 0

    def get_possible_actions_indices(self) -> List[int]:
        return [0]

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"ASETA()"


@dataclass(frozen=True)
class ReactionStateTerminal:
    molecule: Molecule = field(hash=True, compare=True)
    num_reactions: int = field(hash=False, compare=False)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"ST({self.molecule}, {self.num_reactions})"


ReactionState = (
    ReactionState0
    | ReactionStateA
    | ReactionStateB
    | ReactionStateC
    | ReactionStateTerminal
    | ReactionStateEarlyTerminal
)
ReactionAction = ReactionAction0 | ReactionActionA | ReactionActionB | ReactionActionC
ReactionActionSpace = (
    ReactionActionSpace0
    | ReactionActionSpaceA
    | ReactionActionSpaceB
    | ReactionActionSpaceC
    | ReactionActionSpaceEarlyTerminate
)
