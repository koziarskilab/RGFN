from dataclasses import InitVar, dataclass, field
from typing import Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmarts


@dataclass(frozen=True)
class Molecule:
    mol_or_smiles: InitVar[str | Chem.Mol] = field(init=True, repr=False, compare=False, hash=False)
    smiles: str = field(init=False, hash=True, compare=True)
    rdkit_mol: Chem.Mol = field(init=False, repr=False, compare=False, hash=False)
    idx: int | None = field(repr=False, compare=False, default=None, hash=False)
    valid: bool = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self, mol_or_smiles: str | Chem.Mol):
        rdkit_mol = (
            Chem.MolFromSmiles(mol_or_smiles) if isinstance(mol_or_smiles, str) else mol_or_smiles
        )
        if rdkit_mol is None:
            canonical_smiles = mol_or_smiles
            valid = False
        else:
            if Chem.SanitizeMol(rdkit_mol, catchErrors=True) == 0:
                rdkit_mol = Chem.RemoveHs(rdkit_mol)
                valid = True
            else:
                valid = False
            canonical_smiles = Chem.MolToSmiles(rdkit_mol)

        object.__setattr__(self, "rdkit_mol", rdkit_mol)
        object.__setattr__(self, "smiles", canonical_smiles)
        object.__setattr__(self, "valid", valid)

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.idx is not None:
            return f"F({self.idx})"
        else:
            return self.smiles


@dataclass(frozen=True)
class Pattern:
    pattern: str = field(hash=True, compare=True)
    rdkit_pattern: Chem.Mol = field(init=False, hash=False, compare=False)

    def __post_init__(self):
        rdkit_pattern = MolFromSmarts(self.pattern)
        if rdkit_pattern is None:
            raise ValueError(f"Invalid pattern SMILES: {self.pattern}")
        object.__setattr__(self, "rdkit_pattern", rdkit_pattern)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"P({self.pattern})"


@dataclass(frozen=True)
class Reaction:
    reaction: str = field(hash=True, compare=True)
    rdkit_rxn: AllChem.ChemicalReaction = field(init=False, hash=False, compare=False)
    left_side_patterns: Tuple[Pattern, ...] = field(init=False, hash=False, compare=False)
    idx: int = field(hash=False, compare=False)

    def __post_init__(self):
        left, right = self.reaction.split(">>")
        left, right = left.strip(), right.strip()
        reaction = f"{left} >> {right}"
        rxn = AllChem.ReactionFromSmarts(reaction)
        if rxn is None:
            raise ValueError(f"Invalid reaction SMILES: {self.reaction}")
        left_side_rdkit_patterns = tuple(Pattern(p) for p in left.split("."))
        object.__setattr__(self, "reaction", reaction)
        object.__setattr__(self, "rdkit_rxn", rxn)
        object.__setattr__(self, "left_side_patterns", left_side_rdkit_patterns)

    def reversed(self) -> "Reaction":
        left, right = self.reaction.split(">>")
        disconnection = f"{right} >> {left}"
        return Reaction(disconnection, self.idx)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"R({self.reaction})"


@dataclass(frozen=True)
class AnchoredReaction(Reaction):
    """
    A reaction with an anchored pattern from the left side of the reaction. The anchored pattern is moved
    to the first position of the left side of the reaction.

    Attributes:
        anchor_pattern_idx: The index of the pattern to be anchored and moved to the first position.
        anchored_pattern: The pattern that is anchored.
        fragment_patterns: The rest of the left side patterns in unchanged order.
    """

    anchor_pattern_idx: int = field(hash=False, compare=False)
    anchored_pattern: Pattern = field(init=False, hash=False, compare=False)
    fragment_patterns: Tuple[Pattern, ...] = field(init=False, hash=False, compare=False)

    def __post_init__(self):
        left, right = self.reaction.split(">>")
        left, right = left.strip(), right.strip()
        left = left.split(".")
        left = (
            [left[self.anchor_pattern_idx]]
            + left[: self.anchor_pattern_idx]
            + left[self.anchor_pattern_idx + 1 :]
        )
        left = ".".join(left)
        reaction = f"{left} >> {right}"
        object.__setattr__(self, "reaction", reaction)
        super().__post_init__()
        object.__setattr__(self, "anchored_pattern", self.left_side_patterns[0])
        object.__setattr__(self, "fragment_patterns", self.left_side_patterns[1:])

    def reversed(self) -> "AnchoredReaction":
        left, right = self.reaction.split(">>")
        disconnection = f"{right} >> {left}"
        return AnchoredReaction(disconnection, idx=self.idx, anchor_pattern_idx=0)
