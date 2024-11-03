from copy import copy
from pathlib import Path
from typing import Dict, List, Tuple

import gin
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles

from rgfn.gfns.reaction_gfn.api.data_structures import AnchoredReaction, Molecule
from rgfn.gfns.reaction_gfn.api.reaction_api import Reaction


@gin.configurable()
class ReactionDataFactory:
    def __init__(
        self,
        reaction_path: str | Path,
        fragment_path: str | Path | None = None,
        docking: bool = False,
    ):
        if docking:
            sheet_reactions = "Reactions_Docking"
            sheet_fragments = "Fragments_Docking"
        else:
            sheet_reactions = "Reactions_NoDocking"
            sheet_fragments = "Fragments_NoDocking"

        reactions = list(pd.read_excel(reaction_path, sheet_name=sheet_reactions)["Reaction"])
        reactions = [r for r in reactions if isinstance(r, str)]

        self.reactions = [Reaction(r, idx) for idx, r in enumerate(reactions)]
        self.disconnections = [reaction.reversed() for reaction in self.reactions]

        self.anchored_reactions = []
        self.reaction_anchor_map: Dict[Tuple[Reaction, int], AnchoredReaction] = {}
        for reaction in self.reactions:
            for i in range(len(reaction.left_side_patterns)):
                anchored_reaction = AnchoredReaction(
                    reaction=reaction.reaction,
                    idx=len(self.anchored_reactions),
                    anchor_pattern_idx=i,
                )
                self.reaction_anchor_map[(reaction, i)] = anchored_reaction
                self.anchored_reactions.append(anchored_reaction)

        self.anchored_disconnections = [reaction.reversed() for reaction in self.anchored_reactions]

        # Load Fragments
        if fragment_path is None:
            fragments_list = pd.read_excel(reaction_path, sheet_name=sheet_fragments)[
                "Fragment"
            ].tolist()
        else:
            fragments_list = pd.read_csv(fragment_path)["SMILES"].tolist()
        fragments_list = list(set(MolToSmiles(MolFromSmiles(x)) for x in fragments_list))
        self.fragments = [Molecule(f, idx=idx) for idx, f in enumerate(fragments_list)]

        print(
            f"Using {len(self.fragments)} fragments, {len(self.reactions)} reactions, and {len(self.anchored_reactions)} anchored reactions"
        )

    def get_reactions(self) -> List[Reaction]:
        return copy(self.reactions)

    def get_disconnections(self) -> List[Reaction]:
        return copy(self.disconnections)

    def get_anchored_reactions(self) -> List[AnchoredReaction]:
        return copy(self.anchored_reactions)

    def get_reaction_anchor_map(self) -> Dict[Tuple[Reaction, int], AnchoredReaction]:
        return copy(self.reaction_anchor_map)

    def get_anchored_disconnections(self) -> List[AnchoredReaction]:
        return copy(self.anchored_disconnections)

    def get_fragments(self) -> List[Molecule]:
        return copy(self.fragments)
