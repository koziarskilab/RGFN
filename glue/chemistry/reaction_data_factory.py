"""``GlueReactionDataFactory`` — load a ``ChemLibrary`` into RGFN's action space.

Upstream's ``rgfn...ReactionDataFactory`` reads ``data/chemistry.xlsx`` and hands the
reaction/fragment lists to ``ReactionEnv``. It is ``@gin.configurable`` and the only
place chemistry enters RGFN, but its source is hardcoded to the workbook's sheets.
This subclass reads a canonical ``ChemLibrary`` directory instead — building the
*same* attributes with the *same* upstream classes (``Reaction``, ``Molecule``,
``AnchoredReaction``), so the action-space semantics are byte-identical to today;
only the *source* of the lists changes. ``rgfn/`` stays pristine (`CLAUDE.md`).

Select it from a ``configs/glue/`` overlay (``docs/CHEM_LIBRARY_FORMAT.md`` §5):

    data_factory/gin.singleton.constructor = @GlueReactionDataFactory
    GlueReactionDataFactory.library_dir = 'data/libraries/glue_standard_v1'
"""

from __future__ import annotations

from typing import Dict, Tuple

import gin
from rdkit.Chem import MolFromSmiles, MolToSmiles

from glue.chemistry.library import ChemLibrary
from rgfn.gfns.reaction_gfn.api.data_structures import AnchoredReaction, Molecule
from rgfn.gfns.reaction_gfn.api.reaction_api import Reaction
from rgfn.gfns.reaction_gfn.api.reaction_data_factory import ReactionDataFactory


@gin.configurable()
class GlueReactionDataFactory(ReactionDataFactory):
    """A ``ReactionDataFactory`` sourced from a canonical ``ChemLibrary`` directory.

    Deliberately does **not** call ``super().__init__`` (that reads the xlsx); it
    reproduces upstream's exact construction (reaction_data_factory.py lines 28-56)
    over the library's reaction SMARTS + fragment SMILES.

    Args:
        library_dir: path to ``data/libraries/<name>/`` (canonical format).
        docking: advisory only (a canonical library is already system-specific);
            recorded on the instance for parity with the upstream signature.
    """

    def __init__(self, library_dir: str, docking: bool = True):
        library = ChemLibrary.from_canonical_dir(library_dir)
        self.library_dir = library_dir
        self.docking = docking

        # Reactions — mirror upstream ReactionDataFactory.__init__.
        reactions = [r.smarts for r in library.reactions]
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
        self.anchored_disconnections = [r.reversed() for r in self.anchored_reactions]

        # Fragments — canonicalize + dedup exactly as upstream does.
        fragments_list = list(set(MolToSmiles(MolFromSmiles(f.smiles)) for f in library.fragments))
        self.fragments = [Molecule(f, idx=idx) for idx, f in enumerate(fragments_list)]

        print(
            f"[GlueReactionDataFactory] {library_dir}: "
            f"{len(self.fragments)} fragments, {len(self.reactions)} reactions, "
            f"{len(self.anchored_reactions)} anchored reactions"
        )
