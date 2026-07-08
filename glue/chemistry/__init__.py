"""Chemistry — the standard, swappable fragment + reaction library.

This subpackage owns the two *inputs* every reaction-GFlowNet builds molecules from
— a building-block set and a reaction-template set — plus an optional cost
annotation (per-block price, per-reaction yield). The point is a *single standard
set* that RGFN, RxnFlow, and SCENT can all consume, so a benchmark run varies the
**generator** while holding the **chemistry** fixed. Full design + build order:
``docs/CHEM_LIBRARY_FORMAT.md``.

Split of responsibilities:
    - ``library.py``              = "what is the library" (format-agnostic, no RGFN
                                     plumbing; loaders in, exporters out, cost maps).
    - ``reaction_data_factory.py`` = "how does that library become RGFN's action
                                     space" (subclasses upstream ``ReactionDataFactory``
                                     so ``rgfn/`` stays pristine).

Cost enters generation two ways (see the doc): natively for SCENT (its own
``PathCostProxy``), retroactively for everyone else (``validation/harness/cost.py``
prices recorded routes with the same numbers + formula).

**Status:** non-functional stubs — every method raises ``NotImplementedError``.
Imported here so ``glue.registry`` registers ``GlueReactionDataFactory`` with gin.
"""

from glue.chemistry.library import ChemLibrary, FragmentSpec, ReactionSpec  # noqa: F401
from glue.chemistry.reaction_data_factory import GlueReactionDataFactory  # noqa: F401

__all__ = [
    "ChemLibrary",
    "FragmentSpec",
    "ReactionSpec",
    "GlueReactionDataFactory",
]
