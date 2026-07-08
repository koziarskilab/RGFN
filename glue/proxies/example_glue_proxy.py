"""Reference adapter showing how to plug a glue oracle into RGFN.

This is a *template*, not a scientific scorer — it returns QED so the wiring is
runnable and testable end-to-end without docking/GPU. Copy this file as the
starting point for a real adapter, replace `_compute_proxy_output` with a call
into your oracle in `glue.oracles`, and register the class in
`glue/proxies/__init__.py`.

Pattern notes:
    - Subclass `CachedProxyBase` (from upstream) to get free memoization of
      scores per state, which matters when the oracle is expensive (e.g. docking).
    - `@gin.configurable` makes the class addressable from gin as
      `@ExampleGlueProxy`; constructor args become gin-settable parameters.
    - Declare `is_non_negative` / `higher_is_better` so the reward transform
      handles the sign correctly.
"""

from typing import List

import gin
from rdkit.Chem.QED import qed

from rgfn.api.type_variables import TState
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase


@gin.configurable()
class ExampleGlueProxy(CachedProxyBase[ReactionState]):
    """Template proxy. Returns QED as a stand-in for a real glue oracle score."""

    def __init__(self):
        super().__init__()
        # Early-terminal (invalid) states get the worst possible score.
        self.cache = {ReactionStateEarlyTerminal(None): 0.0}

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def _compute_proxy_output(self, states: List[TState]) -> List[float]:
        # Replace this with a call into glue.oracles.<your_oracle>.
        return [qed(state.molecule.rdkit_mol) for state in states]
