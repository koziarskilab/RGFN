"""glue — molecular-glue research package built on top of upstream RGFN.

This package holds **all of our additions** to the RGFN codebase: new oracles,
reward shaping, batch-selection strategies, dataset tooling, and the gin-registered
adapters that wire them into the upstream training loop.

Boundary rule (see CLAUDE.md):
    - `rgfn/`  = pristine upstream RGFN. Never edit it.
    - `glue/`  = everything we add. Extend the system from here.

Gin discovers components by class name, but only sees a class once it has been
*imported*. Importing this package imports `glue.registry`, which in turn imports
every submodule so that all `@gin.configurable` classes become available to gin.
Entry points that use our components must therefore `import glue` before parsing
gin configs (see `scripts/train.py`).
"""

from glue import registry  # noqa: F401  (side effect: registers our components)

__all__ = ["registry"]
