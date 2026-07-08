"""Oracles — scoring functions that evaluate generated molecules.

An *oracle* is the scientific scorer (docking against a ternary complex, an MD
stability estimate, a learned glue classifier, ...). Keep oracle logic here,
free of RGFN-specific plumbing. The thin adapter that exposes an oracle to the
training loop as an RGFN proxy lives in `glue.proxies`.

Why the split:
    - oracles/  = "how do we score a molecule" (pure science, unit-testable)
    - proxies/  = "how does that score become a GFN reward" (the rgfn.ProxyBase
                  contract, caching, batching, gin registration)

Upstream's docking proxy (`rgfn/gfns/reaction_gfn/proxies/docking_proxy/`) is the
reference for what a real docking integration looks like; our new ternary /
neosubstrate-differential oracle work belongs here rather than inside `rgfn/`.

Add new oracle modules here and import them below so `glue.registry` picks them up.
"""

from glue.oracles.base import GlueOracle  # noqa: F401
from glue.oracles.docking_6td3_oracle import Docking6TD3Oracle  # noqa: F401
from glue.oracles.docking_gpu_differential_oracle import (  # noqa: F401
    Docking6TD3GpuOracle,
    GpuDifferentialDockingOracle,
)
from glue.oracles.docking_seh_oracle import (  # noqa: F401
    DockingClpPOracle,
    DockingSEHOracle,
)
from glue.oracles.mock_oracle import MockGlueOracle  # noqa: F401

__all__ = [
    "GlueOracle",
    "Docking6TD3Oracle",
    "Docking6TD3GpuOracle",
    "GpuDifferentialDockingOracle",
    "DockingSEHOracle",
    "DockingClpPOracle",
    "MockGlueOracle",
]
