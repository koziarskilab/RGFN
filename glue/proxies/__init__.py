"""Proxies — adapters that expose our oracles to the RGFN training loop.

A "proxy" is RGFN's abstraction for the reward scorer. The training loop calls
`ProxyBase.compute_proxy_output(states) -> ProxyOutput` (see
`rgfn/api/proxy_base.py`). This subpackage holds thin adapters that wrap an oracle
from `glue.oracles` in that contract and register it with gin by class name.

To add a new proxy:
    1. Implement the science in `glue/oracles/`.
    2. Write a `@gin.configurable` adapter here (see `example_glue_proxy.py`).
    3. Import it below so `glue.registry` registers it.
    4. Reference it from a gin config in `configs/glue/` as `@YourProxyClass`.

Nothing here should require editing `rgfn/`.
"""

from glue.proxies.example_glue_proxy import ExampleGlueProxy  # noqa: F401
from glue.proxies.learned_proxy import LearnedGlueProxy  # noqa: F401
from glue.proxies.oracle_reward_proxy import OracleRewardProxy  # noqa: F401

__all__ = ["ExampleGlueProxy", "LearnedGlueProxy", "OracleRewardProxy"]
