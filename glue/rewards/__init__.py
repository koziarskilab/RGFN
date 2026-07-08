"""Rewards — turning oracle scores into the reward signal the GFN trains on.

Upstream computes rewards from a proxy value via configurable reward transforms
(see `configs/rewards/*.gin` and `rgfn/api/reward.py`). Put our new reward
shaping here, for example:

    - neosubstrate-differential reward (Tier2 - Tier1 on the same pose)
    - multi-objective / composed rewards specific to glue generation
    - reward transforms that penalize warhead-only binders

Keep these as small, gin-configurable classes/functions and import them below so
`glue.registry` registers them.
"""

# from glue.rewards.neosubstrate_differential import NeosubstrateDifferentialReward  # noqa: F401
