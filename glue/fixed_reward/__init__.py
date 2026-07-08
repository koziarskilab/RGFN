"""Fixed-reward (single-shot) pipeline — the RGFN-paper training mode.

The counterpart to ``glue.active_learning``: train the GFN **once** against a **fixed
reward generator** (a frozen surrogate like the sEH proxy, or docking called directly),
then sample + emit candidates. No active-learning loop, no oracle, no proxy refit.
See ``glue.fixed_reward.pipeline`` for the full rationale.
"""

from glue.fixed_reward.pipeline import FixedRewardPipeline

__all__ = ["FixedRewardPipeline"]
