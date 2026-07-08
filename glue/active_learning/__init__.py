"""Active-learning orchestration for RGFN glue generation.

This subpackage holds the *outer* multi-round loop from ``[bengio2021gflownet]``
Alg. 1 (transcribed in ``docs/RESEARCH_CONTEXT.md``): fit the proxy ``M`` on the
accumulated oracle labels, train RGFN against ``M(x)^beta`` (one full inner
``Trainer.train()`` run), sample a query batch from the trained policy, score it
with the expensive oracle ``O``, accumulate the labels, and repeat for ``N``
rounds. The inner GFN training is RGFN's own ``Trainer`` — untouched; the loop
only sequences the rounds and moves data between the proxy, policy and oracle.

It is a new subpackage rather than a sampler/reward because it sits *above* the
trainer and owns the round structure. Everything it uses (proxy, oracle, dataset)
plugs in through the existing ``glue`` interfaces, so it stays oracle-agnostic.
"""

from glue.active_learning.acquisition_trace import AcquisitionTrace  # noqa: F401
from glue.active_learning.loop import ActiveLearningLoop  # noqa: F401

__all__ = ["ActiveLearningLoop", "AcquisitionTrace"]
