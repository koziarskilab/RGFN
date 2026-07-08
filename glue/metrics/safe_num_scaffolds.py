"""Crash-safe variant of upstream ``NumScaffoldsFound``.

Upstream ``NumScaffoldsFound.compute_metrics`` iterates over *all* terminal
states and unconditionally reads ``state.molecule.smiles``. But a trajectory can
end via the early-terminate action, whose last state is a
``ReactionStateEarlyTerminal`` — which has only ``previous_state`` and **no**
``molecule`` field (see ``rgfn/gfns/reaction_gfn/api/reaction_api.py``). When such
a state's proxy value crosses one of the metric's thresholds, upstream raises::

    AttributeError: 'ReactionStateEarlyTerminal' object has no attribute 'molecule'

Every *other* metric in ``rgfn/trainer/metrics/reaction_metrics.py`` (``QED``,
``UniqueMolecules``, ``AllMolecules``, ``TanimotoSimilarityModes``) already guards
this exact case with ``isinstance(state, ReactionStateTerminal)``. ``NumScaffoldsFound``
is the lone exception — a latent upstream bug, not a design choice. We surface it
because our lower-is-better Vina-differential proxy emits values that frequently
cross thresholds on early-terminated trajectories.

Per the repo rule (don't edit ``rgfn/``), we fix it here by subclassing and adding
the same guard the sibling metrics use, leaving behaviour otherwise identical.
Reference it from ``configs/glue/`` via ``@SafeNumScaffoldsFound()`` in
``train_metrics``.
"""

from typing import Dict

import gin
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from rgfn.api.trajectories import Trajectories
from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateTerminal
from rgfn.trainer.metrics.reaction_metrics import NumScaffoldsFound


@gin.configurable()
class SafeNumScaffoldsFound(NumScaffoldsFound):
    """``NumScaffoldsFound`` that skips non-terminal (early-terminated) states."""

    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, float]:
        reward_outputs = trajectories.get_reward_outputs()
        terminal_states = trajectories.get_last_states_flat()
        values = (
            reward_outputs.proxy
            if self.proxy_component_name is None
            else reward_outputs.proxy_components[self.proxy_component_name]
        )
        for state, proxy_value in zip(terminal_states, values):
            # The only change vs. upstream: skip states without a molecule
            # (ReactionStateEarlyTerminal), matching every sibling metric.
            if not isinstance(state, ReactionStateTerminal):
                continue
            for threshold in self.proxy_value_threshold_list:
                if (self.proxy_higher_better and proxy_value.item() > threshold) or (
                    not self.proxy_higher_better and proxy_value.item() < threshold
                ):
                    self.threshold_to_set[threshold].add(
                        MurckoScaffoldSmiles(state.molecule.smiles)
                    )

        return {
            f"num_scaffolds_{threshold}": len(self.threshold_to_set[threshold])
            for threshold in self.proxy_value_threshold_list
        }
