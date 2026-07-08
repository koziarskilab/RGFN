"""A cheap CPU oracle so the active-learning loop is runnable without gnina/GPU.

This is a *test fixture*, not science. It lets us validate the whole loop —
seed -> fit proxy -> train RGFN -> sample batch -> score -> refit -> repeat — end
to end on a laptop, exactly the way ``glue/proxies/example_glue_proxy.py`` lets
the proxy wiring be exercised without docking. The real signal comes from
``Docking6TD3Oracle``; swap that in via gin for a Balam run.

It returns a deterministic, smooth function of cheap RDKit descriptors so the
MPNN proxy has a learnable target (QED blended with a normalised molecular
weight). Determinism keeps loop runs reproducible.
"""

from typing import List

import gin
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.QED import qed

from glue.oracles.base import GlueOracle


@gin.configurable()
class MockGlueOracle(GlueOracle):
    """Deterministic cheap stand-in for an expensive glue oracle (local testing)."""

    name = "mock_glue_oracle"
    higher_is_better = True

    def __init__(self, mw_center: float = 400.0, mw_scale: float = 150.0):
        """
        Args:
            mw_center / mw_scale: a soft molecular-weight preference band, so the
                synthetic target is not a pure function of QED (gives the proxy a
                slightly richer surface to fit). Values are arbitrary test knobs.
        """
        self.mw_center = mw_center
        self.mw_scale = mw_scale

    def score(self, smiles: List[str]) -> List[float]:
        scores: List[float] = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s) if s is not None else None
            if mol is None:
                scores.append(float("nan"))
                continue
            mw_term = float(
                np.exp(-(((Descriptors.MolWt(mol) - self.mw_center) / self.mw_scale) ** 2))
            )
            scores.append(float(qed(mol)) * mw_term)
        return scores
