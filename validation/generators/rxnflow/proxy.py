"""Proxy ``M`` for the RxnFlow active-learning loop.

The proxy is the fast in-loop reward of the active-learning loop
(``[bengio2021gflownet]`` §4.3 / Alg. 1): the generator trains against ``M(x)^β``,
the expensive docking oracle ``O`` is queried only on per-round query batches (via
the oracle bridge), and ``M`` is refit on the accumulated labels each round.

For a fair head-to-head, ``M`` must be the **same** model for every entrant. We
therefore reuse FragGFN's :class:`AtomMPNNProxy` *verbatim* (it is generator-agnostic
— pure SMILES→reward science with no FragGFN/RGFN dependency, importing only the
Bengio-2021 atom-graph MPNN from ``gflownet.models.bengio2021flow``). Re-exporting it
here, rather than copying it, guarantees the proxy architecture/featurization/label
scaling is byte-identical to the FragGFN and RGFN entrants — the import path is the
only thing that differs.

Caveat (flagged for Balam validation, ``docs/REFACTOR_LOG.md``): RxnFlow's bundled
``gflownet`` (v0.2.0) and FragGFN's pinned Recursion ``gflownet`` must expose the
same ``bengio2021flow.MPNNet``/``mol2graph``/``mols2batch`` for ``M`` to be truly
identical across envs. ``bengio2021flow`` is the canonical, stable Bengio-2021 net,
so this is expected to hold; confirm on the cluster where both envs exist.
"""

from validation.generators.fraggfn.proxy import AtomMPNNProxy

__all__ = ["AtomMPNNProxy"]
