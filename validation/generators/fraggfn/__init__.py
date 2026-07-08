"""FragGFN — fragment-based GFlowNet baseline (non-synthesizable), the RGFN paper's
foil for synthesizability-by-construction.

Thin adapter over Recursion's ``gflownet`` (``FragMolBuildingEnvContext``), trained
through the SAME active-learning loop, proxy, oracle, seed and budget as the RGFN
entrant so the comparison is apples-to-apples. Runs in its own ``fraggfn`` conda
env (installed by ``external/setup_fraggfn.sh``); it reaches the shared docking
oracle across the env boundary via ``scripts/score_batch.py`` (the ``rgfn`` env).

Boundary rule: this package may import ``glue``/``rgfn`` in principle, but by
design does NOT — its env can't load them. All ``glue`` usage (oracle + standard
candidate-dataset logging) happens in the bridge subprocess. See ``README.md``.
"""
