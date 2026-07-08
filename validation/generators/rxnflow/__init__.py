"""RxnFlow — reaction-template + building-block GFlowNet baseline (synthesizable),
the synthesis-aware peer to RGFN (where FragGFN is the *non*-synthesizable foil).

Thin adapter over RxnFlow (``[seo2024rxnflow]``; built on a bundled Recursion
``gflownet``), trained through the SAME active-learning loop, proxy ``M``, oracle,
seed and budget as the RGFN entrant so the comparison isolates the generator. Runs
in its own ``rxnflow`` conda env (installed by ``external/setup_rxnflow.sh``); it
reaches the shared docking oracle across the env boundary via
``scripts/score_batch.py`` (the ``rgfn`` env).

Unlike FragGFN, RxnFlow builds molecules along an explicit synthetic pathway, so the
standard candidate dataset records each molecule's route (``has_route=1`` +
``routes.jsonl``) — the headline differentiator vs. the non-synthesizable baseline.

Boundary rule: this package may import ``glue``/``rgfn`` in principle, but by design
does NOT — its env can't load them. All ``glue`` usage (oracle + standard
candidate-dataset logging) happens in the bridge subprocess. See ``README.md``.
"""
