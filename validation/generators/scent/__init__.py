"""SCENT entrant — cost-aware template-based GFlowNet baseline ([gainski2025scent]).

A thin adapter over SCENT (github.com/koziarskilab/SCENT, an RGFN fork), driven
through the same active-learning loop / oracle / budget as the RGFN entrant so the
comparison isolates what SCENT's cost-awareness buys. See README.md.

NOTE: the runtime modules (``proxy``, ``route``, ``al_loop``, ``run_scent_al``) use
plain *sibling* imports and run only inside the ``scent`` conda env (where SCENT's
package is named ``rgfn``). They are intentionally NOT imported here — importing
them as ``validation.generators.scent.*`` would require the repo root on sys.path,
which would shadow SCENT's installed ``rgfn``. Run via ``run_scent_al.py``.
"""
