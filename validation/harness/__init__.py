"""validation.harness — the benchmark runner + evaluation-only metrics.

Kept import-light on purpose: the synthesizability evaluator (``synthesizability.py``)
runs inside the dedicated ``aizynth`` conda env, which does **not** have the heavy
``glue``/torch/dgl stack. So do not import torch-/glue-dependent modules at package
import time, or ``python -m validation.harness.synthesizability`` breaks in that env.
Import submodules lazily where needed. See ``README.md``.
"""
