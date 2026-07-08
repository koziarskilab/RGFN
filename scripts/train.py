#!/usr/bin/env python
"""Training entry point that registers our `glue` components first.

Upstream `train.py` (repo root) only imports `rgfn`, so gin cannot see anything
defined under `glue/`. This wrapper imports `glue` (which registers our oracles /
rewards / samplers / proxies with gin) and then runs the *unmodified* upstream
`train.py` as `__main__`, forwarding all CLI args.

Use this instead of the root `train.py` whenever a config references a `glue`
component (anything in `configs/glue/`). Plain RGFN configs work through either.

    python scripts/train.py --cfg configs/glue/<your_config>.gin --seed 42
"""

import runpy
from pathlib import Path

import glue  # noqa: F401  (side effect: registers our gin components)

REPO_ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    runpy.run_path(str(REPO_ROOT / "train.py"), run_name="__main__")
