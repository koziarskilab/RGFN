#!/usr/bin/env python
"""Inference entry point that registers our `glue` components first.

Same rationale as `scripts/train.py`: imports `glue` so gin can resolve any
`glue` components referenced by the config, then runs the root `infer.py` as
`__main__`, forwarding all CLI args.

    python scripts/infer.py --cfg configs/glue/<your_config>.gin \
        --checkpoint_path <ckpt> --n_molecules 1000 --output samples.json
"""

import runpy
from pathlib import Path

import glue  # noqa: F401  (side effect: registers our gin components)

REPO_ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    runpy.run_path(str(REPO_ROOT / "infer.py"), run_name="__main__")
