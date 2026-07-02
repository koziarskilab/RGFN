#!/usr/bin/env python
"""Entry point for the RGFN fixed-reward (single-shot) pipeline (``[koziarski2024rgfn]``).

The single-shot counterpart to ``scripts/active_learning.py``. Like it, this imports
``glue`` first so gin can resolve our components, then parses a ``configs/glue/`` config
that builds a ``FixedRewardPipeline`` and runs it. Unlike the AL driver (an outer
multi-round loop), this trains the GFN **once** against a **fixed reward generator** (a
frozen sEH proxy, or docking called directly), samples a batch, and writes candidates.

    python scripts/fixed_reward.py --cfg configs/glue/fixed_reward_seh_proxy.gin --seed 42
    python scripts/fixed_reward.py --cfg configs/glue/fixed_reward_seh_docking.gin --seed 42

The config must define a ``FixedRewardPipeline`` singleton wired to the same reward
generator instance the trainer's reward uses (both -> ``%train_proxy``). See
``configs/glue/fixed_reward_seh_proxy.gin`` for the canonical wiring.
"""

import argparse
from pathlib import Path

import gin

import glue  # noqa: F401  (side effect: registers our gin components)
from gin_config import get_time_stamp
from glue.fixed_reward import FixedRewardPipeline
from rgfn.trainer.trainer import (  # noqa: F401  (registers @Trainer for gin, as upstream train.py does)
    Trainer,
)
from rgfn.utils.helpers import seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help=(
            "Override gin's user_root_dir (where run dirs/CSVs/checkpoints are written). "
            "Required on a Balam compute node, where $HOME is read-only and run outputs "
            "must go to $SCRATCH; the default 'experiments' is a repo-relative path that "
            "only works on the writable login node."
        ),
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    config_name = Path(args.cfg).stem
    # Group fixed-reward run outputs under experiments/fixed_reward/<run>/ so they land in
    # a per-run dir mirroring the AL convention (fixed_reward_seh_proxy -> fixed_reward/
    # seh_proxy). Run outputs are timestamped subdirs and stay git-ignored.
    if config_name.startswith("fixed_reward_"):
        config_name = "fixed_reward/" + config_name[len("fixed_reward_") :]
    run_name = f"{config_name}/{get_time_stamp()}"
    # Bind the run seed onto the pipeline so it lands in the candidate-dataset manifest.
    bindings = [f'run_name="{run_name}"', f"FixedRewardPipeline.seed={args.seed}"]
    if args.root_dir is not None:
        bindings.append(f'user_root_dir="{args.root_dir}"')
    gin.parse_config_files_and_bindings([args.cfg], bindings=bindings)

    pipeline = FixedRewardPipeline()
    pipeline.trainer.logger.log_to_file(gin.operative_config_str(), "operative_config")
    pipeline.trainer.logger.log_to_file(gin.config_str(), "config")
    pipeline.run()
    pipeline.trainer.close()
