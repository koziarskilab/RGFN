#!/usr/bin/env python
"""Entry point for the RGFN active-learning loop (``[bengio2021gflownet]`` Alg. 1).

Like ``scripts/train.py``, this imports ``glue`` first so gin can resolve our
oracles / proxies / dataset / loop, then parses an ``configs/glue/`` config that
builds an ``ActiveLearningLoop`` and runs it. Unlike ``train.py`` (one inner GFN
run), this drives the *outer* multi-round loop.

    python scripts/active_learning.py --cfg configs/glue/active_learning_mock.gin
    python scripts/active_learning.py --cfg configs/glue/active_learning_6td3.gin --seed 42

The config must define an ``ActiveLearningLoop`` singleton wired to the same
proxy instance the trainer's reward uses (both -> ``%train_proxy``). See
``configs/glue/active_learning_mock.gin`` for the canonical wiring.
"""

import argparse
from pathlib import Path

import gin

import glue  # noqa: F401  (side effect: registers our gin components)
from gin_config import get_time_stamp
from glue.active_learning import ActiveLearningLoop
from rgfn.trainer.trainer import (  # noqa: F401  (registers @Trainer for gin, as upstream train.py does)
    Trainer,
)
from rgfn.utils.helpers import seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--acquisition",
        type=str,
        default=None,
        choices=["policy", "random"],
        help=(
            "Which policy proposes each round's query batch: 'policy' (the learned "
            "RGFN loop) or 'random' (uniform-policy baseline, the [bengio2021gflownet] "
            "Fig. 7 control). Overrides ActiveLearningLoop.acquisition from the config; "
            "if omitted, the config's value (default 'policy') is used. Lets one config "
            "drive both arms of the oracle-efficiency comparison."
        ),
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help=(
            "Override gin's user_root_dir (where run dirs/CSVs/checkpoints are "
            "written). Required on a Balam compute node, where $HOME is read-only "
            "and run outputs must go to $SCRATCH; the default 'experiments' is a "
            "repo-relative path that only works on the writable login node."
        ),
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    config_name = Path(args.cfg).stem
    # Group AL run outputs under experiments/active_learning/<run>/ so they land in
    # the same per-run dir as the committed code + seed (e.g. active_learning_seh ->
    # active_learning/seh). See docs/ARCHITECTURE.md. (Run outputs are timestamped
    # subdirs and stay git-ignored.)
    if config_name.startswith("active_learning_"):
        config_name = "active_learning/" + config_name[len("active_learning_") :]
    # Disambiguate the run dir by acquisition arm + seed *before* the timestamp:
    # the curve campaign launches several arms/seeds of the SAME config at once, and
    # get_time_stamp() has only second resolution — concurrent jobs sharing a config
    # would otherwise collide on one run dir and clobber each other's per-round
    # artifacts (observed in the pilot, Logs/023). acq defaults to the config's
    # value ("policy") when --acquisition is not passed; the oracle_calls.csv
    # 'acquisition' column remains the source of truth for the arm.
    acq_tag = args.acquisition or "policy"
    run_name = f"{config_name}/{acq_tag}_seed{args.seed}/{get_time_stamp()}"
    # Bind the run seed onto the loop so it lands in the suggestion-log manifest
    # provenance (the loop itself only sees what gin gives it; --seed is a CLI arg).
    bindings = [f'run_name="{run_name}"', f"ActiveLearningLoop.seed={args.seed}"]
    if args.acquisition is not None:
        bindings.append(f'ActiveLearningLoop.acquisition="{args.acquisition}"')
    if args.root_dir is not None:
        bindings.append(f'user_root_dir="{args.root_dir}"')
    gin.parse_config_files_and_bindings([args.cfg], bindings=bindings)

    loop = ActiveLearningLoop()
    loop.trainer.logger.log_to_file(gin.operative_config_str(), "operative_config")
    loop.trainer.logger.log_to_file(gin.config_str(), "config")
    loop.run()
    loop.trainer.close()
