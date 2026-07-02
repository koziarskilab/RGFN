#!/usr/bin/env python
"""Entry point for the SCENT **fixed-reward** (single-shot) run on sEH.

The fixed-reward counterpart of ``run_scent_al.py``: parse a gin config that builds
SCENT's cost-guided reaction-GFN + its frozen pretrained ``@SehMoleculeProxy`` (the fixed
reward generator) + the :class:`ScentFixedRewardRun`, then run it — train once, sample,
emit a standard candidate dataset with routes. No active-learning loop, no oracle. SCENT's
entry in the matched four-way sEH comparison.

    conda run -n scent python validation/generators/scent/run_scent_fixed.py \
        --cfg validation/configs/scent_seh_fixed.gin \
        --root-dir $SCRATCH/rgfn_runs/experiments

Same namespace hygiene as ``run_scent_al.py`` (SCENT's package is named ``rgfn``): never
put the repo root on ``sys.path``; chdir into the SCENT clone so its gin includes + SMALL
library resolve; make our paths absolute first. Candidate emission shells to
``scripts/ingest_candidates.py`` under the ``rgfn`` env (cwd = repo root).
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import gin

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

_REPO_ROOT = _HERE.parents[2]
_SCENT_ROOT = _REPO_ROOT / "external" / "scent"


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cfg", required=True, help="gin config (validation/configs/scent_*_fixed.gin)"
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument(
        "--root-dir",
        default=None,
        help="base run dir (absolute). On Balam set to $SCRATCH/rgfn_runs/experiments.",
    )
    args = ap.parse_args()

    if not _SCENT_ROOT.exists():
        raise SystemExit(
            f"SCENT clone not found at {_SCENT_ROOT}. Run `bash external/setup_scent.sh` first."
        )
    sys.path.insert(1, str(_SCENT_ROOT))

    cfg_path = Path(args.cfg)
    cfg_abs = cfg_path if cfg_path.is_absolute() else (_REPO_ROOT / cfg_path)
    if not cfg_abs.exists():
        raise SystemExit(f"config not found: {cfg_abs}")
    root_dir = Path(args.root_dir).resolve() if args.root_dir else (_REPO_ROOT / "experiments")

    # Derive the run-dir name from the config stem so sEH vs DRD2 runs don't collide
    # (scent_seh_fixed.gin -> scent_seh; scent_drd2_fixed.gin -> scent_drd2).
    variant = cfg_path.stem.replace("_fixed", "")
    run_name = f"fixed_reward/{variant}/{_timestamp()}"
    run_dir = (root_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(_SCENT_ROOT)
    gin.add_config_file_search_path(str(_SCENT_ROOT))

    try:
        from rgfn.utils.helpers import seed_everything

        seed_everything(args.seed)
    except Exception as exc:  # noqa: BLE001
        print(f"[SCENT-FR] WARNING seed_everything unavailable ({exc}); continuing", flush=True)

    # Register SCENT's gin configurables + our fixed-reward run + @Trainer (SCENT's
    # trainer/__init__ does not import trainer.py, so @Trainer registers only here).
    import fixed_reward  # noqa: F401  (side effect: registers @ScentFixedRewardRun)
    from fixed_reward import ScentFixedRewardRun

    import rgfn  # noqa: F401  (side effect: registers most SCENT gin components)
    from rgfn.trainer.trainer import Trainer  # noqa: F401  (registers @Trainer)

    bindings = [
        f'user_root_dir="{root_dir}"',
        f'run_name="{run_name}"',
        f'ScentFixedRewardRun.run_dir="{run_dir}"',
        f'ScentFixedRewardRun.repo_root="{_REPO_ROOT}"',
        f"ScentFixedRewardRun.seed={args.seed}",
    ]
    gin.parse_config_files_and_bindings([str(cfg_abs)], bindings=bindings)
    print(
        f"[SCENT-FR] cfg={cfg_abs.name} run_dir={run_dir} seed={args.seed}\n"
        f"[SCENT-FR] scent_clone={_SCENT_ROOT} repo_root={_REPO_ROOT}",
        flush=True,
    )

    run = ScentFixedRewardRun()
    (run_dir / "operative_config.gin").write_text(gin.operative_config_str())
    (run_dir / "config.gin").write_text(gin.config_str())
    run.run()
    run.trainer.close()


if __name__ == "__main__":
    main()
