#!/usr/bin/env python
"""Entry point for the SCENT active-learning loop (our cost-aware baseline).

Runs in the ``scent`` conda env. Mirrors ``scripts/active_learning.py`` (the RGFN
loop entry point): it parses a gin config that builds SCENT's cost-guided
reaction-GFlowNet + our learned proxy ``M`` + the outer :class:`ScentActiveLearningLoop`,
then runs it. Each round's query batch is labelled by the SHARED docking oracle via
the bridge (``scripts/score_batch.py`` under the ``rgfn`` env).

    conda run -n scent python validation/generators/scent/run_scent_al.py \
        --cfg validation/configs/scent_6td3.gin \
        --seed-csv experiments/active_learning/6td3/seed_6td3.csv \
        --root-dir $SCRATCH/rgfn_runs/experiments

Namespace hygiene (the reason this script is structured oddly)
-------------------------------------------------------------
SCENT's python package is *named* ``rgfn`` (it is an RGFN fork). In the ``scent``
env it is ``pip install -e .``, so ``import rgfn`` must resolve to SCENT's
**installed** package. We therefore:

  * **never** put the RGFN-Fork repo root on ``sys.path`` (that would shadow SCENT's
    ``rgfn`` with our repo-local ``rgfn/``). The adapter modules are imported as
    plain siblings (``import proxy`` / ``import al_loop``) off this file's directory,
    which is ``sys.path[0]`` — not via ``validation.generators.scent.*``;
  * ``chdir`` into the SCENT clone (``external/scent``) so SCENT's own relative paths
    resolve exactly as its ``train.py`` expects: gin ``include 'configs/…'``,
    ``import gin_config``, and the SMALL library's ``data/small/…`` cost/yield files.
    All of OUR paths (the config, the seed CSV, the run dir, the repo root the oracle
    bridge subprocess runs in) are made absolute up front, before the chdir.
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import gin

# Absolutise this file's directory and put it FIRST on sys.path (so `import proxy`
# / `import al_loop` work after we chdir away). Do NOT add the repo root — see the
# module docstring (it would shadow SCENT's installed `rgfn`).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# RGFN-Fork repo root = .../validation/generators/scent -> up 3. The SCENT clone is
# external/scent under it (created by external/setup_scent.sh).
_REPO_ROOT = _HERE.parents[2]
_SCENT_ROOT = _REPO_ROOT / "external" / "scent"


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", required=True, help="gin AL config (validation/configs/scent_*.gin)")
    ap.add_argument(
        "--seed-csv",
        default="experiments/active_learning/6td3/seed_6td3.csv",
        help="seed D_0 CSV (repo-relative or absolute)",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument(
        "--root-dir",
        default=None,
        help="base run dir (absolute). On Balam set to $SCRATCH/rgfn_runs/experiments "
        "($HOME is read-only on compute nodes). Defaults to <repo>/experiments.",
    )
    args = ap.parse_args()

    if not _SCENT_ROOT.exists():
        raise SystemExit(
            f"SCENT clone not found at {_SCENT_ROOT}. Run `bash external/setup_scent.sh` first."
        )

    # Put the SCENT clone root on sys.path: SCENT's `gin_config` helper package is NOT
    # pip-installed (poetry packages only `rgfn`), so `import gin_config` — done inside
    # SCENT's rgfn_base.gin — relies on the clone root being importable. SCENT's own
    # train.py gets this for free by running from the clone root; we add it explicitly.
    # This resolves `rgfn` to SCENT's source too (same as its editable install) — and the
    # repo root is still NOT on sys.path, so our repo-local `rgfn/` can't shadow it.
    sys.path.insert(1, str(_SCENT_ROOT))

    # --- Make every OUR-side path absolute BEFORE chdir into the SCENT clone. -----
    cfg_path = Path(args.cfg)
    cfg_abs = cfg_path if cfg_path.is_absolute() else (_REPO_ROOT / cfg_path)
    if not cfg_abs.exists():
        raise SystemExit(f"config not found: {cfg_abs}")
    seed_csv = Path(args.seed_csv)
    seed_abs = seed_csv if seed_csv.is_absolute() else (_REPO_ROOT / seed_csv)
    if not seed_abs.exists():
        raise SystemExit(f"seed D_0 CSV not found: {seed_abs}")
    root_dir = Path(args.root_dir).resolve() if args.root_dir else (_REPO_ROOT / "experiments")

    run_name = f"active_learning/scent_6td3/{_timestamp()}"
    run_dir = (root_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Enter the SCENT clone so its relative config/data paths resolve. ---------
    os.chdir(_SCENT_ROOT)
    # Resolve gin includes ('configs/…') against the SCENT clone regardless of CWD.
    gin.add_config_file_search_path(str(_SCENT_ROOT))

    # Seed (SCENT ships rgfn.utils.helpers.seed_everything, like our fork).
    try:
        from rgfn.utils.helpers import seed_everything

        seed_everything(args.seed)
    except Exception as exc:  # noqa: BLE001 - non-fatal; gin/seeds still set below
        print(f"[SCENT-AL] WARNING seed_everything unavailable ({exc}); continuing", flush=True)

    # Register SCENT's gin configurables. `import rgfn` runs its package __init__
    # (registers gfns / proxies / policies / samplers / metrics), but its
    # trainer/__init__ does NOT import trainer.py, so @Trainer is registered only by
    # importing it explicitly — exactly what SCENT's own train.py does. Without this,
    # gin parse fails with "No configurable matching 'Trainer'".
    # Import the adapter modules so their @gin.configurable classes register
    # (LearnedDockingProxy via proxy, ScentActiveLearningLoop in al_loop).
    import al_loop  # noqa: F401  (side effect: gin registration)
    from al_loop import ScentActiveLearningLoop

    import rgfn  # noqa: F401  (side effect: registers most SCENT gin components)
    from rgfn.trainer.trainer import Trainer  # noqa: F401  (registers @Trainer)

    bindings = [
        f'user_root_dir="{root_dir}"',
        f'run_name="{run_name}"',
        f'ScentActiveLearningLoop.run_dir="{run_dir}"',
        f'ScentActiveLearningLoop.repo_root="{_REPO_ROOT}"',
        f'ScentActiveLearningLoop.seed_csv="{seed_abs}"',
        f"ScentActiveLearningLoop.seed={args.seed}",
    ]
    gin.parse_config_files_and_bindings([str(cfg_abs)], bindings=bindings)
    print(
        f"[SCENT-AL] cfg={cfg_abs.name} run_dir={run_dir} seed={args.seed}\n"
        f"[SCENT-AL] scent_clone={_SCENT_ROOT} repo_root={_REPO_ROOT}",
        flush=True,
    )

    loop = ScentActiveLearningLoop()
    # Persist the resolved config next to the run for provenance (mirrors
    # scripts/active_learning.py logging the operative config).
    (run_dir / "operative_config.gin").write_text(gin.operative_config_str())
    (run_dir / "config.gin").write_text(gin.config_str())
    loop.run()
    loop.trainer.close()


if __name__ == "__main__":
    main()
