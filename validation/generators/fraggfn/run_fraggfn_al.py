#!/usr/bin/env python
"""Entry point for the FragGFN active-learning loop (the non-synthesizable baseline).

Runs in the ``fraggfn`` conda env. Mirrors ``scripts/active_learning.py`` (the RGFN
loop entry point) but drives Recursion's fragment-based GFlowNet against our learned
proxy ``M``, labelling each round's query batch with the SHARED docking oracle via
the bridge (``scripts/score_batch.py`` under the ``rgfn`` env).

    conda run -n fraggfn python validation/generators/fraggfn/run_fraggfn_al.py \
        --cfg validation/configs/fraggfn_6td3.yaml \
        --seed-csv experiments/active_learning/6td3/seed_6td3.csv \
        --root-dir $SCRATCH/rgfn_runs/experiments

Must be launched from the repo root (so the bridge's ``scripts/...`` path and the
seed/config relative paths resolve). See validation/generators/fraggfn/README.md.
"""

import argparse
import sys
from pathlib import Path

import torch
from gflownet.config import Config, init_empty
from omegaconf import OmegaConf

# Make the repo root importable when launched as a script (sys.path[0] is this
# file's dir, not the repo root), so `validation.generators.fraggfn.*` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.generators.fraggfn.al_loop import FragGFNActiveLearningLoop, LabelStore
from validation.generators.fraggfn.proxy import AtomMPNNProxy
from validation.generators.fraggfn.task import (
    FragGFNTrainer,
    build_constant_temperature,
)


def _timestamp() -> str:
    # gflownet forbids Date.now()-free determinism concerns don't apply here; a
    # wall-clock stamp keeps run dirs unique like scripts/active_learning.py.
    import datetime

    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cfg", required=True, help="YAML run config (validation/configs/fraggfn_*.yaml)"
    )
    ap.add_argument(
        "--seed-csv", default=None, help="override seed D_0 CSV (else cfg.loop.seed_csv)"
    )
    ap.add_argument("--seed", type=int, default=None, help="override RNG seed (else cfg.loop.seed)")
    ap.add_argument(
        "--root-dir", default=None, help="base run dir (else cfg.run.root_dir or ./experiments)"
    )
    ap.add_argument("--device", default=None, help="cpu | cuda (else auto)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    loop_c = cfg.get("loop", {})
    proxy_c = cfg.get("proxy", {})
    gfn_c = cfg.get("gflownet", {})
    oracle_c = cfg.get("oracle", {})

    seed = args.seed if args.seed is not None else int(loop_c.get("seed", 42))
    seed_csv = args.seed_csv or loop_c.get("seed_csv")
    if not seed_csv or not Path(seed_csv).exists():
        raise SystemExit(
            f"seed D_0 CSV not found: {seed_csv!r} (set --seed-csv or cfg.loop.seed_csv)"
        )
    higher_is_better = bool(oracle_c.get("higher_is_better", False))

    device = args.device or gfn_c.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- run dir (timestamped, mirrors scripts/active_learning.py routing). ------
    root = Path(args.root_dir or cfg.get("run", {}).get("root_dir", "experiments"))
    run_name = cfg.get("run", {}).get("name", "active_learning/fraggfn_6td3")
    run_dir = root / run_name / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "run_config.yaml")
    print(f"[FGFN-AL] run_dir={run_dir} device={device} seed={seed}", flush=True)

    # --- proxy M (atom-graph MPNN; same architecture as RGFN's LearnedGlueProxy). -
    proxy = AtomMPNNProxy(
        higher_is_better=higher_is_better,
        dim=int(proxy_c.get("dim", 64)),
        num_conv_steps=int(proxy_c.get("num_conv_steps", 12)),
        lr=float(proxy_c.get("lr", 5e-4)),
        weight_decay=float(proxy_c.get("weight_decay", 0.0)),
        batch_size=int(proxy_c.get("batch_size", 128)),
        max_epochs=int(proxy_c.get("max_epochs", 100)),
        patience=int(proxy_c.get("patience", 10)),
        val_fraction=float(proxy_c.get("val_fraction", 0.1)),
        clip=float(proxy_c.get("clip", 10.0)),
        seed=seed,
        device=device,
    )

    # --- gflownet Config (init_empty + only the fields we set; rest = defaults). --
    gcfg = init_empty(Config())
    gcfg.log_dir = str(run_dir / "train")
    gcfg.device = device
    gcfg.seed = seed
    gcfg.overwrite_existing_exp = True
    gcfg.print_every = int(gfn_c.get("print_every", 50))
    gcfg.num_training_steps = int(
        loop_c.get("n_train_steps", 300)
    )  # informational; loop drives steps
    gcfg.algo.max_nodes = int(gfn_c.get("max_nodes", 9))
    gcfg.algo.sampling_tau = float(gfn_c.get("sampling_tau", 0.9))
    gcfg.model.num_emb = int(gfn_c.get("num_emb", 128))
    gcfg.model.num_layers = int(gfn_c.get("num_layers", 4))
    gcfg.opt.learning_rate = float(gfn_c.get("learning_rate", 1e-4))
    build_constant_temperature(gcfg, float(loop_c.get("beta", 8)))  # fixed β, matches RGFN

    trainer = FragGFNTrainer(gcfg, proxy=proxy)

    # --- dataset D_0. ------------------------------------------------------------
    dataset = LabelStore(lower_is_better=not higher_is_better)
    n_seed = dataset.load_seed(
        seed_csv,
        smiles_col=loop_c.get("smiles_column", "smiles"),
        label_col=loop_c.get("label_column", "label"),
    )
    print(f"[FGFN-AL] seeded D_0 with {n_seed} labelled molecules from {seed_csv}", flush=True)

    # --- oracle bridge command (runs scripts/score_batch.py under the rgfn env). --
    conda_exe = oracle_c.get("conda_exe", "conda")
    bridge_cmd = [
        conda_exe,
        "run",
        "--no-capture-output",
        "-n",
        oracle_c.get("env", "rgfn"),
        "python",
        oracle_c.get("script", "scripts/score_batch.py"),
        "--oracle",
        oracle_c.get("name", "docking_6td3_gpu"),
    ]
    for k, v in dict(oracle_c.get("args", {})).items():
        bridge_cmd += ["--oracle-arg", f"{k}={v}"]

    loop = FragGFNActiveLearningLoop(
        trainer=trainer,
        proxy=proxy,
        dataset=dataset,
        bridge_cmd=bridge_cmd,
        run_dir=str(run_dir),
        n_rounds=int(loop_c.get("n_rounds", 3)),
        n_train_steps=int(loop_c.get("n_train_steps", 300)),
        query_batch_size=int(loop_c.get("query_batch_size", 32)),
        sample_oversample=float(loop_c.get("sample_oversample", 4.0)),
        top_k=int(loop_c.get("top_k", 16)),
        seed_csv=seed_csv,
        system=loop_c.get("system"),
        seed=seed,
        oracle_threshold=float(loop_c.get("oracle_threshold", -2.0)),
        oracle_higher_is_better=higher_is_better,
    )
    loop.run()
    trainer.terminate()


if __name__ == "__main__":
    main()
