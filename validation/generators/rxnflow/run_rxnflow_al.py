#!/usr/bin/env python
"""Entry point for the RxnFlow active-learning loop (the synthesizable baseline).

Runs in the ``rxnflow`` conda env. Mirrors ``scripts/active_learning.py`` (the RGFN
loop entry point) and ``run_fraggfn_al.py`` (the FragGFN baseline), but drives
RxnFlow's synthesis GFlowNet against our learned proxy ``M``, labelling each round's
query batch with the SHARED docking oracle via the bridge (``scripts/score_batch.py``
under the ``rgfn`` env) and passing each molecule's synthesis route through to the
standard candidate dataset.

    conda run -n rxnflow python validation/generators/rxnflow/run_rxnflow_al.py \
        --cfg validation/configs/rxnflow_6td3.yaml \
        --seed-csv experiments/active_learning/6td3/seed_6td3.csv \
        --root-dir $SCRATCH/rgfn_runs/experiments

Must be launched from the repo root (so the bridge's ``scripts/...`` path and the
seed/config relative paths resolve). See validation/generators/rxnflow/README.md.

API note (flagged for Balam validation, ``docs/REFACTOR_LOG.md``): the RxnFlow Config
schema (``rxnflow.config``) and the env-dir / action-subsampling field names below are
taken from the RxnFlow docs/examples; confirm against the cloned repo. Fields we are
unsure of are set defensively (``_set_if`` helper) so a schema mismatch surfaces as a
clear message rather than an AttributeError mid-run.
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Make the repo root importable when launched as a script (sys.path[0] is this file's
# dir, not the repo root), so `validation.generators.rxnflow.*` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.generators.rxnflow.al_loop import LabelStore, RxnFlowActiveLearningLoop
from validation.generators.rxnflow.proxy import AtomMPNNProxy
from validation.generators.rxnflow.task import (
    RxnFlowGlueTrainer,
    build_constant_temperature,
)


def _timestamp() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _set_if(cfg, dotted: str, value) -> None:
    """Assign ``cfg.<dotted> = value`` only if every parent attribute exists, so an
    unknown RxnFlow Config field is reported (not silently ignored, not a crash)."""
    if value is None:
        return
    obj = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not hasattr(obj, p):
            print(f"[RXN-AL] WARNING config has no '{dotted}' ({p} missing) — skipping", flush=True)
            return
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        print(f"[RXN-AL] WARNING config has no '{dotted}' — skipping", flush=True)
        return
    setattr(obj, parts[-1], value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cfg", required=True, help="YAML run config (validation/configs/rxnflow_*.yaml)"
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
    rxn_c = cfg.get("rxnflow", {})
    oracle_c = cfg.get("oracle", {})

    seed = args.seed if args.seed is not None else int(loop_c.get("seed", 42))
    seed_csv = args.seed_csv or loop_c.get("seed_csv")
    if not seed_csv or not Path(seed_csv).exists():
        raise SystemExit(
            f"seed D_0 CSV not found: {seed_csv!r} (set --seed-csv or cfg.loop.seed_csv)"
        )
    higher_is_better = bool(oracle_c.get("higher_is_better", False))

    device = args.device or rxn_c.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- run dir (timestamped, mirrors scripts/active_learning.py routing). ------
    root = Path(args.root_dir or cfg.get("run", {}).get("root_dir", "experiments"))
    run_name = cfg.get("run", {}).get("name", "active_learning/rxnflow_6td3")
    run_dir = root / run_name / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "run_config.yaml")
    print(f"[RXN-AL] run_dir={run_dir} device={device} seed={seed}", flush=True)

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

    # --- RxnFlow Config (init_empty + only the fields we set; rest = defaults). ----
    # RxnFlow ships its own Config (extends the bundled gflownet Config) with the
    # synthesis env settings (building blocks + reaction templates via env_dir, and
    # action-space subsampling). Confirm exact field paths on Balam.
    from rxnflow.config import Config, init_empty

    gcfg = init_empty(Config())
    _set_if(gcfg, "log_dir", str(run_dir / "train"))
    _set_if(gcfg, "device", device)
    _set_if(gcfg, "seed", seed)
    _set_if(gcfg, "overwrite_existing_exp", True)
    _set_if(gcfg, "print_every", int(rxn_c.get("print_every", 50)))
    # informational; the AL loop drives the actual per-round step count.
    _set_if(gcfg, "num_training_steps", int(loop_c.get("n_train_steps", 300)))
    _set_if(gcfg, "opt.learning_rate", float(rxn_c.get("learning_rate", 1e-4)))
    _set_if(gcfg, "model.num_emb", int(rxn_c.get("num_emb", 128)))
    # RxnFlow's layer count lives under model.graph_transformer (no model.num_layers).
    _set_if(gcfg, "model.graph_transformer.num_layers", int(rxn_c.get("num_layers", 4)))
    _set_if(gcfg, "algo.sampling_tau", float(rxn_c.get("sampling_tau", 0.9)))
    _set_if(gcfg, "algo.max_len", int(rxn_c.get("max_reactions", 4)))  # synthesis depth

    # RxnFlow-specific: env dir (prepared blocks + templates) + action subsampling.
    env_dir = rxn_c.get("env_dir")
    if not env_dir:
        raise SystemExit(
            "rxnflow.env_dir is required (prepared building blocks + reaction templates). "
            "Run external/setup_rxnflow.sh first; see validation/generators/rxnflow/README.md."
        )
    _set_if(gcfg, "env_dir", str(env_dir))
    _set_if(gcfg, "algo.action_subsampling.sampling_ratio", rxn_c.get("action_sampling_ratio"))
    build_constant_temperature(gcfg, float(loop_c.get("beta", 8)))  # fixed β, matches RGFN

    trainer = RxnFlowGlueTrainer(gcfg, proxy=proxy)

    # --- dataset D_0. ------------------------------------------------------------
    dataset = LabelStore(lower_is_better=not higher_is_better)
    n_seed = dataset.load_seed(
        seed_csv,
        smiles_col=loop_c.get("smiles_column", "smiles"),
        label_col=loop_c.get("label_column", "label"),
    )
    print(f"[RXN-AL] seeded D_0 with {n_seed} labelled molecules from {seed_csv}", flush=True)

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

    loop = RxnFlowActiveLearningLoop(
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
    try:
        trainer.terminate()
    except Exception:
        pass


if __name__ == "__main__":
    main()
