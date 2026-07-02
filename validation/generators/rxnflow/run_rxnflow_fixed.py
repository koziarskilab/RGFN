#!/usr/bin/env python
"""Entry point for the RxnFlow **fixed-reward** (single-shot) run on sEH.

The fixed-reward counterpart of ``run_rxnflow_al.py``: train RxnFlow's synthesis GFN
**once** against the frozen pretrained sEH proxy (:class:`SEHFrozenReward`) — no oracle,
no proxy refit — then sample a batch WITH synthesis routes and emit a standard candidate
dataset (``has_route=1``). RxnFlow's entry in the matched four-way sEH comparison (the
synthesizable peer to RGFN).

    conda run -n rxnflow python validation/generators/rxnflow/run_rxnflow_fixed.py \
        --cfg validation/configs/rxnflow_seh_fixed.yaml \
        --root-dir $SCRATCH/rgfn_runs/experiments

Candidate emission shells to ``scripts/ingest_candidates.py`` under the ``rgfn`` env
(this env cannot import ``glue``), passing the routes JSONL. Launch from the repo root.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.generators.rxnflow.al_loop import LabelStore, RxnFlowActiveLearningLoop
from validation.generators.rxnflow.fixed_reward import DRD2FrozenReward, SEHFrozenReward
from validation.generators.rxnflow.task import (
    RxnFlowGlueTrainer,
    build_constant_temperature,
)


def _timestamp() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _set_if(cfg, dotted: str, value) -> None:
    """Assign ``cfg.<dotted> = value`` only if every parent attribute exists (mirrors
    run_rxnflow_al.py so an unknown RxnFlow Config field is reported, not a crash)."""
    if value is None:
        return
    obj = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not hasattr(obj, p):
            print(f"[RXN-FR] WARNING config has no '{dotted}' ({p} missing) — skipping", flush=True)
            return
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        print(f"[RXN-FR] WARNING config has no '{dotted}' — skipping", flush=True)
        return
    setattr(obj, parts[-1], value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cfg", required=True, help="YAML run config (validation/configs/*_fixed.yaml)"
    )
    ap.add_argument("--seed", type=int, default=None, help="override RNG seed (else cfg.run.seed)")
    ap.add_argument("--root-dir", default=None, help="base run dir (else cfg.run.root_dir)")
    ap.add_argument("--device", default=None, help="cpu | cuda (else auto)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    run_c = cfg.get("run", {})
    fr_c = cfg.get("fixed_reward", {})
    rxn_c = cfg.get("rxnflow", {})
    reward_c = cfg.get("reward", {})

    seed = args.seed if args.seed is not None else int(run_c.get("seed", 42))
    device = args.device or rxn_c.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    beta = float(fr_c.get("beta", 8))
    n_train_steps = int(fr_c.get("n_train_steps", 5000))
    n_samples = int(fr_c.get("n_samples", 1000))
    reward_type = reward_c.get("type", "seh_proxy")
    system = fr_c.get("system", "seh")
    reward_name = fr_c.get("reward_name", "seh_proxy")
    score_units = fr_c.get("score_units", f"{reward_name} (higher is better)")

    root = Path(args.root_dir or run_c.get("root_dir", "experiments"))
    run_name = run_c.get("name", "fixed_reward/rxnflow_seh")
    run_dir = root / run_name / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "run_config.yaml")
    print(
        f"[RXN-FR] run_dir={run_dir} device={device} seed={seed} steps={n_train_steps}", flush=True
    )

    # --- fixed reward generator (frozen; no refit): sEH proxy or DRD2 oracle. ------
    if reward_type == "drd2":
        reward = DRD2FrozenReward(
            model_path=reward_c.get("model_path", "oracle/drd2_current.pkl"),
            clip=float(reward_c.get("clip", 10.0)),
        )
    else:
        reward = SEHFrozenReward(
            device=device,
            clip=float(reward_c.get("clip", 10.0)),
            batch_size=int(reward_c.get("batch_size", 128)),
        )
    print(f"[RXN-FR] reward={reward_type} system={system}", flush=True)

    # --- RxnFlow Config (mirrors run_rxnflow_al.py). -------------------------------
    from rxnflow.config import Config, init_empty

    gcfg = init_empty(Config())
    _set_if(gcfg, "log_dir", str(run_dir / "train"))
    _set_if(gcfg, "device", device)
    _set_if(gcfg, "seed", seed)
    _set_if(gcfg, "overwrite_existing_exp", True)
    _set_if(gcfg, "print_every", int(rxn_c.get("print_every", 100)))
    _set_if(gcfg, "num_training_steps", n_train_steps)
    _set_if(gcfg, "opt.learning_rate", float(rxn_c.get("learning_rate", 1e-4)))
    _set_if(gcfg, "model.num_emb", int(rxn_c.get("num_emb", 128)))
    _set_if(gcfg, "model.graph_transformer.num_layers", int(rxn_c.get("num_layers", 4)))
    _set_if(gcfg, "algo.sampling_tau", float(rxn_c.get("sampling_tau", 0.9)))
    _set_if(gcfg, "algo.max_len", int(rxn_c.get("max_reactions", 4)))

    env_dir = rxn_c.get("env_dir")
    if not env_dir:
        raise SystemExit(
            "rxnflow.env_dir is required (prepared building blocks + reaction templates). "
            "Run external/setup_rxnflow.sh first; see validation/generators/rxnflow/README.md."
        )
    _set_if(gcfg, "env_dir", str(env_dir))
    _set_if(gcfg, "algo.action_subsampling.sampling_ratio", rxn_c.get("action_sampling_ratio"))
    build_constant_temperature(gcfg, beta)  # fixed β, matches RGFN

    trainer = RxnFlowGlueTrainer(gcfg, proxy=reward)

    # Reuse the AL loop's train/sample internals (no duplication); we never call
    # loop.run(), only _train_steps() / _sample_query_batch() / _write_routes_jsonl().
    loop = RxnFlowActiveLearningLoop(
        trainer=trainer,
        proxy=reward,
        dataset=LabelStore(lower_is_better=False),
        bridge_cmd=[],
        run_dir=str(run_dir),
        n_rounds=1,
        n_train_steps=n_train_steps,
        query_batch_size=n_samples,
        sample_oversample=float(fr_c.get("sample_oversample", 4.0)),
        top_k=int(fr_c.get("top_k", 100)),
        system=system,
        seed=seed,
        oracle_higher_is_better=True,  # sEH/DRD2 rewards are both higher-is-better
    )

    out_dir = run_dir / "fixed_reward"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. train the synthesis-GFN ONCE against the frozen sEH reward.
    print(
        f"[RXN-FR] training {n_train_steps} steps against frozen sEH proxy (beta={beta})",
        flush=True,
    )
    loop._train_steps(n_train_steps)

    # 2. sample a batch of unique valid molecules WITH synthesis routes.
    batch, routes = loop._sample_query_batch()
    print(f"[RXN-FR] sampled {len(batch)} unique valid candidates", flush=True)

    # 3. score with the reward generator itself (its raw value = the score column).
    scores = reward.predict(batch)

    # 4. write pairs.csv + routes.jsonl, then emit the standard dataset via the rgfn env.
    pairs_path = out_dir / "pairs.csv"
    with open(pairs_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "score"])
        for smi, sc in zip(batch, scores):
            w.writerow([smi, sc])
    routes_path = out_dir / "routes.jsonl"
    loop._write_routes_jsonl(routes_path, batch, routes)

    ingest_cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "rgfn",
        "python",
        "scripts/ingest_candidates.py",
        "--pairs",
        str(pairs_path),
        "--routes",
        str(routes_path),
        "--out-dir",
        str(out_dir / "candidates"),
        "--generator",
        "rxnflow",
        "--reward-name",
        reward_name,
        "--system",
        system,
        "--seed",
        str(seed),
        "--score-higher-is-better",
        "--score-units",
        score_units,
        "--source",
        str(run_dir),
    ]
    print(f"[RXN-FR] ingest -> {' '.join(ingest_cmd)}", flush=True)
    subprocess.run(ingest_cmd, check=True)

    try:
        trainer.terminate()
    except Exception:
        pass
    print(f"[RXN-FR] done. candidates at {out_dir / 'candidates'}", flush=True)


if __name__ == "__main__":
    main()
