#!/usr/bin/env python
"""Entry point for the FragGFN **fixed-reward** (single-shot) run on sEH.

The fixed-reward counterpart of ``run_fraggfn_al.py``. Instead of the active-learning
loop (fit proxy / train / oracle-label / repeat), this trains Recursion's fragment-GFN
**once** against the frozen pretrained sEH proxy (:class:`SEHFrozenReward`) — no oracle,
no proxy refit — then samples a batch and emits it as a standard candidate dataset. This
is FragGFN's entry in the matched four-way sEH comparison (the non-synthesizable foil).

    conda run -n fraggfn python validation/generators/fraggfn/run_fraggfn_fixed.py \
        --cfg validation/configs/fraggfn_seh_fixed.yaml \
        --root-dir $SCRATCH/rgfn_runs/experiments

Because this env cannot import ``glue``, candidates are emitted by shelling out to
``scripts/ingest_candidates.py`` under the ``rgfn`` env (mirroring how the AL entrant
shells to the oracle bridge). Launch from the repo root.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import torch
from gflownet.config import Config, init_empty
from omegaconf import OmegaConf
from rdkit import Chem

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.generators.fraggfn.al_loop import FragGFNActiveLearningLoop, LabelStore
from validation.generators.fraggfn.fixed_reward import DRD2FrozenReward, SEHFrozenReward
from validation.generators.fraggfn.task import (
    FragGFNTrainer,
    build_constant_temperature,
)


def _timestamp() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _canonical(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def _sample_chunked(trainer, it, n_samples, oversample, chunk=128):
    """Sample ``n_samples`` unique valid SMILES from FragGFN in bounded-memory CHUNKS.

    FragGFN's ``_sample_query_batch`` samples all ``n*oversample`` trajectories in ONE
    ``create_training_data_from_own_samples`` call — fine for the AL batch (~800) but it
    OOMs at fixed-reward scale (n_samples=1000 -> 4000 graphs through the transformer at
    once, ~37 GB; job 69564). We sample in small chunks instead, keeping only SMILES and
    dropping each chunk's graphs, so peak memory stays at one chunk. Same per-trajectory
    logic as ``_sample_query_batch`` (graph_to_obj -> canonical -> dedup)."""
    tr = trainer
    tr.model.to(tr.device)
    tr.model.eval()
    budget = int(n_samples * oversample)
    seen, batch = set(), []
    drawn = 0
    with torch.no_grad():
        while len(batch) < n_samples and drawn < budget:
            this = min(chunk, budget - drawn)
            cond_info = tr.task.sample_conditional_information(this, it)
            trajs = tr.algo.create_training_data_from_own_samples(
                tr.model, this, cond_info["encoding"].to(tr.device), random_action_prob=0.0
            )
            drawn += this
            for t in trajs:
                if not t.get("is_valid", True):
                    continue
                try:
                    smi = Chem.MolToSmiles(tr.ctx.graph_to_obj(t["result"]))
                except Exception:
                    continue
                canon = _canonical(smi)
                if canon is None or canon in seen:
                    continue
                seen.add(canon)
                batch.append(canon)
                if len(batch) >= n_samples:
                    break
            del trajs, cond_info
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return batch


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
    gfn_c = cfg.get("gflownet", {})
    reward_c = cfg.get("reward", {})

    seed = args.seed if args.seed is not None else int(run_c.get("seed", 42))
    device = args.device or gfn_c.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    beta = float(fr_c.get("beta", 8))
    n_train_steps = int(fr_c.get("n_train_steps", 5000))
    n_samples = int(fr_c.get("n_samples", 1000))
    # Which fixed reward generator: sEH proxy (default) or DRD2 (RGFN paper's proxies).
    reward_type = reward_c.get("type", "seh_proxy")
    system = fr_c.get("system", "seh")
    reward_name = fr_c.get("reward_name", "seh_proxy")
    score_units = fr_c.get("score_units", f"{reward_name} (higher is better)")

    root = Path(args.root_dir or run_c.get("root_dir", "experiments"))
    run_name = run_c.get("name", "fixed_reward/fraggfn_seh")
    run_dir = root / run_name / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "run_config.yaml")
    print(
        f"[FGFN-FR] run_dir={run_dir} device={device} seed={seed} steps={n_train_steps}", flush=True
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
    print(f"[FGFN-FR] reward={reward_type} system={system}", flush=True)

    # --- gflownet Config (mirrors run_fraggfn_al.py). -----------------------------
    gcfg = init_empty(Config())
    gcfg.log_dir = str(run_dir / "train")
    gcfg.device = device
    gcfg.seed = seed
    gcfg.overwrite_existing_exp = True
    gcfg.print_every = int(gfn_c.get("print_every", 100))
    gcfg.num_training_steps = n_train_steps
    gcfg.algo.max_nodes = int(gfn_c.get("max_nodes", 9))
    gcfg.algo.sampling_tau = float(gfn_c.get("sampling_tau", 0.9))
    gcfg.model.num_emb = int(gfn_c.get("num_emb", 128))
    gcfg.model.num_layers = int(gfn_c.get("num_layers", 4))
    gcfg.opt.learning_rate = float(gfn_c.get("learning_rate", 1e-4))
    # Temperature schedule. Default 'constant' β (matches RGFN's single fixed β — used by
    # the matched four-way runs). 'uniform' enables gflownet's native temperature-annealed
    # exploration (β sampled per-trajectory from [low, high]) — the DRD2 exploration
    # diagnostic (Logs/019): a constant high β can't discover a sparse-reward mode.
    temp_c = cfg.get("temperature", {})
    temp_mode = temp_c.get("mode", "constant")
    temp_high = float(temp_c.get("high", 64.0))
    if temp_mode == "uniform":
        temp_low = float(temp_c.get("low", 0.0))
        gcfg.cond.temperature.sample_dist = "uniform"
        gcfg.cond.temperature.dist_params = [temp_low, temp_high]
        print(
            f"[FGFN-FR] TRAIN temperature=uniform[{temp_low},{temp_high}] (annealed exploration)",
            flush=True,
        )
    else:
        build_constant_temperature(gcfg, beta)  # fixed β, matches RGFN
        print(f"[FGFN-FR] TRAIN temperature=constant beta={beta}", flush=True)

    trainer = FragGFNTrainer(gcfg, proxy=reward)

    # Reuse the AL loop's train/sample internals (no duplication) without running the
    # AL algorithm: we never call loop.run(), only _train_steps() and
    # _sample_query_batch(). dataset/bridge_cmd are unused here.
    loop = FragGFNActiveLearningLoop(
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

    # 1. train the fragment-GFN ONCE against the frozen sEH reward.
    print(
        f"[FGFN-FR] training {n_train_steps} steps against frozen sEH proxy (beta={beta})",
        flush=True,
    )
    loop._train_steps(n_train_steps)

    # For a uniform-temperature (annealed) run, sample the final batch from the
    # EXPLOITATION policy: condition at a fixed high β via the *uniform* branch
    # (dist_params=[sb, sb]) so the thermometer encoding matches what the model saw for
    # high-β trajectories. (The 'constant' branch encodes β as zeros, which a
    # uniform-trained model reads as β≈0 = explore — the opposite of what we want.)
    if temp_mode == "uniform":
        from gflownet.utils.conditioning import TemperatureConditional

        sb = float(temp_c.get("sample_beta", temp_high))
        trainer.cfg.cond.temperature.sample_dist = "uniform"
        trainer.cfg.cond.temperature.dist_params = [sb, sb]
        trainer.task.temperature_conditional = TemperatureConditional(trainer.cfg)
        print(f"[FGFN-FR] final sampling at fixed beta={sb} (exploitation policy)", flush=True)

    # 2. sample a batch of unique valid molecules (chunked to bound GPU memory; a single
    #    n_samples*oversample sampling call OOMs at this scale — job 69564).
    batch = _sample_chunked(trainer, loop._it, n_samples, float(fr_c.get("sample_oversample", 4.0)))
    print(f"[FGFN-FR] sampled {len(batch)} unique valid candidates", flush=True)

    # 3. score them with the reward generator itself (its raw value = the score column).
    scores = reward.predict(batch)

    # 4. write pairs.csv, then emit the standard candidate dataset via the rgfn env.
    pairs_path = out_dir / "pairs.csv"
    with open(pairs_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "score"])
        for smi, sc in zip(batch, scores):
            w.writerow([smi, sc])

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
        "--out-dir",
        str(out_dir / "candidates"),
        "--generator",
        "fraggfn",
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
    print(f"[FGFN-FR] ingest -> {' '.join(ingest_cmd)}", flush=True)
    subprocess.run(ingest_cmd, check=True)

    trainer.terminate()
    print(f"[FGFN-FR] done. candidates at {out_dir / 'candidates'}", flush=True)


if __name__ == "__main__":
    main()
