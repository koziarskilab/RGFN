#!/usr/bin/env python
"""Generate the seed dataset ``D_0`` for the sEH active-learning loop.

``[bengio2021gflownet]`` Alg. 1 starts from an initial labelled set ``D_0``. The
*in-distribution* source for ``D_0`` is the molecule space RGFN actually explores,
so we draw it from the RGFN reaction policy itself (random rollouts from the
freshly-initialised forward policy — no GFN training yet), then label each
molecule with the real sEH GPU-docking oracle. This is exactly the sampler +
oracle the loop uses each round (see ``glue/active_learning/loop.py``), so the
seed is byte-for-byte consistent with the per-round query batches.

Run on a GPU node (login or compute) with the QuickVina2-GPU build available
(from the repo root, so the repo-relative --cfg/--out defaults resolve):

    python experiments/active_learning/seh/make_seh_seed.py \
        --cfg configs/glue/active_learning_seh.gin --n 200

Writes ``experiments/active_learning/seh/seed_seh.csv`` (columns: smiles,label),
the path ``active_learning_seh.gin`` points ``OracleLabeledDataset.seed_csv`` at.
Molecules that fail to dock (nan) are dropped, so the file may hold fewer than
``--n`` rows; oversample with ``--n`` accordingly.
"""

import argparse
import csv
import math
import time
from pathlib import Path

import gin

import glue  # noqa: F401  (registers our gin components: oracle/proxy/dataset/loop)
from gin_config import get_time_stamp
from glue.oracles.docking_seh_oracle import DockingSEHOracle
from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateTerminal
from rgfn.trainer.trainer import Trainer  # noqa: F401  (registers @Trainer for gin)
from rgfn.utils.helpers import seed_everything


def _sample_unique(trainer, n_target: int, oversample: float) -> list:
    """Sample unique valid terminal SMILES from the (untrained) forward policy.

    Mirrors ``ActiveLearningLoop._sample_query_batch`` so the seed comes from the
    same distribution as the per-round batches.
    """
    sampler = trainer.train_forward_sampler
    n_sample = int(n_target * oversample)
    batch_size = trainer.train_batch_size
    seen, batch = set(), []
    for trajectories in sampler.get_trajectories_iterator(n_sample, batch_size):
        for state in trajectories.get_last_states_flat():
            if not isinstance(state, ReactionStateTerminal):
                continue
            smi = state.molecule.smiles
            if smi in seen:
                continue
            seen.add(smi)
            batch.append(smi)
            if len(batch) >= n_target:
                return batch
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/glue/active_learning_seh.gin")
    parser.add_argument("--n", type=int, default=200, help="target number of seed molecules")
    parser.add_argument(
        "--oversample",
        type=float,
        default=2.0,
        help="sample this multiple of --n trajectories to absorb invalid/duplicate terminals",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/active_learning/seh/seed_seh.csv",
        help="output seed CSV path (matches OracleLabeledDataset.seed_csv in the config)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help=(
            "Override gin's user_root_dir (where the sampler's run dir/logs land). "
            "Required on a Balam compute node, where $HOME is read-only: the default "
            "'experiments' is a repo-relative path that only works on the writable "
            "login node. Mirrors scripts/active_learning.py."
        ),
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    # run_name is required by the base config's logger/run-dir macros; the trainer
    # is only used here as a *sampler* (no GFN training happens).
    run_name = f"make_seh_seed/{get_time_stamp()}"
    bindings = [f'run_name="{run_name}"']
    if args.root_dir is not None:
        bindings.append(f'user_root_dir="{args.root_dir}"')
    gin.parse_config_files_and_bindings([args.cfg], bindings=bindings)

    # Build the same components the loop uses. The policy is freshly initialised
    # (no .train() call), so these are random in-distribution rollouts = D_0.
    trainer = Trainer()
    oracle = DockingSEHOracle()
    print(
        f"[seed] sampling up to {args.n} unique molecules "
        f"(oversample x{args.oversample}) from the untrained policy...",
        flush=True,
    )
    smiles = _sample_unique(trainer, args.n, args.oversample)
    print(f"[seed] sampled {len(smiles)} unique candidates; docking against sEH...", flush=True)

    # Free torch's cached GPU memory before docking so the QuickVina2-GPU subprocess
    # can allocate on the same GPU (Logs/014). Low risk here (this samples from an
    # untrained policy — no training footprint), but kept consistent with the loop.
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    t0 = time.time()
    scores = oracle.score(smiles)
    dock_dt = time.time() - t0
    rows = [(s, y) for s, y in zip(smiles, scores) if y is not None and not math.isnan(y)]
    per_mol = dock_dt / len(smiles) if smiles else float("nan")
    print(
        f"[seed] docked {len(smiles)} molecules in {dock_dt:.1f}s "
        f"({per_mol:.2f}s/mol avg); {len(rows)} succeeded, "
        f"{len(smiles) - len(rows)} dropped as nan.",
        flush=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["smiles", "label"])
        for smi, y in rows:
            writer.writerow([smi, f"{y:.3f}"])
    if rows:
        ys = [y for _, y in rows]
        print(
            f"[seed] wrote {out_path} ({len(rows)} rows). "
            f"Vina label range [{min(ys):.2f}, {max(ys):.2f}] kcal/mol, "
            f"mean {sum(ys)/len(ys):.2f}.",
            flush=True,
        )
    trainer.close()


if __name__ == "__main__":
    main()
