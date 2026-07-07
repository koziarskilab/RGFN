"""``ScentActiveLearningLoop`` — the SCENT counterpart of
``glue/active_learning/loop.py`` (``[bengio2021gflownet]`` Alg. 1).

Same algorithm as the RGFN loop, step for step::

    Init: proxy M; policy π_θ (SCENT cost-guided reaction-GFN); oracle O; i = 1
    while i <= N:
        fit M on dataset D_{i-1}
        train π_θ with reward r(x) = M(x)^β            # SCENT, not RGFN
        sample query batch B = {x_1..x_b}, x_j ~ π_θ   # keep each route
        evaluate B with O:  D̂_i = {(x_j, O(x_j))}      # via the oracle bridge
        D_i = D̂_i ∪ D_{i-1}
        i = i + 1
    return TopK(D_N)

SCENT is a *fork of RGFN* (same lab) whose generator adds Recursive Cost Guidance, an
Exploitation Penalty, and a Dynamic Library. Because it shares RGFN's ``rgfn.api`` —
the same ``Trainer.train()``, the same forward sampler, the same ``ReactionState``
machine — this loop is almost line-for-line the RGFN loop: it drives a gin-configured
SCENT ``Trainer`` in process and refits the in-loop proxy ``M`` between rounds, exactly
like ``ActiveLearningLoop``. Two deliberate differences, both structural:

  * **Generator.** π_θ is SCENT's cost-guided reaction-GFlowNet, configured entirely
    via SCENT's own gin (``configs/scent_base.gin`` etc.), trained against the *same*
    learned proxy ``M`` (:class:`~validation.generators.scent.proxy.LearnedDockingProxy`,
    identical architecture to RGFN's ``LearnedGlueProxy``). SCENT IS synthesizable, so
    we record each molecule's route (``has_route=1`` + ``routes.jsonl``).
  * **Oracle call.** ``O`` (gnina + QuickVina2-GPU + ``glue``) lives in the ``rgfn``
    env, unreachable from the ``scent`` env (SCENT's package is named ``rgfn`` and
    shadows ours). We label each batch by shelling out to ``scripts/score_batch.py``
    under ``rgfn`` — the shared scoring standard. That bridge ALSO writes the standard
    candidate-dataset format + per-round ``batch_metrics`` (the same ``glue`` code
    RGFN's ``SuggestionLog`` uses), and — passed ``--routes`` — joins our per-round
    route JSONL so SCENT's output is directly comparable to RGFN's.

The invariant is preserved: oracle labels enter training only by refitting ``M``
(never as a direct GFN reward). ``M`` is the gin singleton the SCENT ``Reward`` wraps,
so refitting it in place updates the reward the generator trains against; we
``clear_cache()`` right after each fit so stale predictions aren't served.
"""

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gin

# Plain sibling imports (NOT package-relative): the runner puts THIS directory on
# sys.path[0] and deliberately keeps the repo root OFF it, so SCENT's installed
# `rgfn` fork isn't shadowed by our repo-local `rgfn/`. See run_scent_al.py.
from guidance_io import save_guidance_models  # noqa: E402  (sibling module)
from proxy import (  # noqa: E402  (sibling module, not a package import)
    LearnedDockingProxy,
)
from rdkit import Chem
from route import extract_route  # noqa: E402

from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateTerminal


def _canonical(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def _free_gpu_cache() -> None:
    """Return this process's reserved-but-unused GPU memory to the driver before the
    docking bridge runs. The bridge's QuickVina2-GPU (a subprocess) needs GPU memory,
    but our torch training holds the card's caching allocator on the same GPU — which
    can starve it (Logs/014: 40 GB → ~1 GB free → every dock ``no_pose``). SCENT's
    footprint left room in job 69513, but this makes it robust regardless.
    ``empty_cache()`` frees the reserved blocks; the small live model stays resident.
    Best-effort/no-op without torch/CUDA."""
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class LabelStore:
    """Accumulating ``canonical-SMILES -> label`` store (the dataset ``D``).

    Mirrors ``glue.datasets.OracleLabeledDataset`` (which can't be imported in the
    ``scent`` env: it lives behind ``glue/__init__`` → our ``rgfn``). Lower label =
    better glue, so Top-K sorts ascending (the 6TD3 ``dvina`` differential).
    """

    def __init__(self, lower_is_better: bool = True):
        self._data: Dict[str, float] = {}
        self.lower_is_better = lower_is_better

    def __len__(self) -> int:
        return len(self._data)

    def load_seed(self, csv_path: str, smiles_col: str = "smiles", label_col: str = "label") -> int:
        added = 0
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    y = float(row[label_col])
                except (TypeError, ValueError, KeyError):
                    continue
                if self.add_one(row.get(smiles_col, ""), y):
                    added += 1
        return added

    def add_one(self, smiles: str, label: float) -> bool:
        canon = _canonical(smiles)
        if canon is None or label is None or label != label:  # invalid / NaN
            return False
        self._data[canon] = float(label)
        return True

    def add(self, smiles: List[str], labels: List[float]) -> int:
        return sum(int(self.add_one(s, y)) for s, y in zip(smiles, labels))

    def to_lists(self) -> Tuple[List[str], List[float]]:
        items = list(self._data.items())
        return [s for s, _ in items], [y for _, y in items]

    def top_k(self, k: int) -> List[Tuple[str, float]]:
        return sorted(self._data.items(), key=lambda kv: kv[1], reverse=not self.lower_is_better)[
            :k
        ]

    def save_csv(self, path: str) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["smiles", "label"])
            for s, y in self._data.items():
                w.writerow([s, y])


@gin.configurable()
class ScentActiveLearningLoop:
    """Outer active-learning loop driving SCENT's cost-guided reaction-GFN generator."""

    def __init__(
        self,
        trainer,
        proxy: LearnedDockingProxy,
        run_dir: str,
        repo_root: str,
        seed_csv: str,
        n_rounds: int = 3,
        query_batch_size: int = 32,
        sample_oversample: float = 4.0,
        top_k: int = 16,
        reset_replay_each_round: bool = True,
        smiles_column: str = "smiles",
        label_column: str = "label",
        system: Optional[str] = "6td3",
        seed: int = 42,
        oracle_threshold: float = -2.0,
        oracle_higher_is_better: bool = False,
        # --- oracle bridge (cross-env scoring under the rgfn env) ----------------
        conda_exe: str = "conda",
        oracle_env: str = "rgfn",
        score_script: str = "scripts/score_batch.py",
        oracle_name: str = "docking_6td3_gpu",
        oracle_args: Optional[Dict] = None,
    ):
        """
        Args:
            trainer: a gin-built SCENT ``Trainer`` whose ``Reward`` wraps ``proxy``
                (wire both to ``%train_proxy`` in the config, exactly as RGFN does).
            proxy: the refit-able ``LearnedDockingProxy`` (the in-loop reward ``M``).
            run_dir: where per-round CSVs + suggestions land (absolute; the runner
                binds it to a timestamped dir, like scripts/active_learning.py).
            repo_root: the RGFN-Fork repo root — the working directory the bridge
                subprocess runs in (so ``scripts/score_batch.py`` + ``experiments/…``
                resolve). The loop itself runs with CWD = the SCENT clone.
            seed_csv: seed dataset ``D_0`` CSV (also passed to the bridge as
                ``--reference-csv`` for the novelty metric). Absolute path.
            n_rounds/query_batch_size/sample_oversample/top_k: budget — match the
                RGFN 6td3 GPU config for an apples-to-apples comparison.
            oracle_higher_is_better: orientation of the oracle/labels; MUST equal
                the proxy's, else the GFN would be rewarded for the wrong end.
            conda_exe/oracle_env/score_script/oracle_name/oracle_args: the oracle
                bridge — these reproduce ``configs/glue/active_learning_6td3_gpu.gin``'s
                oracle so SCENT sees the identical scorer (``--oracle-arg KEY=VAL``).
        """
        self.trainer = trainer
        self.proxy = proxy
        self.run_dir = Path(run_dir)
        self.repo_root = Path(repo_root)
        self.seed_csv = seed_csv
        self.n_rounds = n_rounds
        self.query_batch_size = query_batch_size
        self.sample_oversample = sample_oversample
        self.top_k = top_k
        self.reset_replay_each_round = reset_replay_each_round
        self.smiles_column = smiles_column
        self.label_column = label_column
        self.system = system
        self.seed = seed
        self.oracle_threshold = oracle_threshold
        self.oracle_higher_is_better = oracle_higher_is_better

        if proxy.higher_is_better != oracle_higher_is_better:
            raise ValueError(
                f"Sign mismatch: proxy.higher_is_better={proxy.higher_is_better} but "
                f"oracle_higher_is_better={oracle_higher_is_better}. Set "
                "LearnedDockingProxy.higher_is_better to match the oracle in the gin config."
            )

        self.dataset = LabelStore(lower_is_better=not oracle_higher_is_better)
        n_seed = self.dataset.load_seed(seed_csv, self.smiles_column, self.label_column)
        print(f"[SCENT-AL] seeded D_0 with {n_seed} labelled molecules from {seed_csv}", flush=True)

        # Build the bridge argv prefix once (per-round --in/--out/--routes/… appended).
        args = oracle_args or {}
        self.bridge_cmd: List[str] = [
            conda_exe,
            "run",
            "--no-capture-output",
            "-n",
            oracle_env,
            "python",
            score_script,
            "--oracle",
            oracle_name,
        ]
        for k, v in dict(args).items():
            self.bridge_cmd += ["--oracle-arg", f"{k}={v}"]

    # --------------------------------------------------------------------- driver
    def run(self) -> List[Tuple[str, float]]:
        out_dir = self.run_dir / "active_learning"
        sug_dir = out_dir / "suggestions"
        out_dir.mkdir(parents=True, exist_ok=True)
        sug_dir.mkdir(parents=True, exist_ok=True)

        if len(self.dataset) < 2:
            raise ValueError("SCENT AL needs a seed D_0 (>=2 labelled molecules).")
        print(
            f"[SCENT-AL] start: {self.n_rounds} rounds, seed |D_0|={len(self.dataset)}", flush=True
        )

        for rnd in range(1, self.n_rounds + 1):
            t0 = time.time()
            # 1. fit M on D_{i-1}
            smiles, labels = self.dataset.to_lists()
            fit_metrics = self.proxy.fit(smiles, labels)
            self.proxy.clear_cache()  # predictions changed -> drop stale cache
            print(
                f"[SCENT-AL] round {rnd}: fit M on |D|={len(self.dataset)} -> {fit_metrics}",
                flush=True,
            )

            # 2. train π_θ against r(x) = M(x)^β  (SCENT inner loop, gin Trainer)
            if self.reset_replay_each_round:
                self._reset_replay_buffer()
            t_fit = time.time()
            self.trainer.train()
            # Persist the backward-policy guidance-model weights (cost + decomposability
            # MLPs) beside the round's last_gfn.pt — objective.state_dict() drops them,
            # so this is what makes the trained P_B exactly recoverable. See guidance_io.
            self._save_guidance_models()
            t_train = time.time()

            # 3. sample a query batch B ~ π_θ (keeping each molecule's route)
            batch, routes = self._sample_query_batch()
            print(f"[SCENT-AL] round {rnd}: sampled {len(batch)} unique candidates", flush=True)
            t_sample = time.time()

            # 4. label B with O via the oracle bridge (also writes standard format +
            #    joins our per-round routes -> has_route=1)
            scores = self._score_via_bridge(batch, routes, rnd, sug_dir)
            t_score = time.time()

            # 5. D_i = D̂_i ∪ D_{i-1}
            n_added = self.dataset.add(batch, scores)
            self.dataset.save_csv(str(out_dir / f"dataset_round_{rnd:03d}.csv"))
            valid = [s for s in scores if s is not None and s == s]
            best = (
                (min(valid) if not self.oracle_higher_is_better else max(valid))
                if valid
                else float("nan")
            )
            print(
                f"[SCENT-AL] round {rnd}: |D|={len(self.dataset)} (+{n_added}); "
                f"oracle mean={sum(valid)/len(valid):.3f} best={best:.3f} "
                f"| t: fit={t_fit-t0:.0f}s train={t_train-t_fit:.0f}s "
                f"sample={t_sample-t_train:.0f}s score={t_score-t_sample:.0f}s"
                if valid
                else f"[SCENT-AL] round {rnd}: |D|={len(self.dataset)} (+{n_added}); no valid scores",
                flush=True,
            )

            # Fail loudly if the oracle produced no usable label for the entire batch
            # (all NaN) — mirrors the post-014 guard in glue/active_learning/loop.py and
            # the FragGFN loop. Signature of a wholesale oracle-backend failure (wedged
            # GPU/OpenCL node returning all no_pose, Logs/013/014): D can't grow, so the
            # next round would refit M on an unchanged D and waste another training run.
            if batch and not valid:
                raise RuntimeError(
                    f"SCENT AL round {rnd}: the oracle returned no usable score for any of the "
                    f"{len(batch)} sampled molecules (all NaN). The dataset cannot grow, so "
                    f"continuing would retrain M on an unchanged D. This usually means the oracle "
                    f"backend failed wholesale (e.g. a wedged GPU/OpenCL node — see Logs/013). "
                    f"Aborting. Inspect {sug_dir} and resubmit on a healthy node."
                )

        # Assemble the canonical standard candidate dataset from per-round shards.
        self._finalize_via_bridge(sug_dir)
        top = self.dataset.top_k(self.top_k)
        self._write_top_k(out_dir / "top_k.csv", top)
        print(f"[SCENT-AL] done. Top-{self.top_k} written ({len(top)} rows).", flush=True)
        return top

    # ----------------------------------------------------------------- internals
    def _sample_query_batch(self) -> Tuple[List[str], List[Dict]]:
        """Sample from the trained forward policy; return ``(smiles, routes)``.

        Mirrors ``ActiveLearningLoop._sample_query_batch``: unique valid terminal
        SMILES plus, for each, the structured synthesis route reconstructed from its
        trajectory. ``get_last_states_flat()[i]`` indexes the same trajectory in
        ``_states_list``/``_actions_list``, which is how we recover each route.
        """
        sampler = self.trainer.train_forward_sampler
        n_sample = int(self.query_batch_size * self.sample_oversample)
        batch_size = self.trainer.train_batch_size
        seen, batch, routes = set(), [], []
        for trajectories in sampler.get_trajectories_iterator(n_sample, batch_size):
            last_states = trajectories.get_last_states_flat()
            for i, state in enumerate(last_states):
                if not isinstance(state, ReactionStateTerminal):
                    continue  # skip early-terminal / invalid molecules
                canon = _canonical(state.molecule.smiles)
                if canon is None or canon in seen:
                    continue
                seen.add(canon)
                batch.append(canon)
                routes.append(
                    extract_route(trajectories._states_list[i], trajectories._actions_list[i])
                )
                if len(batch) >= self.query_batch_size:
                    return batch, routes
        return batch, routes

    def _score_via_bridge(
        self, batch: List[str], routes: List[Dict], rnd: int, sug_dir: Path
    ) -> List[float]:
        """Write the batch SMILES + routes, run the oracle bridge under ``rgfn``,
        read labels. The bridge appends the standard-format shard + batch_metrics for
        this round and (via ``--routes``) stores the routes for finalize."""
        if not batch:
            return []
        smi_path = sug_dir / f"round_{rnd:03d}_batch.smi"
        lbl_path = sug_dir / f"round_{rnd:03d}_labels.csv"
        routes_path = sug_dir / f"round_{rnd:03d}_routes.jsonl"
        smi_path.write_text("\n".join(batch) + "\n")
        # One {"smiles": canonical, ...route fields...} per line — the schema the
        # bridge's --routes expects (glue.active_learning.route), joined onto
        # candidates by SMILES on --finalize -> has_route=1 + num_reactions.
        with open(routes_path, "w") as fh:
            for smi, route in zip(batch, routes):
                fh.write(json.dumps({"smiles": smi, **route}) + "\n")

        cmd = list(self.bridge_cmd) + [
            "--in",
            str(smi_path),
            "--out",
            str(lbl_path),
            "--suggestions-dir",
            str(sug_dir),
            "--routes",
            str(routes_path),
            "--step",
            str(rnd),
            "--generator",
            "scent",
            "--seed",
            str(self.seed),
            "--threshold",
            str(self.oracle_threshold),
        ]
        if self.system:
            cmd += ["--system", self.system]
        if self.seed_csv:
            cmd += ["--reference-csv", str(self.seed_csv)]
        _free_gpu_cache()  # let the bridge's QuickVina2-GPU allocate (Logs/014)
        print(
            f"[SCENT-AL] round {rnd}: bridge (cwd={self.repo_root}) -> {' '.join(cmd)}", flush=True
        )
        # The bridge resolves scripts/score_batch.py + experiments/… relative to the
        # repo root; the loop's own CWD is the SCENT clone, so run the subprocess there.
        subprocess.run(cmd, check=True, cwd=str(self.repo_root))
        return self._read_labels(lbl_path, batch)

    def _finalize_via_bridge(self, sug_dir: Path) -> None:
        cmd = list(self.bridge_cmd) + [
            "--finalize",
            "--suggestions-dir",
            str(sug_dir),
            "--generator",
            "scent",
            "--seed",
            str(self.seed),
        ]
        if self.system:
            cmd += ["--system", self.system]
        try:
            subprocess.run(cmd, check=True, cwd=str(self.repo_root))
        except subprocess.CalledProcessError as e:  # finalize is provenance, not training
            print(f"[SCENT-AL] WARNING finalize failed: {e}", flush=True)

    def _save_guidance_models(self) -> None:
        """Write the backward-policy guidance-model weights beside ``last_gfn.pt``.

        ``last_gfn.pt`` holds only ``objective.state_dict()`` = forward policy + logZ; the
        cost/decomposability guidance MLPs live in a plain Python list on
        ``JointlyGuidedBackwardPolicy`` and are dropped by ``state_dict()``. Saving them
        here makes the round's trained ``P_B`` exactly recoverable. Best-effort."""
        try:
            ckpt_dir = Path(self.trainer.run_dir) / "train" / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            gpath = ckpt_dir / "guidance_models.pt"
            keys = save_guidance_models(self.trainer.objective, gpath)
            print(
                f"[SCENT-AL] saved backward-policy guidance models -> {gpath} (keys={keys})",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[SCENT-AL] WARNING guidance-model save failed: {e}", flush=True)

    def _reset_replay_buffer(self) -> None:
        """Best-effort clear of the replay buffer between rounds (priorities go stale
        once M is refit). Guarded against upstream layout changes, like the RGFN loop."""
        rb = getattr(self.trainer, "train_replay_buffer", None)
        if rb is None:
            return
        if hasattr(rb, "states_list") and hasattr(rb, "states_set"):
            rb.states_list = []
            rb.states_set = set()
            if hasattr(rb, "proxy_value_array"):
                rb.proxy_value_array[:] = 0.0

    @staticmethod
    def _read_labels(path: Path, batch: List[str]) -> List[float]:
        """Read labels keyed by SMILES, returned in ``batch`` order (NaN if missing)."""
        by_smi: Dict[str, float] = {}
        if path.exists():
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        by_smi[row["smiles"]] = float(row["label"])
                    except (KeyError, TypeError, ValueError):
                        by_smi[row.get("smiles", "")] = float("nan")
        return [by_smi.get(s, float("nan")) for s in batch]

    @staticmethod
    def _write_top_k(path: Path, rows: List[Tuple[str, float]]) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "smiles", "oracle_label"])
            for rank, (smi, label) in enumerate(rows, start=1):
                w.writerow([rank, smi, label])
