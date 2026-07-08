"""``FragGFNActiveLearningLoop`` — the FragGFN counterpart of
``glue/active_learning/loop.py`` (``[bengio2021gflownet]`` Alg. 1).

Same algorithm as the RGFN loop, step for step::

    Init: proxy M; policy π_θ (fragment-GFN); oracle O; i = 1
    while i <= N:
        fit M on dataset D_{i-1}
        train π_θ with reward r(x) = M(x)^β            # fragment-GFN, not RGFN
        sample query batch B = {x_1..x_b}, x_j ~ π_θ
        evaluate B with O:  D̂_i = {(x_j, O(x_j))}      # via the oracle bridge
        D_i = D̂_i ∪ D_{i-1}
        i = i + 1
    return TopK(D_N)

Two deliberate differences from the RGFN loop, both forced by the env split
(see ``validation/generators/fraggfn/README.md``):

  * **Generator.** π_θ is Recursion's fragment-based GFlowNet
    (``FragMolBuildingEnvContext`` via :class:`FragGFNTrainer`), trained against
    the *same* learned proxy ``M`` as RGFN. The molecules it builds have **no
    synthesis route** (the headline non-synthesizable baseline).
  * **Oracle call.** ``O`` lives in the ``rgfn`` env (gnina + QuickVina2-GPU +
    ``glue``), unreachable from this py3.10 env. We label each batch by shelling
    out to ``scripts/score_batch.py`` under ``rgfn`` — the shared scoring standard.
    That bridge ALSO writes the standard candidate-dataset format + per-round
    ``batch_metrics`` (the same ``glue`` code RGFN's ``SuggestionLog`` uses), so
    FragGFN output is directly comparable to RGFN's.

The invariant is preserved: oracle labels enter training only by refitting ``M``
(never as a direct GFN reward). ``M`` is held by the task and refit in place each
round; ``num_workers=0`` makes the refit visible to sampling immediately.
"""

import csv
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gflownet.trainer import cycle
from rdkit import Chem

from validation.generators.fraggfn.proxy import AtomMPNNProxy
from validation.generators.fraggfn.task import FragGFNTrainer


def _canonical(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def _free_gpu_cache() -> None:
    """Return this process's reserved-but-unused GPU memory to the driver before the
    docking bridge runs. The bridge's QuickVina2-GPU (a subprocess) needs GPU memory,
    but our torch training holds the card's caching allocator on the same GPU — which
    starves it (Logs/014: 40 GB → ~1 GB free → every dock ``no_pose``). ``empty_cache()``
    frees the reserved blocks; the small live model stays resident. Best-effort/no-op
    without torch/CUDA."""
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

    Mirrors ``glue.datasets.OracleLabeledDataset`` (which can't be imported here:
    it lives behind ``glue/__init__`` → ``rgfn``). Lower label = better glue, so
    Top-K sorts ascending (the 6TD3 ``dvina`` differential, like the RGFN loop).
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


class FragGFNActiveLearningLoop:
    """Outer active-learning loop driving the fragment-GFN generator."""

    def __init__(
        self,
        trainer: FragGFNTrainer,
        proxy: AtomMPNNProxy,
        dataset: LabelStore,
        bridge_cmd: List[str],
        run_dir: str,
        n_rounds: int = 3,
        n_train_steps: int = 300,
        query_batch_size: int = 32,
        sample_oversample: float = 4.0,
        top_k: int = 16,
        seed_csv: Optional[str] = None,
        system: Optional[str] = None,
        seed: int = 42,
        oracle_threshold: float = -2.0,
        oracle_higher_is_better: bool = False,
    ):
        """
        Args:
            trainer: a built :class:`FragGFNTrainer` (its task holds ``proxy``).
            proxy: the refit-able ``AtomMPNNProxy`` (in-loop reward ``M``).
            dataset: the accumulating :class:`LabelStore`, seeded with ``D_0``.
            bridge_cmd: argv prefix that runs the oracle bridge under the ``rgfn``
                env, e.g. ``["conda","run","--no-capture-output","-n","rgfn",
                "python","scripts/score_batch.py","--oracle","docking_6td3_gpu",
                "--oracle-arg","num_modes=9", ...]``. The loop appends
                ``--in/--out/--suggestions-dir/--step/...`` per round.
            run_dir: where per-round CSVs + suggestions land.
            n_rounds/n_train_steps/query_batch_size/top_k: budget (match the RGFN
                6td3 GPU config for an apples-to-apples comparison).
            seed_csv: passed to the bridge as ``--reference-csv`` (novelty metric).
            oracle_higher_is_better: orientation of the oracle/labels; MUST equal
                the proxy's, else the GFN would be rewarded for the wrong end.
        """
        self.trainer = trainer
        self.proxy = proxy
        self.dataset = dataset
        self.bridge_cmd = bridge_cmd
        self.run_dir = Path(run_dir)
        self.n_rounds = n_rounds
        self.n_train_steps = n_train_steps
        self.query_batch_size = query_batch_size
        self.sample_oversample = sample_oversample
        self.top_k = top_k
        self.seed_csv = seed_csv
        self.system = system
        self.seed = seed
        self.oracle_threshold = oracle_threshold
        self.oracle_higher_is_better = oracle_higher_is_better
        self._it = 0  # persistent GFN step counter across rounds (for lr schedules)

        if proxy.higher_is_better != oracle_higher_is_better:
            raise ValueError(
                f"Sign mismatch: proxy.higher_is_better={proxy.higher_is_better} but "
                f"oracle_higher_is_better={oracle_higher_is_better}."
            )

    # --------------------------------------------------------------------- driver
    def run(self) -> List[Tuple[str, float]]:
        out_dir = self.run_dir / "active_learning"
        sug_dir = out_dir / "suggestions"
        out_dir.mkdir(parents=True, exist_ok=True)
        sug_dir.mkdir(parents=True, exist_ok=True)

        if len(self.dataset) < 2:
            raise ValueError("FragGFN AL needs a seed D_0 (>=2 labelled molecules).")
        print(
            f"[FGFN-AL] start: {self.n_rounds} rounds, seed |D_0|={len(self.dataset)}", flush=True
        )

        for rnd in range(1, self.n_rounds + 1):
            t0 = time.time()
            # 1. fit M on D_{i-1}
            smiles, labels = self.dataset.to_lists()
            fit_metrics = self.proxy.fit(smiles, labels)
            print(
                f"[FGFN-AL] round {rnd}: fit M on |D|={len(self.dataset)} -> {fit_metrics}",
                flush=True,
            )

            # 2. train π_θ against r(x) = M(x)^β  (fragment-GFN inner loop)
            t_fit = time.time()
            self._train_steps(self.n_train_steps)
            t_train = time.time()

            # 3. sample a query batch B ~ π_θ
            batch = self._sample_query_batch()
            print(f"[FGFN-AL] round {rnd}: sampled {len(batch)} unique candidates", flush=True)
            t_sample = time.time()

            # 4. label B with O via the oracle bridge (also writes standard format)
            scores = self._score_via_bridge(batch, rnd, sug_dir)
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
                f"[FGFN-AL] round {rnd}: |D|={len(self.dataset)} (+{n_added}); "
                f"oracle mean={sum(valid)/len(valid):.3f} best={best:.3f} "
                f"| t: fit={t_fit-t0:.0f}s train={t_train-t_fit:.0f}s "
                f"sample={t_sample-t_train:.0f}s score={t_score-t_sample:.0f}s"
                if valid
                else f"[FGFN-AL] round {rnd}: |D|={len(self.dataset)} (+{n_added}); no valid scores",
                flush=True,
            )

            # Fail loudly if the oracle produced no usable label for the *entire*
            # batch (all NaN) — mirrors the post-014 guard in
            # glue/active_learning/loop.py. This is the signature of a wholesale
            # oracle-backend failure (e.g. a wedged GPU/OpenCL node returning all
            # no_pose, Logs/014): D can't grow, so the next round would refit M on an
            # unchanged D and burn another fragment-GFN training run for nothing.
            # Per-round provenance (suggestions/shards, dataset_round CSV) is already
            # written, so the failure stays fully inspectable.
            if batch and not valid:
                raise RuntimeError(
                    f"FragGFN AL round {rnd}: the oracle returned no usable score for any of "
                    f"the {len(batch)} sampled molecules (all NaN). The dataset cannot grow, so "
                    f"continuing would retrain M on an unchanged D. This usually means the oracle "
                    f"backend failed wholesale (e.g. a wedged GPU/OpenCL node — see Logs/014). "
                    f"Aborting. Inspect {sug_dir} and resubmit on a healthy node."
                )

        # Assemble the canonical standard candidate dataset from per-round shards.
        self._finalize_via_bridge(sug_dir)
        top = self.dataset.top_k(self.top_k)
        self._write_top_k(out_dir / "top_k.csv", top)
        print(f"[FGFN-AL] done. Top-{self.top_k} written ({len(top)} rows).", flush=True)
        return top

    # ----------------------------------------------------------------- internals
    def _train_steps(self, n: int) -> None:
        """Drive the fragment-GFN trainer for ``n`` minibatches against the current
        ``M`` (held by the task). Mirrors the inner loop of ``GFNTrainer.run`` but
        under our control so we can refit ``M`` between rounds."""
        tr = self.trainer
        tr.model.to(tr.device)
        tr.sampling_model.to(tr.device)
        train_dl = tr.build_training_data_loader()
        start = self._it + 1
        for it, batch in zip(range(start, start + n), cycle(train_dl)):
            info = tr.train_batch(batch.to(tr.device), 0, 0, it)
            if it % max(1, tr.print_every) == 0:
                loss = info.get("loss", float("nan"))
                print(f"[FGFN-AL]   gfn step {it}: loss={loss:.3f}", flush=True)
        self._it += n
        del train_dl

    def _sample_query_batch(self) -> List[str]:
        """Sample unique valid terminal SMILES from the trained policy."""
        tr = self.trainer
        n_sample = int(self.query_batch_size * self.sample_oversample)
        tr.model.to(tr.device)
        tr.model.eval()
        cond_info = tr.task.sample_conditional_information(n_sample, self._it)
        trajs = tr.algo.create_training_data_from_own_samples(
            tr.model, n_sample, cond_info["encoding"].to(tr.device), random_action_prob=0.0
        )
        seen, batch = set(), []
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
            if len(batch) >= self.query_batch_size:
                break
        return batch

    def _score_via_bridge(self, batch: List[str], rnd: int, sug_dir: Path) -> List[float]:
        """Write the batch SMILES, run the oracle bridge under ``rgfn``, read labels.

        The bridge also appends the standard-format shard + batch_metrics row for
        this round (``--suggestions-dir``/``--step``)."""
        if not batch:
            return []
        smi_path = sug_dir / f"round_{rnd:03d}_batch.smi"
        lbl_path = sug_dir / f"round_{rnd:03d}_labels.csv"
        smi_path.write_text("\n".join(batch) + "\n")
        cmd = list(self.bridge_cmd) + [
            "--in",
            str(smi_path),
            "--out",
            str(lbl_path),
            "--suggestions-dir",
            str(sug_dir),
            "--step",
            str(rnd),
            "--generator",
            "fraggfn",
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
        print(f"[FGFN-AL] round {rnd}: bridge -> {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        return self._read_labels(lbl_path, batch)

    def _finalize_via_bridge(self, sug_dir: Path) -> None:
        cmd = list(self.bridge_cmd) + [
            "--finalize",
            "--suggestions-dir",
            str(sug_dir),
            "--generator",
            "fraggfn",
            "--seed",
            str(self.seed),
        ]
        if self.system:
            cmd += ["--system", self.system]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:  # finalize is provenance, not training
            print(f"[FGFN-AL] WARNING finalize failed: {e}", flush=True)

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
