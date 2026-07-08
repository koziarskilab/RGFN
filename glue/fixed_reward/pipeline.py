"""``FixedRewardPipeline`` — the RGFN-paper single-shot training mode (no active learning).

Our active-learning pipeline (``glue/active_learning/loop.py``) trains RGFN through
``[bengio2021gflownet]`` Alg. 1: fit a learned proxy ``M`` on oracle labels, train the
GFN against ``r(x) = M(x)^beta``, label a query batch with the expensive oracle ``O``,
refit ``M``, repeat for ``N`` rounds. The **RGFN paper itself** (``[koziarski2024rgfn]``
§4) does none of that: it trains the GFN **once** against a **fixed reward generator**
and reads off the result. This pipeline is that mode — the single-shot counterpart to
``ActiveLearningLoop``, kept deliberately parallel to it so the two are easy to compare.

Terminology (the project's convention — see ``docs/RESEARCH_CONTEXT.md`` glossary):
a fixed-reward run has **no oracle**. There is only a *reward generator* talking directly
to the GFN every training step. Two flavours, both wired identically here:

    - a **surrogate** reward generator — e.g. the pretrained sEH MPNN ``SehMoleculeProxy``
      (a fast model standing in for expensive docking); or
    - a reward generator that is **not** a surrogate — e.g. ``DockingMoleculeProxy``,
      real GPU docking called directly in the loop.

Deliberate divergences from ``ActiveLearningLoop`` (this is where "fixed reward" differs):
    - **no proxy refit, no oracle, no seed ``D_0``, no rounds, no replay reset.**
    - the reward generator is frozen: it is the *same* proxy instance the trainer's
      reward already uses (wire both to ``%train_proxy``), never retrained here.
    - the per-candidate ``score`` we record is the reward generator's **own value**
      (the proxy value), not an independent oracle label — because there is no oracle.

What it reuses verbatim (so the two pipelines can't drift): ``Trainer.train()`` for the
single GFN run, the AL loop's terminal-state sampling logic, ``extract_route`` for
synthesis routes, ``PhaseTimer`` for wall-clock accounting, and the standard
``CandidateDataset`` writer so a fixed-reward run is just another conformant entrant the
benchmark harness can read next to the baselines.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gin

from glue.active_learning.route import extract_route, route_to_str
from glue.active_learning.timing import PhaseTimer
from glue.datasets.candidates import CandidateDataset
from glue.metrics.dataset_metrics import batch_metrics
from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateTerminal


@gin.configurable()
class FixedRewardPipeline:
    """Single-shot pipeline: train the GFN once against a fixed reward generator, then
    sample a batch of molecules and emit them as a standard candidate dataset."""

    def __init__(
        self,
        trainer,
        reward_generator,
        n_samples: int = 1000,
        sample_oversample: float = 4.0,
        top_k: int = 100,
        run_dir: Optional[str] = None,
        system: Optional[str] = None,
        seed: Optional[int] = None,
        generator_name: str = "rgfn",
        reward_name: Optional[str] = None,
        score_units: Optional[str] = None,
    ):
        """
        Args:
            trainer: a configured RGFN ``Trainer`` whose reward references the same
                ``reward_generator`` instance passed here (wire both to ``%train_proxy``).
            reward_generator: the frozen reward generator (a ``ProxyBase`` — e.g.
                ``SehMoleculeProxy`` or ``DockingMoleculeProxy``). Never refit here; we
                only *read* it, both as the GFN reward (via the trainer) and to score the
                final sampled batch. Its ``higher_is_better`` sets the score orientation.
            n_samples: number of *unique valid* molecules to emit as candidates.
            sample_oversample: sample this multiple of ``n_samples`` trajectories to
                absorb invalid/duplicate terminals before trimming.
            top_k: size of the ``top_k.csv`` deliverable.
            run_dir: where to write outputs (``<run_dir>/fixed_reward/``); defaults to the
                trainer's run_dir.
            system: target-system tag (e.g. ``"seh"``) recorded in the manifest.
            seed: run seed recorded in the manifest (the driver binds it from ``--seed``).
            generator_name: name stamped on candidates + manifest (e.g. ``"rgfn"``).
            reward_name: name of the reward generator recorded in the manifest for
                provenance (e.g. ``"seh_proxy"``, ``"seh_docking"``). NOTE: stored under
                the manifest's ``oracle`` field for schema compatibility with the harness,
                but it is a *reward generator*, not an oracle (see module docstring).
            score_units: free-text description of what ``score`` means (provenance).
        """
        self.trainer = trainer
        self.reward_generator = reward_generator
        self.n_samples = n_samples
        self.sample_oversample = sample_oversample
        self.top_k = top_k
        self.run_dir = Path(run_dir) if run_dir else Path(trainer.run_dir)
        self.system = system
        self.seed = seed
        self.generator_name = generator_name
        self.reward_name = reward_name
        self.score_units = score_units

    # --------------------------------------------------------------------- driver
    def run(self) -> List[tuple]:
        """Train once against the fixed reward, sample, emit candidates; return Top-K."""
        out_dir = self.run_dir / "fixed_reward"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = getattr(self.trainer, "logger", None)
        # Reuse the AL timer (its labels/prefix are generic enough); a fixed-reward run
        # has just one "round", so everything is logged under round 1.
        timer = PhaseTimer(logger=logger, csv_path=out_dir / "phase_timings.csv")

        higher_is_better = bool(self.reward_generator.higher_is_better)
        print(
            f"[FR] start: single-shot training, reward={self.reward_name or 'reward_generator'} "
            f"(higher_is_better={higher_is_better}), n_samples={self.n_samples}",
            flush=True,
        )

        # 1. train pi_theta ONCE against r(x) = reward_generator(x)^beta.
        with timer.phase("train_gfn", 1):
            self.trainer.train()

        # 2. sample a batch from the trained forward policy (keeping each route + the
        #    terminal state object, which we score directly below).
        #    Free torch's cached GPU memory first — the docking reward generator runs
        #    QuickVina2-GPU in a subprocess that needs device memory (Logs/014). No-op
        #    for the CPU sEH proxy.
        self._free_torch_gpu_cache()
        with timer.phase("sample_batch", 1):
            smiles, routes, states = self._sample_batch()
        print(f"[FR] sampled {len(smiles)} unique valid candidates", flush=True)

        # 3. score the batch with the reward generator itself (NOT an oracle). This is
        #    the frozen model's own value — the same quantity the GFN was trained on.
        with timer.phase("score", 1):
            scores, components = self._score_with_reward_generator(states)

        # 4. write the standard candidate dataset + a batch-metrics sidecar + Top-K.
        self._write_candidates(out_dir, smiles, scores, routes, components, higher_is_better)
        self._log_metrics(logger, smiles, scores, higher_is_better)

        pairs = [(s, sc) for s, sc in zip(smiles, scores) if sc is not None and sc == sc]
        pairs.sort(key=lambda p: p[1], reverse=higher_is_better)
        top = pairs[: self.top_k]
        self._write_top_k(out_dir / "top_k.csv", top)
        timer.report_total()
        print(f"[FR] done. candidates + Top-{self.top_k} written to {out_dir}", flush=True)
        return top

    # ----------------------------------------------------------------- internals
    def _sample_batch(self) -> Tuple[List[str], List[Dict], List]:
        """Sample from the trained forward policy; return ``(smiles, routes, states)``.

        Parallel lists: unique valid terminal SMILES, each molecule's structured route
        (``extract_route``), and the terminal ``ReactionStateTerminal`` object (scored
        directly by :meth:`_score_with_reward_generator`). Mirrors
        ``ActiveLearningLoop._sample_query_batch`` but also returns the state objects so
        we can read the reward generator's value without re-sampling.
        """
        sampler = self.trainer.train_forward_sampler
        n_sample = int(self.n_samples * self.sample_oversample)
        batch_size = self.trainer.train_batch_size
        seen: set = set()
        smiles: List[str] = []
        routes: List[Dict] = []
        states: List = []
        for trajectories in sampler.get_trajectories_iterator(n_sample, batch_size):
            last_states = trajectories.get_last_states_flat()
            for i, state in enumerate(last_states):
                if not isinstance(state, ReactionStateTerminal):
                    continue  # skip early-terminal / invalid molecules
                smi = state.molecule.smiles
                if smi in seen:
                    continue
                seen.add(smi)
                smiles.append(smi)
                states.append(state)
                routes.append(
                    extract_route(trajectories._states_list[i], trajectories._actions_list[i])
                )
                if len(smiles) >= self.n_samples:
                    return smiles, routes, states
        return smiles, routes, states

    def _score_with_reward_generator(
        self, states: List
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """Score terminal states with the frozen reward generator (its own value).

        Returns parallel ``(scores, components)`` where ``components[i]`` is a dict of any
        per-molecule breakdown the reward generator exposes (e.g. a docking proxy's Vina
        components), preserved as extra candidate columns. Empty dicts when there are none.
        """
        if not states:
            return [], []
        output = self.reward_generator.compute_proxy_output(states)
        scores = [float(v) for v in output.value.detach().cpu().reshape(-1).tolist()]
        components: List[Dict[str, float]] = [{} for _ in states]
        if output.components:
            for key, tensor in output.components.items():
                vals = tensor.detach().cpu().reshape(-1).tolist()
                for i, v in enumerate(vals):
                    if i < len(components):
                        components[i][key] = float(v)
        return scores, components

    def _write_candidates(
        self,
        out_dir: Path,
        smiles: List[str],
        scores: List[float],
        routes: List[Dict],
        components: List[Dict[str, float]],
        higher_is_better: bool,
    ) -> None:
        """Write the standard candidate dataset under ``<out_dir>/candidates/``."""
        ds = CandidateDataset(
            out_dir / "candidates",
            generator=self.generator_name,
            oracle=self.reward_name,  # provenance: the reward generator's name
            system=self.system,
            seed=self.seed,
            score_higher_is_better=higher_is_better,
            score_units=self.score_units,
            source=getattr(getattr(self.trainer, "logger", None), "run_name", None),
            notes=(
                "Fixed-reward (single-shot) run: no oracle. 'score' is the reward "
                "generator's own value; 'step' is always 1 (no active-learning rounds)."
            ),
        )
        for smi, score, route, comp in zip(smiles, scores, routes, components):
            extra = {"route_str": route_to_str(route)}
            if comp:
                extra.update(comp)
            ds.add(smiles=smi, score=score, step=1, route=route, extra=extra)
        ds.write()

    def _log_metrics(self, logger, smiles: List[str], scores: List[float], higher: bool) -> None:
        """Compute + log set-level diversity/medchem/score metrics for the sampled batch."""
        try:
            metrics = batch_metrics(
                smiles,
                labels=scores,
                oracle_threshold=None,
                oracle_higher_is_better=higher,
            )
        except Exception as exc:  # noqa: BLE001 - metrics must not crash the run
            print(f"[FR] WARNING batch metrics failed: {exc}", flush=True)
            return
        valid = [s for s in scores if s is not None and s == s]
        metrics = {
            "fr_score_mean": (sum(valid) / len(valid)) if valid else float("nan"),
            "fr_score_best": (max(valid) if higher else min(valid)) if valid else float("nan"),
            **metrics,
        }
        print(
            f"[FR] batch: n={metrics.get('n_suggested')} modes={metrics.get('num_modes')} "
            f"div={metrics.get('internal_diversity', float('nan')):.2f} "
            f"MW={metrics.get('mol_weight_mean', float('nan')):.0f} "
            f"score_best={metrics['fr_score_best']:.3f}",
            flush=True,
        )
        if logger is not None:
            logger.log_metrics(metrics=metrics, prefix="fixed_reward")

    @staticmethod
    def _free_torch_gpu_cache() -> None:
        """Return torch's reserved-but-unused GPU memory so the docking reward generator's
        subprocess (QuickVina2-GPU / OpenCL) can allocate (Logs/014). Best-effort; a no-op
        without torch/CUDA and never fatal."""
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info()
                print(
                    f"[FR] freed torch GPU cache before sampling/scoring -> "
                    f"{free // 1024 // 1024} / {total // 1024 // 1024} MiB free",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[FR] WARNING could not free torch GPU cache: {exc}", flush=True)

    @staticmethod
    def _write_top_k(path: Path, rows: List[tuple]) -> None:
        import csv

        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["rank", "smiles", "reward_score"])
            for rank, (smi, score) in enumerate(rows, start=1):
                writer.writerow([rank, smi, score])
