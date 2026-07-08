"""``ActiveLearningLoop`` — the multi-round loop of ``[bengio2021gflownet]`` Alg. 1.

Verbatim algorithm (see ``docs/RESEARCH_CONTEXT.md``):

    Init: proxy M; policy pi_theta; oracle O; i = 1
    while i <= N:
        fit M on dataset D_{i-1}
        train pi_theta with reward r(x) = M(x)^beta
        sample query batch B = {x_1..x_b}, x_j ~ pi_theta
        evaluate B with O:  D̂_i = {(x_j, O(x_j))}
        D_i = D̂_i ∪ D_{i-1}
        i = i + 1
    return TopK(D_N)

Faithfulness notes / deliberate divergences (flagged for Balam validation in
``docs/REFACTOR_LOG.md``):
    - "Train pi_theta" = one full ``Trainer.train()`` run against the current
      proxy. Each round re-runs the same ``n_iterations`` range, warm-started from
      the previous round's policy + optimizer state (the policy persists across
      rounds, as intended). The paper uses "fewer iterations when fitting the
      generative model" in the multi-round setting — so set ``Trainer.n_iterations``
      smaller in the AL config than a single-shot run.
    - The proxy and the trainer's reward share the *same* ``LearnedGlueProxy``
      instance (a gin singleton), so refitting ``M`` in place updates the reward
      RGFN trains against. Right after ``fit`` we ``clear_cache()`` so stale
      cached predictions are not served.
    - ``O`` labels enter training only by retraining ``M`` — never as a direct
      RGFN reward. This loop preserves that invariant: it calls the oracle solely
      to grow ``D`` for the next ``fit``.
"""

from pathlib import Path
from typing import List, Optional

import gin

from glue.active_learning.acquisition_trace import AcquisitionTrace
from glue.active_learning.route import extract_route
from glue.active_learning.timing import PhaseTimer
from glue.datasets.oracle_labeled import OracleLabeledDataset
from glue.datasets.suggestion_log import SuggestionLog
from glue.oracles.base import GlueOracle
from glue.proxies.learned_proxy import LearnedGlueProxy
from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateTerminal


@gin.configurable()
class ActiveLearningLoop:
    """Outer active-learning loop sequencing proxy fit / GFN train / oracle label."""

    def __init__(
        self,
        trainer,
        proxy: LearnedGlueProxy,
        oracle: GlueOracle,
        dataset: OracleLabeledDataset,
        n_rounds: int = 5,
        query_batch_size: int = 200,
        sample_oversample: float = 4.0,
        top_k: int = 100,
        acquisition: str = "policy",
        reset_replay_each_round: bool = True,
        run_dir: Optional[str] = None,
        suggestion_log: Optional[SuggestionLog] = None,
        system: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            trainer: a configured RGFN ``Trainer`` (its reward must reference the
                same ``proxy`` instance passed here — wire both to ``%train_proxy``).
            proxy: the refit-able ``LearnedGlueProxy`` (the in-loop reward ``M``).
            oracle: the expensive scorer ``O`` (e.g. ``Docking6TD3Oracle``).
            dataset: the accumulating ``(smiles, label)`` store, optionally seeded
                with ``D_0``.
            n_rounds: number of outer rounds ``N``.
            query_batch_size: number of *unique valid* molecules to label per round
                (paper uses 200 in the molecule domain).
            sample_oversample: sample this multiple of ``query_batch_size``
                trajectories to absorb invalid/duplicate terminals before trimming.
            top_k: size of the final Top-K deliverable.
            acquisition: which policy proposes the query batch — the arm of the
                oracle-efficiency comparison (``[bengio2021gflownet]`` Fig. 7):

                * ``"policy"`` (default): the learned RGFN forward policy, trained
                  each round against ``M(x)^beta`` — the full Alg. 1 loop.
                * ``"random"``: a uniform-random policy over the *same* reaction
                  action space (upstream ``RandomSampler`` + ``UniformPolicy``), so
                  the baseline still proposes synthesizable-by-construction
                  molecules — a fair apples-to-apples control. In this mode the
                  proxy is neither fit nor consulted and the GFN is not trained;
                  every other knob (oracle, seed, budget, logging) is identical, so
                  the only difference is *how the query batch is chosen*.
            reset_replay_each_round: clear the replay buffer before each round so
                its sampling priorities are not stale w.r.t. the refit proxy.
            run_dir: where to write per-round dataset/Top-K CSVs; defaults to the
                trainer's run_dir.
            suggestion_log: optional ``SuggestionLog`` recording, per round, the
                suggested molecules + their synthesis routes + medchem/diversity
                metrics. Defaults to a fresh one rooted at ``run_dir``. Pure
                observation — never feeds training.
            system: target system tag (e.g. ``"6td3"``) recorded in the suggestion
                log's manifest provenance. Optional.
            seed: RNG seed of the run, recorded in the manifest provenance. The
                driver (``scripts/active_learning.py``) binds this from ``--seed``.
        """
        self.trainer = trainer
        self.proxy = proxy
        self.oracle = oracle
        self.dataset = dataset
        self.n_rounds = n_rounds
        self.query_batch_size = query_batch_size
        self.sample_oversample = sample_oversample
        self.top_k = top_k
        if acquisition not in ("policy", "random"):
            raise ValueError(f"acquisition must be 'policy' or 'random', got {acquisition!r}.")
        self.acquisition = acquisition
        self._random_sampler = None  # lazily built for acquisition='random'
        self.reset_replay_each_round = reset_replay_each_round
        self.run_dir = Path(run_dir) if run_dir else Path(trainer.run_dir)
        self.system = system
        self.seed = seed
        self.suggestion_log = (
            suggestion_log
            if suggestion_log is not None
            else SuggestionLog(
                oracle_name=getattr(oracle, "name", None),
                oracle_higher_is_better=oracle.higher_is_better,
                system=system,
                seed=seed,
                source=getattr(getattr(trainer, "logger", None), "run_name", None),
            )
        )
        self.suggestion_log.set_run_dir(str(self.run_dir))

    # --------------------------------------------------------------------- driver
    def run(self) -> List[tuple]:
        """Run all rounds and return ``TopK(D_N)`` as ``[(smiles, label), ...]``."""
        out_dir = self.run_dir / "active_learning"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Oracles that can attribute their cost to sub-steps (e.g. Docking6TD3Oracle:
        # embed / tier2_dock / pose_select / tier1_rescore) write a sibling CSV, so
        # we can see what dominates the oracle phase. Optional hook -- the loop stays
        # oracle-agnostic; oracles without it (mock, ...) are unaffected.
        enable_step_timing = getattr(self.oracle, "enable_step_timing", None)
        if callable(enable_step_timing):
            enable_step_timing(out_dir / "docking_timings.csv")
        logger = self.trainer.logger
        # Per-phase wall-clock, reported live and appended to CSV so the record
        # survives a mid-run crash (cf. experiment 009's SIGXCPU at the oracle).
        timer = PhaseTimer(logger=logger, csv_path=out_dir / "phase_timings.csv")
        # Oracle-call budget + best-so-far trace: the data behind the
        # top-k-vs-oracle-calls curve (Objective 1). Written per round; cannot be
        # reconstructed after the fact, so it is recorded for *every* AL run
        # regardless of arm/system/oracle.
        trace = AcquisitionTrace(
            out_dir / "oracle_calls.csv",
            acquisition=self.acquisition,
            higher_is_better=self.oracle.higher_is_better,
            top_k=self.top_k,
            seed=self.seed,
        )

        print(
            f"[AL] start: {self.n_rounds} rounds, acquisition={self.acquisition}, "
            f"seed |D_0|={len(self.dataset)}",
            flush=True,
        )
        # Snapshot the seed SMILES (D_0) up front: the suggestion log measures each
        # round's novelty against the *original* seed, so we capture it before any
        # round grows the dataset.
        seed_smiles, seed_labels = self.dataset.to_lists()
        if len(self.dataset) < 2:
            raise ValueError(
                "Active learning needs a seed dataset D_0 (>=2 labelled molecules). "
                "Set OracleLabeledDataset.seed_csv in the config."
            )
        # The proxy is fit on the oracle's labels, so their sign conventions must
        # agree or the GFN would be rewarded for the wrong end of the metric.
        if self.proxy.higher_is_better != self.oracle.higher_is_better:
            raise ValueError(
                "Sign mismatch: proxy.higher_is_better="
                f"{self.proxy.higher_is_better} but oracle.higher_is_better="
                f"{self.oracle.higher_is_better}. Set LearnedGlueProxy.higher_is_better "
                "to match the oracle in the gin config."
            )

        # Round-0 baseline: the Top-K of the seed D_0 at zero *new* oracle calls.
        # Both arms share the identical D_0, so this anchors the curve at a common
        # starting point — the divergence from here is exactly the acquisition
        # effect the Fig. 7 comparison isolates.
        base_row = trace.record(
            0, oracle_calls_round=0, n_labelled_round=0, smiles=seed_smiles, labels=seed_labels
        )
        print(
            f"[AL] baseline (D_0, 0 oracle calls): top{base_row['top_k']} "
            f"mean={base_row['topk_mean']:.3f} best={base_row['topk_best']:.3f}",
            flush=True,
        )

        for rnd in range(1, self.n_rounds + 1):
            # 1. fit M on D_{i-1}, and 2. train pi_theta against r(x) = M(x)^beta.
            # Both steps are the *learned* acquisition; the random-acquisition
            # baseline skips them entirely (it neither consults nor updates the
            # proxy/policy — it just samples uniformly, §Fig. 7 control).
            smiles, labels = self.dataset.to_lists()
            fit_metrics = {}
            if self.acquisition == "policy":
                with timer.phase("fit_proxy", rnd):
                    fit_metrics = self.proxy.fit(smiles, labels)
                    self.proxy.clear_cache()  # predictions changed -> drop stale cache
                print(
                    f"[AL] round {rnd}: fit M on |D|={len(self.dataset)} -> {fit_metrics}",
                    flush=True,
                )
                if self.reset_replay_each_round:
                    self._reset_replay_buffer()
                with timer.phase("train_gfn", rnd):
                    self.trainer.train()

            # 3. sample a query batch B (learned policy, or uniform-random baseline)
            with timer.phase("sample_batch", rnd):
                batch, routes = self._sample_query_batch()
            print(f"[AL] round {rnd}: sampled {len(batch)} unique candidates", flush=True)

            # Free torch's cached GPU memory before docking. The GPU docking oracle
            # runs QuickVina2-GPU in a *subprocess* that needs GPU memory via OpenCL,
            # but torch's caching allocator holds the device's memory after training
            # + sampling — starving the subprocess so every dock returns no_pose
            # (confirmed reproduction, Logs/014: free drops to ~1 GB -> all fail;
            # empty_cache() returns the reserved-but-unused blocks and docking
            # recovers; the small live model/optimizer stay resident). No-op for
            # CPU oracles / CPU-only runs.
            self._free_torch_gpu_cache(rnd)

            # 4. score B with the expensive oracle O. Prefer score_detailed() when
            #    the oracle exposes it (e.g. the GPU differential oracle) so the
            #    suggestion log can record the per-pose breakdown; fall back to the
            #    scalar score() for oracles that don't (mock, ...).
            with timer.phase("oracle_score", rnd):
                oracle_scores, oracle_details = self._score_batch(batch)

            # 5. D_i = D̂_i ∪ D_{i-1}
            n_added = self.dataset.add(batch, oracle_scores)

            # Provenance: record the suggested molecules + routes + per-batch
            # diversity/medchem metrics (does not touch the proxy-fit dataset).
            # Guarded: a logging bug must never kill a run whose training + docking
            # have already been paid for. Failures are loud but non-fatal.
            try:
                batch_metrics = self.suggestion_log.log_round(
                    rnd,
                    smiles=batch,
                    routes=routes,
                    labels=oracle_scores,
                    details=oracle_details,
                    reference_smiles=seed_smiles,
                )
            except Exception as exc:  # noqa: BLE001 - provenance must not crash the loop
                import traceback

                print(f"[AL] round {rnd}: WARNING suggestion log failed: {exc}", flush=True)
                traceback.print_exc()
                batch_metrics = {}

            valid_scores = [s for s in oracle_scores if s is not None and s == s]

            # Oracle-call budget + best-so-far trace (the curve's data). One call
            # per molecule *submitted* to O (docking failures still spent a call);
            # the running Top-K is taken over the whole accumulated D.
            ds_smiles, ds_labels = self.dataset.to_lists()
            trace_row = trace.record(
                rnd,
                oracle_calls_round=len(batch),
                n_labelled_round=len(valid_scores),
                smiles=ds_smiles,
                labels=ds_labels,
            )

            round_metrics = {
                "al_round": rnd,
                "al_dataset_size": len(self.dataset),
                "al_batch_unique": len(batch),
                "al_batch_labelled": n_added,
                "al_oracle_calls_round": len(batch),
                "al_oracle_calls_cumulative": trace.cumulative,
                "al_topk_mean": trace_row["topk_mean"],
                "al_topk_best": trace_row["topk_best"],
                "al_batch_oracle_mean": (sum(valid_scores) / len(valid_scores))
                if valid_scores
                else float("nan"),
                "al_batch_oracle_max": max(valid_scores) if valid_scores else float("nan"),
                **fit_metrics,
                **{f"batch_{k}": v for k, v in batch_metrics.items() if k != "al_round"},
            }
            logger.log_metrics(metrics=round_metrics, prefix="active_learning")
            self.dataset.save_csv(str(out_dir / f"dataset_round_{rnd:03d}.csv"))
            print(
                f"[AL] round {rnd}: |D|={len(self.dataset)} (+{n_added}); "
                f"oracle_calls={trace.cumulative}, "
                f"top{trace_row['top_k']} mean={trace_row['topk_mean']:.3f} "
                f"best={trace_row['topk_best']:.3f}; "
                f"batch modes={batch_metrics.get('num_modes')}, "
                f"MW={batch_metrics.get('mol_weight_mean', float('nan')):.0f}, "
                f"div={batch_metrics.get('internal_diversity', float('nan')):.2f}",
                flush=True,
            )
            timer.report_round(rnd)

            # Fail loudly if the oracle produced no usable label for the *entire*
            # batch: the dataset can't grow, so the next round would refit M on an
            # unchanged D and burn another full GFN training run for nothing. This
            # is the signature of a wholesale oracle-backend failure — e.g. a wedged
            # GPU/OpenCL node returning all no_pose (Logs/014, job 69481). Abort here
            # rather than silently grinding through the remaining rounds. The
            # round's provenance (suggestions/, dataset_round CSV) is already
            # written above, so the failure is fully inspectable.
            if batch and not valid_scores:
                raise RuntimeError(
                    f"Active-learning round {rnd}: the oracle returned no usable score for "
                    f"any of the {len(batch)} sampled molecules (all NaN). The dataset cannot "
                    f"grow, so continuing would retrain the proxy on an unchanged D. This "
                    f"usually means the oracle backend failed wholesale (e.g. a wedged "
                    f"GPU/OpenCL node — see Logs/014). Aborting. Inspect "
                    f"{out_dir / 'suggestions'} and resubmit on a healthy node."
                )

        timer.report_total()
        top = self.dataset.top_k(self.top_k)
        self._write_top_k(out_dir / "top_k.csv", top)
        print(f"[AL] done. Top-{self.top_k} written ({len(top)} rows).", flush=True)
        return top

    # ----------------------------------------------------------------- internals
    def _query_sampler(self):
        """The sampler that proposes the query batch, per the acquisition arm.

        ``"policy"`` → the trainer's learned forward sampler (trained this round).
        ``"random"`` → a uniform-random policy over the *same* reaction env, built
        once and reused. Both expose the identical ``get_trajectories_iterator`` /
        ``Trajectories`` interface, so ``_sample_query_batch`` (route extraction
        included) is unchanged between arms. Reusing the trainer's ``env`` keeps
        the two arms on exactly the same action space / building-block library."""
        if self.acquisition != "random":
            return self.trainer.train_forward_sampler
        if self._random_sampler is None:
            from rgfn.shared.policies.uniform_policy import UniformPolicy
            from rgfn.shared.samplers.random_sampler import RandomSampler

            base = self.trainer.train_forward_sampler
            self._random_sampler = RandomSampler(policy=UniformPolicy(), env=base.env, reward=None)
        return self._random_sampler

    def _sample_query_batch(self) -> tuple:
        """Sample the query batch from the acquisition sampler; return ``(smiles, routes)``.

        Returns parallel lists: the unique valid terminal SMILES and, for each,
        the structured synthesis route (``extract_route``) reconstructed from that
        molecule's trajectory. ``trajectories.get_last_states_flat()`` is built as
        ``[states[-1] for states in _states_list]``, so index ``i`` there indexes
        the same trajectory in ``_states_list``/``_actions_list`` — that alignment
        is how we recover each terminal molecule's route."""
        sampler = self._query_sampler()
        n_sample = int(self.query_batch_size * self.sample_oversample)
        batch_size = self.trainer.train_batch_size
        seen, batch, routes = set(), [], []
        for trajectories in sampler.get_trajectories_iterator(n_sample, batch_size):
            last_states = trajectories.get_last_states_flat()
            for i, state in enumerate(last_states):
                if not isinstance(state, ReactionStateTerminal):
                    continue  # skip early-terminal / invalid molecules
                smi = state.molecule.smiles
                if smi in seen:
                    continue
                seen.add(smi)
                batch.append(smi)
                routes.append(
                    extract_route(trajectories._states_list[i], trajectories._actions_list[i])
                )
                if len(batch) >= self.query_batch_size:
                    return batch, routes
        return batch, routes

    @staticmethod
    def _free_torch_gpu_cache(rnd: int) -> None:
        """Return torch's reserved-but-unused GPU memory to the driver so the GPU
        docking subprocess (QuickVina2-GPU / OpenCL) can allocate. Best-effort and
        guarded: a no-op without torch/CUDA, never fatal. Prints the free VRAM so a
        run's log shows the docker had room (Logs/014 memory-contention fix)."""
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info()
                print(
                    f"[AL] round {rnd}: freed torch GPU cache before docking -> "
                    f"{free // 1024 // 1024} / {total // 1024 // 1024} MiB free",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - memory hygiene must not crash the loop
            print(f"[AL] round {rnd}: WARNING could not free torch GPU cache: {exc}", flush=True)

    def _score_batch(self, batch: List[str]) -> tuple:
        """Score the query batch, returning ``(scores, details)``.

        Uses the oracle's ``score_detailed()`` (per-molecule breakdown dicts) when
        available so the suggestion log can record vina_t2/vina_t1/cnnsc/etc.;
        otherwise falls back to the scalar ``score()`` and emits no details."""
        if not batch:
            return [], None
        score_detailed = getattr(self.oracle, "score_detailed", None)
        if callable(score_detailed):
            details = score_detailed(batch)
            scores = [d.get("dvina", float("nan")) for d in details]
            return scores, details
        return self.oracle.score(batch), None

    def _reset_replay_buffer(self) -> None:
        """Best-effort clear of the replay buffer between rounds (priorities go
        stale once M is refit). Guarded against upstream layout changes."""
        rb = getattr(self.trainer, "train_replay_buffer", None)
        if rb is None:
            return
        if hasattr(rb, "states_list") and hasattr(rb, "states_set"):
            rb.states_list = []
            rb.states_set = set()
            if hasattr(rb, "proxy_value_array"):
                rb.proxy_value_array[:] = 0.0

    @staticmethod
    def _write_top_k(path: Path, rows: List[tuple]) -> None:
        import csv

        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["rank", "smiles", "oracle_label"])
            for rank, (smi, label) in enumerate(rows, start=1):
                writer.writerow([rank, smi, label])
