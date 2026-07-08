"""``SweepRunner`` — run many strategy configurations over many trials → a tidy table.

This is the experiment harness. Given a loaded model and a :class:`SweepSpec` (lists of
hub strategies, molecule strategies, ``m`` values, ``k`` values, and a trial count), it:

  * for each **trial** (a distinct seed): forward-samples once and builds one hub graph
    (so every strategy in that trial is compared on the *same* sampled structure, and
    variance across trials is real sampling variance);
  * for each ``(hub strategy × mol strategy × m × k)`` in the trial: runs
    :func:`~glue.analysis.select.run_selection` reusing that graph, and records one row
    of metrics tagged with the configuration + trial;
  * writes ``results.csv`` (one row per config per trial — the Pareto raw material) and
    ``manifest.json`` (provenance), and optionally each plan's candidate dataset.

Enumeration, if requested, is the expensive part and runs per selected hub per config;
keep the grid small when using it, or use the observed expander for the sweep and
enumerate only the final chosen configurations.
"""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from glue.analysis import registry
from glue.analysis.hub_graph import build_hub_graph
from glue.analysis.select import run_selection
from glue.chemistry.library import ChemLibrary

# A strategy entry is (registry_name, kwargs_dict).
StrategyEntry = Tuple[str, Dict[str, Any]]


@dataclass
class SweepSpec:
    """Everything a sweep varies + the fixed sampling/cost settings."""

    hub_selectors: List[StrategyEntry]
    mol_selectors: List[StrategyEntry]
    m_values: List[int]
    k_values: List[int]
    expander: StrategyEntry = ("observed", {})
    n_trials: int = 3
    sample_size: int = 2000
    batch_size: int = 200
    base_seed: int = 42
    library_dir: Optional[str] = None
    per_reaction_cost: float = 1.0
    yield_adjusted: bool = False
    write_plans: bool = False
    make_plots: bool = True
    compute_tb_flow: bool = False  # annotate hubs with balance-based flow (auto-on if used)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SweepSpec":
        """Build from a JSON/dict spec, normalising strategy entries to ``(name, kwargs)``."""

        def norm(entries) -> List[StrategyEntry]:
            out = []
            for e in entries:
                if isinstance(e, str):
                    out.append((e, {}))
                elif isinstance(e, (list, tuple)):
                    out.append((e[0], dict(e[1]) if len(e) > 1 and e[1] else {}))
                elif isinstance(e, dict):
                    out.append((e["name"], dict(e.get("kwargs", {}))))
                else:
                    raise ValueError(f"bad strategy entry: {e!r}")
            return out

        exp = d.get("expander", ("observed", {}))
        if isinstance(exp, str):
            exp = (exp, {})
        elif isinstance(exp, dict):
            exp = (exp["name"], dict(exp.get("kwargs", {})))
        return cls(
            hub_selectors=norm(d["hub_selectors"]),
            mol_selectors=norm(d["mol_selectors"]),
            m_values=list(d["m_values"]),
            k_values=list(d["k_values"]),
            expander=tuple(exp),
            n_trials=int(d.get("n_trials", 3)),
            sample_size=int(d.get("sample_size", 2000)),
            batch_size=int(d.get("batch_size", 200)),
            base_seed=int(d.get("base_seed", 42)),
            library_dir=d.get("library_dir"),
            per_reaction_cost=float(d.get("per_reaction_cost", 1.0)),
            yield_adjusted=bool(d.get("yield_adjusted", False)),
            write_plans=bool(d.get("write_plans", False)),
            make_plots=bool(d.get("make_plots", True)),
            compute_tb_flow=bool(d.get("compute_tb_flow", False)),
        )


class SweepRunner:
    def __init__(self, gfn, spec: SweepSpec, out_dir):
        self.gfn = gfn
        self.spec = spec
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.library = (
            ChemLibrary.from_canonical_dir(spec.library_dir) if spec.library_dir else None
        )

    def run(self) -> List[Dict[str, Any]]:
        spec = self.spec
        rows: List[Dict[str, Any]] = []
        combos = list(
            itertools.product(spec.hub_selectors, spec.mol_selectors, spec.m_values, spec.k_values)
        )
        # TB flow is needed if asked for, or if any selector ranks by it.
        need_tb = spec.compute_tb_flow or any(
            hs[0] == "highest_tb_flow" for hs in spec.hub_selectors
        )
        # One expander for the whole sweep: enumeration is deterministic per hub key, so
        # its cache is reused across every combo and trial (a hub selected by multiple
        # strategies is enumerated once).
        expander = registry.build_expander(spec.expander[0], spec.expander[1])
        print(
            f"[sweep] {spec.n_trials} trials × {len(combos)} configs "
            f"= {spec.n_trials * len(combos)} rows; expander={spec.expander[0]}"
            f"{'; tb_flow on' if need_tb else ''}",
            flush=True,
        )
        agreement_rows: List[Dict[str, Any]] = []
        first_agreement: Optional[Dict[str, Any]] = None
        for trial in range(spec.n_trials):
            seed = spec.base_seed + trial
            traj = self.gfn.sample_trajectories(spec.sample_size, spec.batch_size, seed=seed)
            graph = build_hub_graph(traj, self.gfn.higher_is_better)
            summary = graph.summary()
            print(
                f"[sweep] trial {trial} (seed {seed}): {summary['n_terminal']} terminals, "
                f"{summary['n_hubs']} hubs, {summary['total_observed_children']} observed children",
                flush=True,
            )
            if need_tb:
                from glue.analysis.tb_flow import annotate_tb_flow, flow_agreement

                annotate_tb_flow(graph, traj, self.gfn)
                agree = flow_agreement(graph, self.gfn)
                print(
                    f"[sweep] trial {trial}: flow agreement (visit vs TB) over {agree['n_hubs']} hubs "
                    f"— pearson(log)={agree['pearson_log']:.3f} spearman={agree['spearman']:.3f}",
                    flush=True,
                )
                agreement_rows.append(
                    {
                        "trial": trial,
                        "seed": seed,
                        "n_hubs": agree["n_hubs"],
                        "pearson_log": agree["pearson_log"],
                        "spearman": agree["spearman"],
                        "log_Z": agree["log_Z"],
                    }
                )
                if first_agreement is None:
                    first_agreement = agree
            for hs, ms, m, k in combos:
                hub_selector = registry.build_hub_selector(hs[0], hs[1])
                mol_selector = registry.build_mol_selector(ms[0], ms[1])
                plan, metrics = run_selection(
                    graph,
                    self.gfn,
                    hub_selector=hub_selector,
                    mol_selector=mol_selector,
                    m=m,
                    k=k,
                    expander=expander,
                    library=self.library,
                    per_reaction_cost=spec.per_reaction_cost,
                    yield_adjusted=spec.yield_adjusted,
                )
                row = {
                    "trial": trial,
                    "seed": seed,
                    "hub_selector": hub_selector.name,
                    "mol_selector": mol_selector.name,
                    "expander": expander.name,
                    "m": m,
                    "k": k,
                    **metrics,
                }
                rows.append(row)
                if spec.write_plans:
                    tag = f"trial{trial}/{hub_selector.name}__{mol_selector.name}__m{m}_k{k}"
                    plan.to_candidate_dataset(
                        self.out_dir / "plans" / tag,
                        oracle=str(
                            getattr(self.gfn.proxy, "__class__", type("x", (), {})).__name__
                        ),
                        higher_is_better=self.gfn.higher_is_better,
                        seed=seed,
                    )
        self._write_results(rows)
        self._write_manifest(rows, combos)
        if agreement_rows:
            self._write_agreement(agreement_rows)
        print(f"[sweep] done → {self.out_dir/'results.csv'} ({len(rows)} rows)", flush=True)
        if self.spec.make_plots:
            self._make_plots()
            if first_agreement is not None:
                self._make_flow_agreement_plot(first_agreement)
        return rows

    def _write_agreement(self, agreement_rows: List[Dict[str, Any]]) -> None:
        with open(self.out_dir / "flow_agreement.csv", "w", newline="") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["trial", "seed", "n_hubs", "pearson_log", "spearman", "log_Z"]
            )
            w.writeheader()
            w.writerows(agreement_rows)

    def _make_flow_agreement_plot(self, agreement: Dict[str, Any]) -> None:
        """Best-effort scatter of sampling flow vs TB flow (training-quality diagnostic)."""
        try:
            from glue.analysis.plot import plot_flow_agreement

            p = plot_flow_agreement(agreement, self.out_dir / "flow_agreement.png")
            print(f"[sweep] plot → {p}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[sweep] WARNING flow-agreement plot skipped: {exc}", flush=True)

    def _make_plots(self) -> None:
        """Best-effort Pareto PNGs (diversity vs concurrency / cost). Never fatal —
        matplotlib is optional and plotting must not fail an otherwise-good sweep."""
        try:
            from glue.analysis.plot import plot_sweep_fronts

            paths = plot_sweep_fronts(self.out_dir / "results.csv", self.out_dir)
            for p in paths:
                print(f"[sweep] plot → {p}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[sweep] WARNING plotting skipped: {exc}", flush=True)

    # ------------------------------------------------------------------- output
    def _write_results(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        # union of keys, stable order: tag columns first, then everything else sorted.
        head = ["trial", "seed", "hub_selector", "mol_selector", "expander", "m", "k"]
        rest = sorted({k for r in rows for k in r} - set(head))
        cols = head + rest
        with open(self.out_dir / "results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    def _write_manifest(self, rows: List[Dict[str, Any]], combos) -> None:
        manifest = {
            "config_path": self.gfn.config_path,
            "checkpoint_path": self.gfn.checkpoint_path,
            "device": self.gfn.device,
            "higher_is_better": self.gfn.higher_is_better,
            "n_rows": len(rows),
            "n_trials": self.spec.n_trials,
            "n_configs": len(combos),
            "spec": {
                "hub_selectors": self.spec.hub_selectors,
                "mol_selectors": self.spec.mol_selectors,
                "m_values": self.spec.m_values,
                "k_values": self.spec.k_values,
                "expander": list(self.spec.expander),
                "sample_size": self.spec.sample_size,
                "batch_size": self.spec.batch_size,
                "base_seed": self.spec.base_seed,
                "library_dir": self.spec.library_dir,
                "per_reaction_cost": self.spec.per_reaction_cost,
                "yield_adjusted": self.spec.yield_adjusted,
            },
        }
        with open(self.out_dir / "manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)
