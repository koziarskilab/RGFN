#!/usr/bin/env python
"""Generic entry point: analyze a trained reaction-GFN for hub-based diversification.

Mirrors ``scripts/infer.py`` — imports ``glue`` first so gin can resolve any ``glue``
components the training config references, then loads the checkpoint and runs a sweep of
hub/molecule strategies into a tidy ``results.csv`` (the raw material for a
diversity-vs-concurrency / -cost Pareto front). See ``glue/analysis/README.md``.

Two ways to specify the sweep:
  * a JSON spec file (``--spec spec.json``, keys of ``glue.analysis.SweepSpec``), or
  * CLI flags for the common grid (``--hub_selectors``, ``--mol_selectors``,
    ``--m_values``, ``--k_values``, ``--expander``, ...).

Example (real trained sEH-proxy checkpoint on the priced standard library):

    python scripts/analyze_gfn.py \
      --cfg configs/glue/fixed_reward_seh_proxy_stdlib.gin \
      --checkpoint_path <run>/train/checkpoints/best_gfn.pt \
      --library data/libraries/glue_standard_v1 \
      --hub_selectors highest_flow,most_modes,highest_expected_reward \
      --mol_selectors top_k_reward,top_k_reward_diverse \
      --m_values 4,8,16 --k_values 10,25 --n_trials 3 --sample_size 3000 \
      --out experiments/diversification/seh_stdlib/<timestamp>
"""

import argparse
import json
from pathlib import Path

import glue  # noqa: F401  (side effect: registers our gin components before config parse)
from gin_config import get_time_stamp
from glue.analysis import SweepRunner, SweepSpec, TrainedGFN


def _csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _ints(s):
    return [int(x) for x in _csv(s)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg", required=True, help="gin config the run was trained with")
    p.add_argument("--checkpoint_path", required=True, help="train/checkpoints/*.pt")
    p.add_argument("--checkpoint_key", default="model")
    p.add_argument("--out", default=None, help="output dir (default: analysis_<cfg>_<ts>)")
    p.add_argument("--spec", default=None, help="JSON sweep spec (overrides the grid flags)")
    # grid flags (used when --spec is absent)
    p.add_argument("--hub_selectors", default="highest_flow")
    p.add_argument("--mol_selectors", default="top_k_reward")
    p.add_argument("--m_values", default="8")
    p.add_argument("--k_values", default="20")
    p.add_argument("--expander", default="observed", help="observed | enumerative")
    p.add_argument("--expander_kwargs", default=None, help="JSON kwargs for the expander")
    p.add_argument("--n_trials", type=int, default=3)
    p.add_argument("--sample_size", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--library", default=None, help="priced ChemLibrary dir for the cost axis")
    p.add_argument("--per_reaction_cost", type=float, default=1.0)
    p.add_argument("--yield_adjusted", action="store_true")
    p.add_argument("--write_plans", action="store_true", help="also dump each plan's dataset")
    p.add_argument("--no_plots", action="store_true", help="skip the Pareto-front PNGs")
    p.add_argument(
        "--compute_tb_flow",
        action="store_true",
        help="also annotate hubs with the balance-based TB flow + write the agreement diagnostic "
        "(auto-enabled if a hub selector is 'highest_tb_flow')",
    )
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg_name = Path(args.cfg).stem
    out_dir = Path(args.out) if args.out else Path(f"analysis_{cfg_name}_{get_time_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.spec:
        with open(args.spec) as fh:
            spec = SweepSpec.from_dict(json.load(fh))
        if args.no_plots:
            spec.make_plots = False
        if args.compute_tb_flow:
            spec.compute_tb_flow = True
    else:
        expander = (args.expander, json.loads(args.expander_kwargs) if args.expander_kwargs else {})
        spec = SweepSpec(
            hub_selectors=[(n, {}) for n in _csv(args.hub_selectors)],
            mol_selectors=[(n, {}) for n in _csv(args.mol_selectors)],
            m_values=_ints(args.m_values),
            k_values=_ints(args.k_values),
            expander=expander,
            n_trials=args.n_trials,
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            base_seed=args.base_seed,
            library_dir=args.library,
            per_reaction_cost=args.per_reaction_cost,
            yield_adjusted=args.yield_adjusted,
            write_plans=args.write_plans,
            make_plots=not args.no_plots,
            compute_tb_flow=args.compute_tb_flow,
        )

    print(f"[analyze_gfn] loading {args.checkpoint_path}\n  with {args.cfg}", flush=True)
    gfn = TrainedGFN.load(
        args.cfg,
        args.checkpoint_path,
        checkpoint_key=args.checkpoint_key,
        device=args.device,
        run_name=f"{cfg_name}/analysis",
    )
    print(
        f"[analyze_gfn] loaded on {gfn.device}; proxy={type(gfn.proxy).__name__} "
        f"higher_is_better={gfn.higher_is_better}",
        flush=True,
    )
    SweepRunner(gfn, spec, out_dir).run()


if __name__ == "__main__":
    main()
