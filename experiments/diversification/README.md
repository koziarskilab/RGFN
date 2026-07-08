# experiments/diversification/ — hub-based late-stage-diversification runs

Runs that use `glue/analysis/` (hub-based late-stage diversification) to characterize a
**trained** GFN and build **diversity-vs-concurrency / -cost** Pareto fronts. One
self-contained dir per run (per `experiments/README.md`): its spec + submit + analysis,
its small committed results, and its git-ignored timestamped outputs.

The reusable machinery lives in `glue/analysis/` (it graduates into `glue/` because it is
pipeline-wide); this tree holds only the one-off harness around a particular run — the
sweep spec, the submit script pointing at a specific checkpoint, and any bespoke plots.

## What a run produces

`scripts/analyze_gfn.py` → a timestamped dir with:
- `results.csv` — one row per `(hub selector × molecule selector × m × k × trial)` with
  diversity / concurrency / cost / reward columns. **This is the Pareto raw material.**
- `manifest.json` — config, checkpoint, spec, provenance.
- `plans/…` (with `--write_plans`) — each configuration's selected molecules as a
  standard candidate dataset (+ `lanes.csv`), readable by the benchmark harness.

## Layout

```
diversification/
├── README.md            # this file
└── seh_stdlib/          # example run group: sEH-proxy GFN on the priced standard library
    ├── spec.json        # a richer sweep spec (SweepSpec keys)
    └── submit_analyze_seh_stdlib.sh   # Balam submit pointing at a trained checkpoint
```

## Notes

- Cost axis needs a **priced** library (`--library data/libraries/glue_standard_v1`);
  the `glue_standard_v1` set carries SCENT's prices/yields.
- The `enumerative` expander is exhaustive but expensive (many proxy calls per hub) —
  fine for the cheap sEH MPNN proxy; use `observed` for a sweep and enumerate only the
  final chosen configurations, or with an expensive in-loop docking proxy.
- Multiple trials (`--n_trials`) re-sample with different seeds → error bars on the front.
