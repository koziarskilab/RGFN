# validation/

**The comparative-evaluation world, in one place.** Everything here exists to
*measure* our pipeline against alternatives — it is never part of the pipeline
itself. Baseline generators, benchmark suites, the runner, high-fidelity
scorers, and committed results all live under this one umbrella.

```
validation/
├── generators/   # entrants — baseline generators (thin adapters)
│   ├── rgfn/         #   adapter → our glue/ pipeline (so the entrant can't drift)
│   ├── synflownet/   #   reaction-based GFlowNet baseline
│   ├── fraggfn/      #   fragment-based GFlowNet (non-synthesizable)
│   └── vae_bo/       #   VAE + Bayesian optimization
├── oracles/      # validation-only scorers (e.g. Boltz-2 co-folding) — never in-loop
│   └── boltz2/
├── suites/       # benchmark tasks/metrics
│   ├── pmo/          #   Practical Molecular Optimization (external standard)
│   └── glue_suite/   #   OUR own glue benchmarks (run on every entrant)
├── harness/      # the runner that drives entrants through suites + eval metrics
├── configs/      # run specs (which entrants × suite × oracle × budget × seeds)
└── results/      # committed result tables + plots (small artifacts only)
```

## Why this exists

Reviewers will ask (`docs/RESEARCH_CONTEXT.md`, Objectives 4–5) whether RGFN
beats random sampling **and** a non-synthesizable generator on the same oracle
and budget, whether the oracle generalizes across systems, and whether the
neosubstrate differential specifically matters (ablation vs. Tier-2 absolute).
A key target plot is the analogue of `[bengio2021gflownet]` Fig. 7 — top-k
quality vs. oracle calls, GFlowNet vs. random acquisition. This directory is
where every such comparison is produced.

---

## The boundary rule (read this before adding anything)

The repo already separates **upstream (`rgfn/`) vs. ours (`glue/`, …)**. This
directory adds a *second* axis: **production pipeline vs. validation**. Keep it
clean with one rule:

> **The dependency arrow points one way.** `validation/` may import from `glue/`
> and `rgfn/`. The production pipeline — `glue/`, `scripts/train.py`,
> `configs/glue/` — must **never** import from `validation/`.

Why this matters: it guarantees you can never accidentally wire a slow
validation-only oracle (Boltz-2, co-folding, MD) into the in-loop reward, and it
keeps the thing we ship (`glue/`) understandable without dragging in every
baseline and benchmark. If you find yourself wanting `glue/` to import something
here, the abstraction lives in the wrong place — push it down into `glue/`.

See `docs/ARCHITECTURE.md` for the full two-axis layout and the rationale.

---

## What goes where (decision guide)

| You're adding… | It goes in… |
|---|---|
| A baseline generator to compare against | `validation/generators/<name>/` (thin adapter) |
| Third-party install for that baseline | `external/setup_<name>.sh` (heavy code is **not** vendored) |
| A scorer used only to check top-k (Boltz-2, co-folding) | `validation/oracles/<name>/` |
| A benchmark task/metric set (PMO, our own) | `validation/suites/<name>/` |
| The code that runs entrants through a suite | `validation/harness/` |
| A reusable metric the *pipeline* also needs (e.g. scaffold counts) | `glue/metrics/` (not here) |
| The in-loop reward oracle/proxy | **not here** → `glue/oracles/`, `glue/proxies/` |

Conventions:
- Heavy raw outputs stay in gitignored scratch; commit only summarized tables and
  figures to `results/`.
- Each comparison is reproducible from a `configs/` spec + a documented command;
  significant runs get an entry in `Logs/` (use the `experiment-log` skill).

> **Status: scaffolding only.** These are placeholder homes (`.gitkeep` /
> README) — no implementations have been imported yet. When code lands, each
> Python subtree gets an `__init__.py` so the harness can import it.
