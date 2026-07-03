# `glue/analysis/` — trained-GFN analysis: hub-based late-stage diversification

Post-hoc analysis of a **trained** reaction-GFN. The first (and current) capability
answers one question:

> Instead of sampling `N` molecules independently, can we pick the `m` highest-value
> **pre-terminal hubs** (shared reaction intermediates) and synthesize `m` large,
> high-quality batches **in parallel** — each batch a single *diverging final reaction*
> off its shared scaffold (late-stage diversification)?

and gives you the infrastructure to trade **diversity** against **concurrency** (number
of parallel lanes) or **cost** (shared-intermediate synthesis) — i.e. everything needed
to draw a Pareto front, without committing to one yet.

This package **imports from `glue/` and `rgfn/` and is never imported by them.** It is
analysis, not shipped in-loop pipeline, and is *not* gin-configurable — it is driven by
`scripts/analyze_gfn.py` and a Python/JSON sweep spec. (That's why `glue.registry` does
not import it.)

## Why a `ReactionStateA` is a hub

A reaction-GFN builds a molecule as a chain of `ReactionStateA` intermediates
(`rgfn/gfns/reaction_gfn/`):

```
S0 → SA(bb,0) → [SB..SC] → SA(m1,1) → … → SA(m_{k-1},k-1) → [SB..SC] → SA(x,k) → Terminal(x)
```

At every `SA` the forward policy either **STOPs** (emit the molecule) or applies **one
more reaction**. So each `SA` intermediate is a *hub*: a shared scaffold from which one
diverging final reaction yields many terminal analogs. `SA(m_{k-1}, k-1)` — the
penultimate intermediate — is the natural late-stage-diversification point for terminal
`x`. Making that intermediate once and splitting it into parallel final steps is exactly
what a chemist would batch.

## Flow

The configs train with **Trajectory Balance**, which parameterizes `logZ + P_F + P_B`
but **not** per-state flows `F(s)`. So a hub's flow is estimated by its **forward-sampling
visit count** — an unbiased Monte-Carlo estimate of `F(hub)/Z`, naturally reward-weighted
because the trained policy samples ∝ reward. (A `ModelStateFlow` reader for SubTB-trained
models can be added later; the exact per-child allocation `P_F(child|hub)` is also
available from the forward policy.)

## The pipeline (each stage is one pluggable piece)

| stage | file | what |
|---|---|---|
| load | `loader.py` | `TrainedGFN.load(cfg, ckpt)` → env, forward policy, proxy, sampler |
| sample | `loader.py` | `sample_trajectories(n)` — forward samples with rewards attached |
| hub graph | `hub_graph.py` | `build_hub_graph(traj)` — intermediates, **flow**, observed children, routes |
| expand children | `expanders.py` | `observed` (cheap, sampled) or `enumerative` (exhaustive, env-driven, proxy-scored) |
| pick hubs | `hub_selectors.py` | `highest_flow`, `most_modes`, `highest_expected_reward`, `highest_child_reward` (+ `DiverseHubSelector`) |
| pick molecules | `mol_selectors.py` | `top_k_reward`, `top_k_reward_diverse`, `scaffold_diverse_k`, `random_k` |
| assemble | `batch_plan.py` | `BatchPlan` = `m` lanes → standard `CandidateDataset` + `lanes.csv` |
| measure | `metrics.py` | diversity / concurrency / cost / reward per plan |
| sweep | `sweep.py` | trials × strategies → tidy `results.csv` |
| front | `pareto.py` | `pareto_front(rows, objectives)` |
| plot | `plot.py` | Pareto scatter + frontier PNGs (auto-written by the sweep) |

## Run it

```bash
python scripts/analyze_gfn.py \
  --cfg configs/glue/fixed_reward_seh_proxy_stdlib.gin \
  --checkpoint_path <run>/train/checkpoints/best_gfn.pt \
  --library data/libraries/glue_standard_v1 \
  --hub_selectors highest_flow,most_modes,highest_expected_reward \
  --mol_selectors top_k_reward,top_k_reward_diverse \
  --m_values 4,8,16 --k_values 10,25 --n_trials 3 --sample_size 3000 \
  --out experiments/diversification/<run>/<timestamp>
```

Outputs: `results.csv` (one row per `hub×mol×m×k×trial`, the Pareto raw material),
`manifest.json` (provenance), **`diversity_vs_concurrency.png` + `diversity_vs_cost.png`**
(the Pareto-front plots, written automatically; `--no_plots` to skip), and — with
`--write_plans` — each plan as a candidate dataset under `plans/`. A JSON spec
(`--spec spec.json`, keys of `SweepSpec`) replaces the grid flags for richer sweeps
(per-strategy kwargs, enumerative expander, etc.).

## Extending it

Add a strategy = subclass the base + one line in `registry.py`:

```python
# hub_selectors.py
class MyHubSelector(HubSelector):
    name = "my_hub"
    def score(self, hub, graph): return ...   # higher == preferred
# registry.py
HUB_SELECTORS[MyHubSelector.name] = MyHubSelector
```

…then reference it by name (`--hub_selectors my_hub`). Same shape for `MoleculeSelector`
(`MOL_SELECTORS`) and `ChildExpander` (`EXPANDERS`).

## Cost model (the cost axis)

`cost.py` prices a route from a `ChemLibrary`'s block prices + reaction yields
(`docs/CHEM_LIBRARY_FORMAT.md`). A plan's cost is reported two ways so the saving is
explicit: **hub-batched** (route-to-hub paid once per lane + each product's final step)
vs **independent** (every molecule from scratch). It is a small, documented first cut —
swap `cost.py` for a different cost story without touching anything else.

## The Pareto front

The sweep writes `diversity_vs_concurrency.png` and `diversity_vs_cost.png` automatically
(scatter of every configuration, colored by hub selector / shaped by molecule selector,
with the non-dominated frontier drawn through it and trial error bars). To recompute or
draw a different pair of axes yourself:

```python
import csv
from glue.analysis import pareto_front, aggregate_trials, plot_pareto
rows = list(csv.DictReader(open("results.csv")))
agg  = aggregate_trials(rows, ["hub_selector","mol_selector","m","k"],
                        ["internal_diversity","concurrency","cost_per_molecule","reward_mean"])
front = pareto_front(agg, [("internal_diversity_mean","max"), ("concurrency_mean","min")])
plot_pareto(agg, x_col="cost_per_molecule_mean", y_col="internal_diversity_mean",
            out_path="div_vs_cost.png", minimize_x=True, maximize_y=True,
            x_err="cost_per_molecule_std", y_err="internal_diversity_std")
```

(Note: after `aggregate_trials`, columns are renamed `X` → `X_mean`/`X_std`; `pareto_front`
raises if an objective column is absent, so a typo fails loudly instead of returning an
empty front.)
