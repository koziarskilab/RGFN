# sEH / RGFN — hub-based late-stage-diversification analysis pipeline

**Date:** 2026-07-03, ~2pm

### Question

Can we turn a trained molecule-generating model into a *plan* for making many diverse
molecules in parallel from shared intermediates, and measure the trade-off between how
diverse the batch is and how many parallel synthesis lanes (or how much cost) it takes?

### Context & Summary

RGFN samples molecules one at a time, but it builds each one through a chain of shared
reaction intermediates — so from a common intermediate a single final reaction can branch
into many analogs at once (what chemists call late-stage diversification: make the shared
scaffold once, split it into parallel final steps). We had no way to look inside a *trained*
model and (a) find which shared intermediates carry the most of the model's "traffic," (b)
pick molecules off them under different strategies, and (c) measure the resulting diversity
against the number of parallel lanes or the synthesis cost. Entries 019–021 gave us trained
GFNs and a standard candidate-dataset format; this entry builds the analysis layer on top of
them.

We built a new, modular analysis package (`glue/analysis/`) that loads a trained GFN, samples
trajectories, and assembles a "hub graph" of intermediates — each hub scored by its **flow**
(how often the trained policy passes through it). Pluggable strategies then choose `m` hubs
(e.g. highest-flow, or the ones with the most distinct products one reaction away) and `k`
molecules per hub (e.g. top-reward, or top-reward spread out by chemical dissimilarity). The
selected molecules become `m` parallel "lanes"; we measure the batch's diversity, its
concurrency (number of lanes), its synthesis cost (shared intermediate made once, then cheap
final steps), and its reward. A sweep runs many strategies over several random-seed trials and
writes a tidy table plus the two Pareto-front plots. We validated the whole thing end-to-end
on a real trained checkpoint — the fixed-reward sEH-proxy model on the shared standard library.

### Answer

The infrastructure works end-to-end on a real trained model and produces the intended Pareto
fronts. As expected, diversity rises with the number of parallel lanes; the diversity-aware
molecule selector beats plain top-reward at every concurrency level; and — the most useful
finding the tool surfaces — making more products per shared intermediate *lowers* the cost per
molecule, so the highest-concurrency configuration was simultaneously the cheapest per molecule
(hub-batching saved 33–50% over synthesizing each molecule independently). This is scaffolding
that makes a future diversity-vs-cost result one sweep away; it is a tooling/methods result,
not yet a scientific claim about glues.

### Relevance to our Publication

This is a methods-and-tooling contribution aimed at the primary targets (Digital Discovery,
Journal of Cheminformatics), where a reusable, reproducible analysis is itself part of the
story. It directly strengthens the project's differentiator — that RGFN's synthesizability is
"doing real work" (Objective 5/6): a trained *synthesizable* generator can be turned into a
cost- and parallelism-aware library-design plan, which a non-synthesizable baseline simply
cannot produce (it has no routes to share). The diversity-vs-cost Pareto front is a concrete,
reviewer-legible figure that quantifies that advantage.

### Next Experiments

**Refining for publication**
- Run the same analysis on the validated **6TD3 glue** GFN (entries 014/021), not just the sEH
  proxy, so the front is about real glue candidates.
- Larger `m`/`k` grids and ≥3 seeds for tight error bars; report the front with variance.
- Add an **observed-vs-enumerative** ablation (does exhaustive enumeration of a hub's products
  change the front vs. only what the policy sampled?) and a real per-block catalog cost model.

**Next steps in project**
- Feed the selected diversified libraries into the AiZynthFinder synthesizability evaluator
  (entry 018) and the docking oracle, to score the actual proposed batches.
- Add a model-based flow readout for SubTB-trained models and an inter-hub diversity constraint,
  then produce the headline diversity-vs-concurrency/-cost front for the paper.

# Re-creation

### Relevant Files

Root: `./` (repository root). Analysis package committed in `9dd7b37`.

**Scripts**
- `scripts/analyze_gfn.py` — entry point: loads a trained GFN + config, runs a strategy sweep,
  writes `results.csv` + the two Pareto PNGs. Mirrors `scripts/infer.py` (imports `glue` first).
- `glue/analysis/loader.py` — `TrainedGFN.load(cfg, ckpt)`: rebuilds the objective/valid-sampler
  from gin (as `infer.py` does), `strict=False` to tolerate sampling-time cache buffers.
- `glue/analysis/hub_graph.py` — `build_hub_graph(traj)`: reduces trajectories to hubs
  (`ReactionStateA` intermediates); flow = trajectory visit count; observed 1-reaction children
  off the penultimate intermediate.
- `glue/analysis/expanders.py` — `ObservedExpander` (sampled children) / `EnumerativeExpander`
  (env-driven exhaustive enumeration of a hub's 1-reaction products, proxy-scored, per-hub cached).
- `glue/analysis/hub_selectors.py` — `highest_flow`, `highest_tb_flow`, `most_modes`,
  `highest_expected_reward`, `highest_child_reward` (+ `DiverseHubSelector`).
- `glue/analysis/tb_flow.py` — balance-based flow: `annotate_tb_flow` computes
  `F(h)=R(x)P_B(h|x)/P_F(x|h)` (`[malkin2022trajectorybalance]`) per hub from the per-step
  `logP_F`/`logP_B` + shaped `logR(x)`; `flow_agreement` correlates it against visit-count flow
  (sampling-vs-balance training-quality diagnostic).
- `glue/analysis/mol_selectors.py` — `top_k_reward`, `top_k_reward_diverse` (Tanimoto MaxMin),
  `scaffold_diverse_k`, `random_k`.
- `glue/analysis/batch_plan.py` — `BatchPlan`/`Lane` → standard `CandidateDataset` + `lanes.csv`.
- `glue/analysis/metrics.py` — per-plan diversity/concurrency/cost/reward (reuses
  `glue.metrics.dataset_metrics` + `glue.chemistry` cost).
- `glue/analysis/cost.py` — route pricing from a `ChemLibrary` (hub-batched vs independent).
- `glue/analysis/select.py` / `sweep.py` — one selection / the trials×strategies sweep.
- `glue/analysis/pareto.py` — Pareto-front extraction + trial aggregation.
- `glue/analysis/plot.py` — `plot_pareto` / `plot_sweep_fronts` (Okabe–Ito palette, marker
  secondary encoding, frontier line, trial error bars) + `plot_flow_agreement` (sampling-vs-TB
  scatter with the `y=x` reference).
- `glue/analysis/registry.py` — name→class maps (add a strategy = subclass + one line).

**Models**
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/seh_proxy_stdlib/2026-07-02_14-18-34/train/checkpoints/best_gfn.pt`
  — the trained GFN analyzed (fixed-reward sEH-proxy run on `glue_standard_v1`, entry 020). An
  idle snapshot, chosen to avoid an actively-training run's directory.

**Datasets**
- `./data/libraries/glue_standard_v1/` — the priced standard library (418 fragments / 112
  reactions, SCENT SMALL set with costs+yields); supplies both the GFN's action space and the
  cost model.

**Results** (git-ignored timestamped experiment output — viewable on disk)
- `./experiments/diversification/seh_stdlib/2026-07-03_14-33-08/`
  — `results.csv` (72 rows), `manifest.json`, **`diversity_vs_concurrency.png`**,
  **`diversity_vs_cost.png`** (the showcase enumerative sweep).
- `./experiments/diversification/seh_stdlib/spec.json` — example richer sweep spec (committed).

**Job Logs**
- Login-node validation runs (no SLURM job); this was an interactive smoke/validation session on
  the Balam login-node A100. A batch submit exists at
  `./experiments/diversification/seh_stdlib/submit_analyze_seh_stdlib.sh`.

### Relevant Versions

```
9dd7b37 Big standard lib run w/ 400-iters, pareto front pipeline
```

The `glue/analysis/` pipeline, `scripts/analyze_gfn.py`, `experiments/diversification/`, and the
`docs/REFACTOR_LOG.md` entry are committed in **`9dd7b37`**. This log file
(`Logs/022_*.md`) and the timestamped plot outputs (git-ignored by pattern) are **not yet
committed** — `[TODO — add commit hash after committing this log]`. Can you commit the log entry?
Once you do, let me know and I'll update this line.

### Relevant Resources

**Sources**
- `[koziarski2024rgfn]` — RGFN (reaction-based GFlowNet); the state machine whose `ReactionStateA`
  intermediates are the hubs.
- `[bengio2021gflownet]` — the original GFlowNet (flow, `F(s)`, `P_F`/`P_B`, `Z`).
- `[malkin2022trajectorybalance]` — Trajectory Balance, the objective RGFN optimizes; it learns
  `logZ + P_F + P_B` but *not* per-state flow `F(s)` (hence flow-by-visit-count), and its balance
  condition `F(h)=R(x)P_B(h|x)/P_F(x|h)` is the basis for the `tb_flow` estimator added in this work.
- Okabe & Ito colorblind-safe categorical palette (used verbatim for the plots).

**Packages**
- `matplotlib` 3.10.9 — `glue/analysis/plot.py` (Agg backend).
- `rdkit` — fingerprints/Tanimoto/scaffolds in `glue/analysis/similarity.py` +
  `glue/metrics/dataset_metrics.py`.
- `torch` 2.3.0+cu118, `gin` — checkpoint load + config in `glue/analysis/loader.py`.

### Method

All runs used the `rgfn` conda env via `~/bin/rgfn-smoke-env.sh` on the login-node A100.

1. **Pure-unit checks** — `py_compile` all modules; imported `glue.analysis`; verified
   `pareto_front` dominance, Tanimoto, and mode-counting on toy inputs.
2. **Observed-expander sweep** (validation): `scripts/analyze_gfn.py` on the checkpoint above,
   `--hub_selectors highest_flow,most_modes,highest_expected_reward`,
   `--mol_selectors top_k_reward,top_k_reward_diverse`, `--m_values 4,8 --k_values 5,10`,
   `--n_trials 2 --sample_size 500 --library data/libraries/glue_standard_v1 --write_plans`.
   → 48-row `results.csv`; both Pareto fronts extracted.
3. **Enumerative-expander sweep** (showcase): same script,
   `--hub_selectors highest_flow,most_modes --mol_selectors top_k_reward,top_k_reward_diverse`,
   `--m_values 2,4,8 --k_values 5,15,30 --n_trials 2 --sample_size 600`,
   `--expander enumerative --expander_kwargs '{"max_fragments_per_pattern":25,"max_children":300}'`.
   → 72-row `results.csv` + `diversity_vs_concurrency.png` + `diversity_vs_cost.png` (auto-written).

### Results

**Hub graph (observed sweep, sEH stdlib checkpoint).** Per 500-trajectory sample: ~477 terminal
molecules, ~1870 distinct hubs, ~475 observed 1-reaction children (trials: 481/474 terminals,
1855/1889 hubs).

**Enumeration reach.** Single hubs enumerated to **29 / 87 / 105 / 106** distinct 1-reaction
products (capped at `max_children=300`, cap announced). This is what the observed expander misses:
the policy typically sampled ~1 child per hub.

**Diversity-aware selection.** On an enumerated hub set (`k`≈25), `top_k_reward_diverse` gave
**all-distinct** products (modes 60/60; 75/75 in the smoke) vs `top_k_reward` (48–50 modes);
`scaffold_diverse_k` gave the highest within-lane diversity (0.541).

**Diversity vs concurrency Pareto front** (enumerative sweep, 4 of 36 configs non-dominated,
mean over 2 trials):

| config | internal diversity | concurrency (lanes) | # modes | cost / molecule |
|---|---|---|---|---|
| m2·k5 most_modes / top_k_reward_diverse | 0.700 | 2.0 | 10 | 23.8 |
| m4·k30 highest_flow / top_k_reward_diverse | 0.717 | 4.0 | 96 | 10.0 |
| m8·k30 most_modes / top_k_reward_diverse | 0.741 | 5.5 | 106 | 11.2 |
| **m8·k30 highest_flow / top_k_reward_diverse** | **0.816** | 8.0 | **213** | **7.25** |

**Cost (hub-batched vs independent synthesis).** Hub-batching — making each lane's shared
intermediate once, then only the cheap final step per product — saved **26–33%** (observed,
sparse children) up to **~50%** (enumerative, k=25/lane) versus synthesizing every molecule from
scratch. Because more products per shared intermediate amortize its cost, the highest-concurrency
config (m8·k30) had the **lowest** cost per molecule (7.25), i.e. it is Pareto-optimal on both
diversity and cost.
