# validation/harness/

The reusable **runner** that drives entrants through a suite, plus the
**evaluation metrics** used only for benchmarking.

Responsibilities (when implemented):

- read a spec from `validation/configs/`,
- load the named entrants from `validation/generators/` and run each through a
  `validation/suites/` suite at an **equal oracle-call budget** (so RGFN and the
  baselines are compared on equal compute),
- optionally score the resulting top-k with a `validation/oracles/` high-fidelity
  scorer (e.g. Boltz-2) for the anti-gaming check,
- write summarized tables/plots to `validation/results/`.

Metric placement: metrics used *only* for evaluation (diversity, SA distribution,
recovery/enrichment, anti-gaming correlations) live here. A metric the
**pipeline** also needs (e.g. scaffold counts during training) belongs in
`glue/metrics/` and is imported from there — never the reverse.

## Implemented metrics

### `synthesizability.py` — AiZynthFinder + SA score (post-hoc)

The synthesizability report every entrant gets, computed **over** a finished
candidate dataset (`docs/CANDIDATE_DATASET_FORMAT.md`) — never written into it,
never in the training loop. It is the analogue of the `AiZynth` column in
`[koziarski2024rgfn]` Table 1 / `[gainski2025scent]` Table 1 and RxnFlow's
"Synthesizability %" (`[seo2024rxnflow]`). Per unique valid molecule it records
whether AiZynthFinder finds a full retrosynthetic route to in-stock building
blocks (USPTO templates + ZINC stock), plus the route length and the RDKit
SA score; aggregated it reports the **AiZynth success rate**, mean #steps, and
the SA distribution, and cross-checks against the generator's *self-reported*
`has_route`/`num_reactions` (RGFN/RxnFlow's by-construction claim).

Because AiZynthFinder pins its own heavy stack, it lives in a **dedicated
`aizynth` conda env** (`external/setup_aizynthfinder.sh`); the evaluator reads the
candidate CSV/JSON directly so it never imports `glue` (no torch/dgl in that env)
— the same env-boundary discipline as the `scripts/score_batch.py` oracle bridge.

```bash
bash external/setup_aizynthfinder.sh           # one-time: env + public dataset
conda run -n aizynth python validation/harness/synthesizability.py \
    --dataset data/synthetic/<run>/candidates \
    --config  data/models/aizynthfinder/config.yml \
    --nproc 8                                   # add --top-k 500 to score only the best
# -> writes synthesizability.csv + synthesizability_summary.json next to the dataset
```

Run it on every entrant's candidate dataset (RGFN, RxnFlow, FragGFN, …) to get
the cross-generator synthesizability table. Orchestration logic is unit-tested
dependency-free in `test_synthesizability.py` (`python -m unittest
validation.harness.test_synthesizability`).

> Runner: scaffolding only — no end-to-end suite driver yet; the synthesizability
> metric above is the first evaluation metric to land.
