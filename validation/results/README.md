# validation/results/

Committed benchmark result tables (small artifacts only — the raw per-molecule datasets
and run outputs live under `$SCRATCH`/`data/synthetic/`, git-ignored).

## Synthesizability comparison

Each entrant's candidate dataset is scored by `validation/harness/synthesizability.py`
(AiZynthFinder route-finding + RDKit SA), which writes a `synthesizability_summary.json`
next to it. `validation/harness/aggregate_synthesizability.py` gathers those per-generator
summaries into one comparison table.

### `seh_fixed_reward/`

The matched four-way **fixed-reward** benchmark on the sEH target: RGFN, FragGFN, RxnFlow,
and SCENT each trained **once** against the *same* frozen Bengio-2021 sEH proxy (no active
learning, no oracle — see `glue/fixed_reward/` and the `*_seh_fixed` configs), then scored
for synthesizability.

- `comparison.csv` / `comparison.md` — the assembled table.
- Columns: `aizynth_success` (headline: fraction of unique valid molecules with a full
  retrosynthetic route to in-stock precursors), `steps_mean` (route length), `sa_mean`
  (RDKit synthetic accessibility), `self_route_rate` (the generator's by-construction route
  claim — RGFN/RxnFlow/SCENT = 1.0, FragGFN = 0), and `mol_weight_mean`/`qed_mean`
  (drug-likeness).

Reproduce (after the four fixed-reward runs + per-dataset AiZynth scoring):

    python validation/harness/aggregate_synthesizability.py \
        --dataset rgfn=<run>/rgfn/candidates \
        --dataset fraggfn=<run>/fraggfn/candidates \
        --dataset rxnflow=<run>/rxnflow/candidates \
        --dataset scent=<run>/scent/candidates \
        --out validation/results/seh_fixed_reward/comparison.csv \
        --out-md validation/results/seh_fixed_reward/comparison.md \
        --title "Matched four-way fixed-reward sEH benchmark"
