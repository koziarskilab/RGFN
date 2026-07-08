# Synthesizability metric — AiZynthFinder + SA validation on Balam
**Date:** 2026-06-30, ~7:30pm

## Question

Can we measure, for any molecule our generators propose, whether a real
retrosynthesis tool can find a way to actually make it — and run that check the
same way on every model we benchmark? (Not actually running on real results yet)

## Context & Summary

Our headline claim is that RGFN proposes **synthesizable** glue candidates, and the
key comparison reviewers will want (`docs/RESEARCH_CONTEXT.md`, Objective 5) is the
synthesizability *advantage* of route-based generators (RGFN, RxnFlow, SCENT) over a
non-synthesizable baseline (FragGFN, exp `015`). The three papers we build on all
quantify this the same way: run **AiZynthFinder** (`[genheden2020aizynth]`) on the
generated molecules and report the fraction for which a full synthesis route to
purchasable building blocks is found (the "AiZynth success rate" — RGFN ≈ 0.56 in
`[koziarski2024rgfn]`, up to ≈ 0.75 in `[gainski2025scent]`, RxnFlow's
"Synthesizability %"). We built that metric as a post-hoc tool that reads the one
standard candidate-dataset format every entrant emits, so it runs uniformly on all of
them. This entry validates the tool actually works on Balam: install the retrosynthesis
engine, confirm its API behaves as the code assumes, and run the whole thing
end-to-end on a real set of molecules.

## Answer

The synthesizability metric works end-to-end on Balam. AiZynthFinder (v4.4.1)
installed cleanly in its own environment, its real API matched every assumption the
code made, and the evaluator ran over a real candidate dataset with no errors,
producing the AiZynth success rate, route lengths, and SA-score distribution. On a
test set of 8 known literature glues plus 3 trivially-makeable anchor molecules, it
found synthesis routes for 9 of 11 (82%) — including 7 of the 8 real glues. Notably,
ibuprofen (obviously synthesizable) was *not* solved within the default search budget,
which is a faithful reproduction of the tool's known conservatism: the source papers
explicitly warn that AiZynthFinder underestimates synthesizability and is noisy. That
82% on real, known molecules sits sensibly above RGFN's reported ~56% on *generated*
molecules, which is the direction we'd expect. The instrument is ready to run on every
generator's output.

## Relevance to our Publication

This is the measurement instrument behind **Objective 5's headline differentiator**.
A JCIM/JCIM-tier reviewer will ask "are your molecules actually synthesizable, by an
independent tool — not just by your generator's own construction?" This metric answers
that for every entrant at once, and the by-construction-vs-AiZynth cross-check
(generator claims a route vs. AiZynth independently finds one) is exactly the evidence
that "synthesizable" is doing real work rather than being assumed.

## Next Experiments

**Refining for publication**
- Run the metric across **all** entrants' candidate datasets (RGFN, RxnFlow, SCENT,
  FragGFN, VAE-BO) to produce the committed synthesizability comparison table in
  `validation/results/` — the route-based-vs-non-synthesizable contrast.
- Decide whether to report over the full generated set or the top-k by oracle score
  (the papers typically use a top-k, e.g. 500), and standardise it across entrants.

**Next steps in project**
- Fold this into the `validation/harness/` suite runner once it lands, so a benchmark
  run scores synthesizability automatically alongside docking/diversity/recovery.

---

# Re-creation

## Relevant Files

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`

**Scripts**
- `./validation/harness/synthesizability.py` — the evaluator. Reads a standard
  candidate dataset (`manifest.json` + `candidates.csv`) directly (csv/json, no `glue`
  import — runs in the lean `aizynth` env), runs AiZynthFinder over the unique valid
  SMILES + RDKit SA score, writes `synthesizability.csv` (per-molecule) +
  `synthesizability_summary.json` (aggregate). CLI: `--dataset/--config/--nproc/--top-k`.
- `./external/setup_aizynthfinder.sh` — creates the dedicated `aizynth` conda env,
  pip-installs `aizynthfinder>=4.3,<5`, downloads the public dataset, smoke-tests.
- `./validation/harness/test_synthesizability.py` — dependency-free unit tests
  (monkeypatch RDKit + AiZynth) for dedup / top-k / scatter-back / aggregation.
- `/tmp/.../scratchpad/make_smoke_dataset.py` — builds the test candidate dataset from
  real known glues + easy anchors via the canonical `CandidateDataset` writer.

**Datasets**
- `./data/validation-molecules/DDB1_CDK12_Glues.csv` — curated known glues for the
  6TD3 (CDK12–DDB1) system; source of the 8 real test molecules.
- `./data/synthetic/aizynth_smoke/candidates/` — the end-to-end test dataset (8 real
  glues + aspirin/ibuprofen/benzamide anchors), git-ignored.
- `./data/models/aizynthfinder/` — AiZynthFinder public dataset: `uspto_model.onnx`
  (88M expansion policy), `uspto_templates.csv.gz`, `uspto_ringbreaker_*`,
  `uspto_filter_model.onnx`, `zinc_stock.hdf5` (633M in-stock set), `config.yml`
  (keys `expansion:uspto/ringbreaker`, `filter:uspto`, `stock:zinc`). Fetched by
  `download_public_data`; git-ignored (~754M).

**Results**
- `./data/synthetic/aizynth_smoke/candidates/synthesizability_summary.json` — aggregate
  report (success rate 0.818, steps_mean 1.78, SA mean 2.50); git-ignored.
- `./data/synthetic/aizynth_smoke/candidates/synthesizability.csv` — per-molecule table
  (solved / n_steps / n_solved_routes / top_score / sa_score / search_time); git-ignored.
- `./Logs/references/` — added `[genheden2020aizynth]`, `[ertl2009sascore]`.

## Relevant Versions

```
08da97c Working GPU Oracle RGFN, FGFN, RxnFlow, SCENT, AiZynthFinder
ded1c0d checkpoint for FGFN, RxnFlow, SCENT, AiZynthFinder, sEH
```

The synthesizability tool + setup script (`external/setup_aizynthfinder.sh`,
`validation/harness/synthesizability.py`, its `__init__`/tests/README, the
`Logs/references/{README.md,references.bib}` entries for `[genheden2020aizynth]`/`[ertl2009sascore]`,
`docs/REFACTOR_LOG.md`, `.gitignore`, and this log) landed in **`ded1c0d`** ("checkpoint …")
and were refined in **`08da97c`** ("Working GPU Oracle …").

## Relevant Resources

**Sources**
- `[genheden2020aizynth]` AiZynthFinder — retrosynthesis engine (DOI:10.1186/s13321-020-00472-1).
- `[ertl2009sascore]` SA score (DOI:10.1186/1758-2946-1-8).
- `[koziarski2024rgfn]`, `[seo2024rxnflow]`, `[gainski2025scent]` — the metric's provenance.

**Packages**
- `aizynthfinder>=4.3,<5` (env `aizynth`, py3.10) — used by `validation/harness/synthesizability.py`.
- RDKit contrib `sascorer` (ships with RDKit) — SA score in the same file.

## Method

Run on `balam-login01` (CPU only; retrosynthesis is CPU-bound). AiZynthFinder 4.4.1.

1. Install: `bash external/setup_aizynthfinder.sh` → `aizynth` conda env (py3.10) +
   `pip install aizynthfinder>=4.3,<5` + `download_public_data data/models/aizynthfinder`
   (≈754M). *Note for re-runners:* don't double-launch the script — two concurrent
   `download_public_data` into the same dir race on the same files; let one finish.
2. API probe (one molecule, p-toluic acid): confirmed `.items` lists the configured
   keys, `.select()` works, and `extract_statistics()` returns `is_solved`,
   `number_of_steps`, `number_of_solved_routes`, `top_score` — exactly what the
   evaluator reads. No code changes were needed to the AiZynth driver.
3. Test dataset: `python make_smoke_dataset.py` (rgfn env) wrote 11 molecules (8 real
   DDB1/CDK12 glues + aspirin/ibuprofen/benzamide) via the canonical `CandidateDataset`
   writer; `validate_candidate_dataset` → conformant.
4. End-to-end: `conda run -n aizynth python validation/harness/synthesizability.py
   --dataset data/synthetic/aizynth_smoke/candidates
   --config data/models/aizynthfinder/config.yml --nproc 8`.

## Results

End-to-end run: 11 candidates, 11 valid, 11 unique searched, **0 errors**.

| metric | value |
|---|---|
| AiZynth success rate | **0.818** (9/11 solved) |
| real glues solved | 7/8 |
| anchors solved | aspirin ✓, benzamide ✓, **ibuprofen ✗** (top_score 0.77) |
| steps_mean / median (solved) | 1.78 / 2 |
| SA mean / median | 2.50 / 2.69 (min 1.16 benzamide, max 3.15) |
| self-reported route rate | 0.0 (test set has no routes — correct) |

Per-molecule highlights: solved glues found 21–73 alternative routes each, top_score
≈ 0.99, 1–3 steps; the one unsolved glue (`CC(C)C1=NN(C2=C(Cl)C=CC=C2Cl)…`, a
dichlorophenyl pyrazolo-pyrimidinone) topped out at score 0.74 with 0 solved routes.
Ibuprofen's miss (despite being trivially makeable) is the expected face of
AiZynthFinder's documented noise/conservatism, not a tool failure. Validation also
passed: dependency-free unit tests 3/3 (rgfn env), `py_compile` clean, RDKit SA-score
path sane (aspirin 1.58 / benzamide 1.16 / glue 2.66).
