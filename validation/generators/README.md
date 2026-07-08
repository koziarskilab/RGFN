# validation/generators/

Baseline molecule generators we compare our RGFN pipeline against. Each is a
**thin adapter** exposing one common interface; the heavy third-party code is
installed separately (not vendored into this repo).

These are *entrants* in a benchmark, not part of the production pipeline. Nothing
in `glue/` imports from here (see the boundary rule in `../README.md`).

## Planned interface (not yet implemented)

A common `Generator` base (intended home: `base.py`) so the harness
(`validation/harness/`) can drive every entrant identically — roughly:

- `setup(config)` — load weights / build the model.
- `generate(oracle, budget) -> molecules` — produce candidates under a fixed
  oracle-call budget (so RGFN and the baselines are compared on equal compute).
- `name`, `is_synthesizable` — metadata the harness records per run.

The point of the equal-budget interface is the headline comparison reviewers will
ask for (`docs/RESEARCH_CONTEXT.md`, Objective 4): does our **synthesizable**
generator beat non-synthesizable baselines on the *same* oracle and budget?

## Output format (the standard every entrant emits)

`generate(...)` must write its candidates in the **standard candidate-dataset
format** — `manifest.json` + `candidates.csv` (+ `routes.jsonl` for synthesizable
entrants). This is what makes the comparison uniform: the harness reads any
entrant's output the same way. Use the shared writer (it lives in `glue/`, which
we may import from here):

```python
from glue.datasets.candidates import CandidateDataset      # or from_smiles_table
```

Full spec + examples: [`docs/CANDIDATE_DATASET_FORMAT.md`](../../docs/CANDIDATE_DATASET_FORMAT.md).
A `fraggfn`/`vae_bo` entrant simply omits routes (`is_synthesizable = False`);
`synflownet` and our `rgfn` adapter include them.

## Entrants

| Dir | Generator | Synthesizable? | Install |
|---|---|---|---|
| `rgfn/` | **Our pipeline** (thin adapter → `glue/` + `rgfn/`) | ✅ by construction | in-repo |
| `rxnflow/` | RxnFlow (reaction-template + building-block GFlowNet) — **implemented** (exp `016`) | ✅ `has_route=1` | `external/setup_rxnflow.sh` |
| `scent/` | SCENT (cost-aware template GFlowNet, RGFN fork) — **implemented** (exp `017`) | ✅ `has_route=1` | `external/setup_scent.sh` |
| `fraggfn/` | Fragment-based GFlowNet (Recursion `gflownet`) — **implemented** (exp `015`) | ❌ (fragment assembly, no route) | `external/setup_fraggfn.sh` |
| `synflownet/` | SynFlowNet (reaction-based GFlowNet baseline) | ✅ | `external/setup_synflownet.sh` |
| `vae_bo/` | VAE + Bayesian optimization | ❌ | `external/setup_vae_bo.sh` |

The `rgfn/` adapter is deliberately thin — it calls the real production pipeline
in `glue/` rather than copying it, so a benchmark entrant can never drift from
what we actually ship.

> Status: `fraggfn/` (non-synthesizable foil), `rxnflow/` (synthesizable peer) and
> `scent/` (cost-aware peer) are implemented — each in its own conda env, driven
> through an active-learning loop against the shared oracle via the bridge
> `scripts/score_batch.py` (route-aware via `--routes` for synthesizable entrants);
> see their READMEs and `Logs/015`/`Logs/016`/`Logs/017`. All three can target
> **either** benchmark: the 6TD3 glue `dvina` differential (`docking_6td3_gpu`) or
> the classic **sEH** single-target binding reproduction check (`docking_seh`) —
> pick the `validation/configs/<gen>_{6td3,seh}.*` spec. `rgfn/`, `synflownet/`,
> `vae_bo/` are still placeholders. See `external/` for each baseline's install
> script.

## Note on FragGFN's interface

FragGFN doesn't expose the single-shot `generate(oracle, budget)` method sketched
above — it runs an **active-learning loop** (`[bengio2021gflownet]` Alg. 1) matching
the RGFN entrant, and labels each round's batch through the oracle bridge
(`scripts/score_batch.py`), which is also what writes the standard candidate
dataset. When the harness lands, a `generate(...)`-style wrapper can call
`run_fraggfn_al.py`; the on-disk output is already the standard format the harness
reads. See `fraggfn/README.md`.
