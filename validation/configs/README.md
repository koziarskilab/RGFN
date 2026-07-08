# validation/configs/

Run specs for comparative studies: *which generators*, on *which oracle /
benchmark*, at *what oracle-call budget and how many seeds*.

These are the validation-side analogue of `configs/glue/` (which configures the
production training pipeline). A spec here typically names:

- one or more entrants from `validation/generators/`,
- the oracle/benchmark to score against (a `validation/suites/` suite, and/or a
  `validation/oracles/` high-fidelity check),
- budget + seed settings so comparisons are apples-to-apples
  (`docs/RESEARCH_CONTEXT.md` Objective 4 wants ≥3 seeds with error bars).

Format follows the generator's own config system: **YAML** for FragGFN / RxnFlow
(`<gen>_<target>.yaml`), **gin** for SCENT (`scent_<target>.gin`, which `include`s
SCENT's own gin base). Each spec's budget/seed/β is held identical to the matching
RGFN run in `configs/glue/active_learning_<target>.gin` so comparisons are
apples-to-apples; the `oracle:` block names a `scripts/score_batch.py` oracle so
every entrant is scored by the same standard.

## Current configs

| Generator | 6TD3 (glue dvina) | sEH (single-target Vina) | smoke (mock oracle) |
|---|---|---|---|
| FragGFN | `fraggfn_6td3.yaml` | `fraggfn_seh.yaml` | `fraggfn_smoke.yaml` |
| RxnFlow | `rxnflow_6td3.yaml` | `rxnflow_seh.yaml` | `rxnflow_smoke.yaml` |
| SCENT | `scent_6td3.gin` | `scent_seh.gin` | `scent_smoke.gin` |

The two targets differ in the oracle: **6TD3** uses `docking_6td3_gpu` (the
neosubstrate `dvina` differential, threshold −2.0); **sEH** uses `docking_seh` (the
raw QuickVina2-GPU binding energy, threshold −8.0) — the classic GFlowNet benchmark,
used as a reproduction check. The generator-level `*_smoke` configs are
target-agnostic (mock oracle, tiny budget) and validate the loop on CPU.
