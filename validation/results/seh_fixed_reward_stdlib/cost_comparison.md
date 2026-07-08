# Same-library synthesis-cost comparison — sEH fixed-reward (glue_standard_v1)

Retroactive route pricing (SCENT `PathCostProxy` recursion) over the shared standard
library `glue_standard_v1` (SCENT SMALL prices+yields). Lower cost = cheaper synthesis.

| generator | priced | top-100 cost (median) | (mean) | route len | any-fallback frac |
|---|---|---|---|---|---|
| scent | 1000/1000 | 2.88 | 7.33 | 2.71 | 0.744 |
| rxnflow | 1000/1000 | 7.78 | 14.4 | 2.98 | 0.0 |
| rgfn | 1000/1000 | 34.15 | 36.9 | 3.94 | 0.0 |
| fraggfn | 0/1000 | None | None | None | None |

**Coverage caveat:** RGFN and RxnFlow price at 0% fallback (every route component is a
base `glue_standard_v1` block/reaction → exact). SCENT's routes hit ~74% fallback because
its **dynamic library** promotes synthesized intermediates to building blocks that are not
in the base library; those are imputed at the default cost (1.0), so SCENT's cost is an
approximate lower bound, not directly comparable to the fully-priced RGFN/RxnFlow numbers.
FragGFN carries no routes (non-synthesizable foil) → unpriced.
