# validation/suites/

Benchmark suites — the actual tasks/metrics an entrant is scored on.

- `pmo/` — **Practical Molecular Optimization** (PMO): a community-standard
  benchmark of oracle-budgeted optimization tasks. We run it as an *external*
  sanity check that our generators (and the baselines) are competitive on a task
  the field already agrees on — independent of our glue-specific story.
- `glue_suite/` — **our own** benchmarks, the ones that carry the paper's claims
  and are applied to **every** entrant (RGFN + baselines), e.g.:
  - recovery / retrospective enrichment of known glues (glutarimide, purine
    chemistry),
  - the differential-matters ablation (Tier 2 − Tier 1 vs. Tier-2-absolute),
  - synthesizability advantage (by-construction routes vs. non-synthesizable
    baselines),
  - anti-gaming (in-loop proxy vs. high-fidelity `validation/oracles/` on top-k).

A suite defines *what good looks like*; the `../harness/` runs entrants through it
and `../results/` stores the committed numbers.

> Scaffolding only — both suites are placeholders.
