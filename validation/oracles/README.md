# validation/oracles/

**Validation-only scorers.** High-fidelity, expensive oracles used to *check*
top-k outputs after a run — they never enter the training loop.

This is the structural counterpart to `glue/oracles/`:

| | In-loop oracle (`glue/oracles/`) | Validation oracle (here) |
|---|---|---|
| Role | the expensive `O` the active-learning loop queries on query batches | post-hoc check of final top-k |
| Speed budget | must be callable per query batch | can be very slow (run once, on a handful) |
| Wired into reward? | yes (via a `glue/proxies/` adapter) | **never** |

Keeping these in a separate tree is what makes "never wired into the reward" a
property of the *structure*, not just a convention — `glue/` cannot import from
`validation/` (see `../README.md`).

## Planned scorers

- `boltz2/` — **Boltz-2 co-folding** validation of generated glue candidates.
  This is the high-fidelity check in `docs/RESEARCH_CONTEXT.md` Objective 5
  ("High-fidelity validation of top-k (co-folding / Boltz-2 / short MD)") and
  feeds the anti-gaming check: does the in-loop proxy correlate with a
  trusted high-fidelity score on the top-k?

> Note: short-MD stability is a candidate *complementary in-loop* oracle for the
> CRBN ceiling (Objective 3) — that belongs in `glue/oracles/`, not here. Only
> use this directory for scorers that are purely post-hoc validation.

> Scaffolding only — `boltz2/` is a placeholder.
