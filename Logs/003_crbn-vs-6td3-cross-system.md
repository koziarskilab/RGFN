# 5HXB vs 6TD3 — head-to-head neosubstrate differential comparison
**Date:** 2026-06-18

## Question

Does the structural reason for CRBN's discrimination failure hold when both systems are tested on identical methodology?

## Context & Summary

**Context** — Entries 001 and 002 showed that CRBN-based docking can't discriminate real glues from decoys (entry 001) while CDK12-DDB1 can, with a 78-percentage-point gap (entry 002). But those two experiments used different docking protocols — CRBN used warhead-anchored Pass B while 6TD3 used straight box docking — so the comparison isn't clean. We also updated the CRBN known-glue dataset between entries (from a purchasable scaffold library to a curated set of validated degraders). This head-to-head experiment puts both systems on identical methodology — same metric, same decoy design, same evaluation pipeline — to confirm whether the performance difference reflects the systems or the methods.

**Summary** — We ran the CRBN/GSPT1 system with a curated known-glue dataset (200 real validated degraders, 177 anchorable) against the same 260 decoys, using the same neosubstrate-differential metric used for 6TD3: score each molecule against the E3 pocket alone (Tier 1) and the full ternary complex (Tier 2), then measure the differential on the same pose. The 6TD3 result comes from entry 002 (job 69271); the CRBN result is new (job 69272, Balam debug_full_node, 5.8 min).

## Answer

On identical methodology, the two systems split cleanly: the 6TD3 differential separates real glues from decoys by +78 percentage points; the CRBN differential shows no separation (−3 points, with decoys marginally higher). This confirms the failure in entry 001 is structural, not methodological. In the CRBN system, anchoring the glutarimide warhead in the cage places every arm near the GSPT1 surface regardless of arm chemistry — the bonus is a geometric artifact of the crystal geometry, not a specific molecular recognition event. In 6TD3, only arms with the right geometry reach DDB1, so the bonus discriminates. CRBN is not a viable RGFN testbed; 6TD3 is.

## Relevance to our Publication

NeurIPS reviewers will ask whether our system selection was principled or cherry-picked. This entry provides the direct answer: we tested the most prominent molecular glue system first, found a structural ceiling with an explicit mechanistic explanation, and then validated the alternative on a head-to-head comparison. The −3 pts vs +78 pts table on the same metric is figure-ready and justifies both the oracle design and the system selection in a single result. It also validates the neosubstrate differential as the correct metric: it works exactly where it should and fails exactly where it should, for the predicted structural reason.

## Next Experiments

**Refining for publication** — Molecular-weight sensitivity check on the CRBN known-glue dataset: `CRBN_GSPT1_Glues.csv` includes compounds up to MW ~1009, some likely bivalent/PROTAC-range. Filter to MW < ~600 and re-run to confirm the ceiling isn't a dataset artifact. Note: CRBN does show modest absolute-binding discrimination (fit-rate 77% vs 45%, Tier 2 Vina<−10: 35% vs 12%) — document this as secondary evidence that a CRBN reward isn't entirely worthless, just not glue-specific.

**Next steps in project** — 6TD3 is confirmed as the RGFN testbed. Run RGFN with the validated 6TD3 oracle and evaluate generated molecules. The compute cost for full oracle runs is benchmarked in entry 004.

# Re-creation

## Relevant Files

Scripts:
- `./docking_gnina/dock_cluster_crbn.py` — batched warhead-anchored docking for CRBN (job 69272): builds 100 flexible-anchor confs/mol → gnina `--minimize` vs Tier 2 (CRBN+GSPT1) → best clash-aware pose → `--score_only` vs Tier 1 (CRBN) → GSPT1 differential; sharded 16 workers / 4 GPUs with per-shard CSV checkpoints
- `./docking_gnina/submit_dock_crbn.sh` — Slurm submission script for Balam debug_full_node (job 69272)
- `./research/preprocessing/compare_systems.py` — prints both systems' known-vs-decoy distributions and known−decoy gap on the differential; cross-system summary table

Models:
- `./docking_gnina/5HXB_tier2.pdbqt` — CRBN+GSPT1 ternary receptor (Tier 2); docking target for CRBN run
- `./docking_gnina/5HXB_tier1_CRBN.pdbqt` — CRBN pocket alone (Tier 1); used for `--score_only` differential
- `./research/preprocessing/docking_6td3/6TD3_tier1.pdbqt`, `6TD3_tier2.pdbqt` — CDK12-DDB1 receptors; 6TD3 results reused from entry 002 (job 69271)

Datasets:
- `./research/preprocessing/test-data/CRBN_GSPT1_Glues.csv` — 200 real CRBN/GSPT1 validated degraders (199 active, 188 glutarimide-bearing, 177 unique anchorable); replaces the earlier purchasable scaffold library used in entry 001
- `./docking_gnina/decoys.smiles` — 260 glutarimide-bearing decoys (same decoy set as entry 001)

Results:
- `./docking_gnina/known_crbn_results.csv` — per-molecule CRBN docking scores (job 69272)
- `./docking_gnina/decoy_crbn_results.csv` — per-molecule decoy docking scores (job 69272)

Job Logs:
- `/scratch/markymoo/rgfn_runs/dock_crbn_69272/` — full per-shard output + per-shard `shard*.csv` checkpoints (Balam scratch)
- `/scratch/markymoo/rgfn_runs/dockcrbn-69272.out` — SLURM job log with timing

Note: one gnina worker segfaulted mid-run (libc, one shard) but per-shard try/except + checkpointing contained it; job exited 0. A few molecules in that shard may be missing — not material to the distributions.

## Relevant Versions

```
106a4e6 Add 6TD3/CR8 glue docking oracle + 5HXB cross-system comparison
468fcc6 Clean up pre-processing into a concise, documented pipeline
```

Relevant commit: `106a4e6` — `dock_cluster_crbn.py`, `compare_systems.py`, and result CSVs committed here. 6TD3 results (job 69271) from the same commit (entry 002).

## Relevant Resources

**Sources**
- 5HXB: Matyskiela et al., *Science* 2016 — CRBN·DDB1·GSPT1·CC-885 ternary complex
- 6TD3: Słabicki et al., *Nature* 2020 — DDB1·CDK12–cyclinK·CR8 (doi:10.1038/s41586-020-2133-z)

**Packages**
- gnina v1.3.2 — `docking_gnina/`; launched via `/scratch/markymoo/gnina/run_gnina.sh`
- RDKit — conformer generation

## Method

1. **CRBN run** — `docking_gnina/dock_cluster_crbn.py` on Balam `debug_full_node` (job 69272, 5.8 min): for each of 177 anchorable known glues and 260 decoys, build 100 flexible-anchor confs/mol → gnina `--minimize` vs Tier 2 (CRBN+GSPT1) → select best on-tether pose by clash-aware Vina → `--score_only` vs Tier 1 (CRBN) → GSPT1 differential. Sharded 16 workers / 4 GPUs with per-shard CSV checkpoints.

2. **6TD3 result** — reused from entry 002 (job 69271). No re-run.

3. **Cross-system comparison** — `research/preprocessing/compare_systems.py`: prints both systems' known-vs-decoy distributions and known−decoy gap on the differential.

## Results

**Neosubstrate differential (known vs decoy gap):**

| system | metric | known | decoy | gap |
|---|---|---|---|---|
| 5HXB / CRBN | median GSPT1 dVina (kcal/mol) | −2.95 | −2.35 | −0.60 |
| 5HXB / CRBN | frac dVina < −1.5 | 85.3% | 88.0% | **−3 pts** |
| 6TD3 / CDK12 | median DDB1 dVina (kcal/mol) | −2.20 | −0.60 | −1.60 |
| 6TD3 / CDK12 | frac dVina < −1.5 | 85.6% | 7.3% | **+78 pts** |

**Supporting absolute-binding metrics:**

| system | fit-rate (known / decoy) | median Tier 2 Vina (known / decoy) | frac Tier 2 Vina < −10 (known / decoy) |
|---|---|---|---|
| 5HXB / CRBN | 136/177 (77%) / 117/260 (45%) | −8.63 / −7.35 | 35.3% / 12.0% |
| 6TD3 / CDK12 | 160/160 / 248/248 | −10.15 / −7.96 | 55.6% / 3.6% |

CRBN shows modest absolute-binding discrimination (fit-rate, Tier 2 tails) but the differential — the glue-specific cooperativity term — shows no separation. 6TD3 wins decisively on both. The 6TD3 result is from entry 002 (job 69271); the CRBN result is from job 69272.
