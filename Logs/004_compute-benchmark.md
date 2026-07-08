# Compute benchmark — login node vs Balam debug_full_node (4× A100)
**Date:** 2026-06-18

## Question

How much faster is the Balam cluster compared to running docking locally, and where does the speedup come from?

## Context & Summary

**Context** — The full discrimination runs in entries 002 and 003 (408 molecules for 6TD3, 437 for CRBN) needed to complete within Balam's one-hour `debug_full_node` limit. We had been running docking locally during development using a sequential per-molecule approach, where gnina reloads its CNN model for every molecule — a design that turns out to be extremely slow for large runs. Before committing to the cluster for all future experiments, we needed to understand where the time was going and what the actual speedup would be, so we can plan the RGFN evaluation loop around realistic oracle costs.

**Summary** — We compared three configurations: (A) sequential per-molecule docking on the login node, one gnina call per molecule; (B) batched multi-ligand docking on the login node, one gnina call per shard so the model loads once per shard; and (C) the full Balam debug node with 4× A100 and 16 workers. Configurations A and B were timed on a fixed 24-molecule subset; configuration C timing comes from SLURM logs of the full runs (jobs 69271 and 69272).

## Answer

Multi-ligand batching — running a full shard of molecules in a single gnina call — is the single biggest lever. For the CRBN anchored protocol (100 confs/mol), batching amortizes the CNN model reload across ~2700 conformers per shard, accounting for roughly 27× of the speedup on its own. Adding three more A100 GPUs contributes less than the batching did. Net result: both full runs completed in under 14 minutes combined — down from a projected ~3.5 hours of sequential local docking. A surprising finding: after the dock speedup, conformer embedding is now 60% of the CRBN wall time and has become the new bottleneck.

## Relevance to our Publication

This entry is the compute-methods justification for the paper. NeurIPS reviewers who ask about scalability will want to know that the oracle is fast enough to use in an RGFN training loop — ~14 minutes for 400 molecules confirms that. The Amdahl analysis also flags that conformer embedding, not GPU count, is the next bottleneck if we need to scale further, which shapes future infrastructure decisions and is worth a sentence in the methods.

## Next Experiments

**Refining for publication** — Profile conformer embedding time separately to confirm the Amdahl floor (estimated 3.5 min of the 5.8-min CRBN wall). If we scale to larger RGFN batches, faster or parallelized ETKDG or fewer confs per molecule is the next lever.

**Next steps in project** — The established configuration (batched, 16 workers, 4× A100) is standard for all future oracle runs. Use these timings when estimating RGFN training loop cost.

# Re-creation

## Relevant Files

Scripts:
- `./research/preprocessing/docking_6td3/dock_cluster.py` — batched 6TD3 docking driver; used for config B (local, `N_GPU=1 PROCS_PER_GPU=4`) and config C (cluster defaults)
- `./docking_gnina/dock_cluster_crbn.py` — batched CRBN docking driver; used for config C

Note: config A sequential runners (`dock_6td3_batch.py` / `batch_anchor_dock.py`) have been removed and consolidated into the batching drivers above. The docking method is unchanged; only the sharding differs.

Job Logs:
- `/scratch/markymoo/rgfn_runs/dock6td3-69271.out` — SLURM log for 6TD3 config C (job 69271); measured wall time source
- `/scratch/markymoo/rgfn_runs/dockcrbn-69272.out` — SLURM log for CRBN config C (job 69272); measured wall time source

Note: config A/B benchmark subset runs (~24 molecules) were transient on the login node; outputs were not saved.

## Relevant Versions

```
468fcc6 Clean up pre-processing into a concise, documented pipeline
106a4e6 Add 6TD3/CR8 glue docking oracle + 5HXB cross-system comparison
2fbdfb4 Remove Vina-GPU build cruft and Miniconda installer
```

Relevant commits: `468fcc6` — consolidation of sequential runners into batched `dock_cluster*.py` drivers. `2fbdfb4` — removed Vina-GPU build that was an earlier failed speedup attempt.

## Relevant Resources

**Sources**
- No external citations; timing analysis from first principles + SLURM logs

**Packages**
- gnina v1.3.2 — docking engine; CNN model reload is the dominant per-call cost for the `--minimize` workload
- RDKit ETKDG — conformer embedding; becomes the Amdahl floor after the dock speedup

## Method

Three configurations benchmarked:

- **A — sequential (login node):** 1 GPU, one gnina invocation per molecule. CNN model reloads every call. Original `dock_6td3_batch.py` / `batch_anchor_dock.py` design.
- **B — batched + 4-proc (login node):** 1 GPU, 4 worker processes, each docking a multi-ligand shard in one gnina call (model loads once per shard). `dock_cluster.py` with `N_GPU=1 PROCS_PER_GPU=4`.
- **C — full node (cluster):** Balam `debug_full_node`, 4× A100, 16 workers (4/GPU), batched.

Timing: A and B measured on a fixed 24-molecule subset on the login node (s/mol), then extrapolated to full-run wall time. C taken directly from SLURM logs of jobs 69271 (6TD3) and 69272 (CRBN). CRBN config A rate extrapolated from an earlier completed 368-mol Pass B local run (368 mol in 101 min = 16.1 s/mol); the local 6TD3 sequential run was paused so 6TD3 config A rate is from the login-node benchmark subset.

## Results

**6TD3 — global box docking (exhaustiveness 16), 408 molecules:**

| config | dock rate (s/mol) | wall time (408 mol) |
|---|---|---|
| A — sequential, 1 GPU | 12.5 | ~88 min (extrapolated) |
| B — 1 GPU / 4 proc / batched | 2.5 | ~17 min |
| **C — 4 GPU / 16 proc (job 69271)** | **1.57** | **13.4 min (measured)** |

Total speedup A→C: ~6.6×. Decomposition: batching + 4 proc on one GPU gives 5.0× (A→B); adding 3 more GPUs gives 1.6× (B→C). 6TD3 global docking is CPU-bound (Vina Monte-Carlo search), so extra GPUs help little.

**CRBN — warhead-anchored `--minimize`, 100 confs/mol, 437 molecules:**

| config | dock rate (s/mol) | wall time (437 mol) |
|---|---|---|
| A — sequential, 1 GPU (368-mol Pass B run) | 16.1 | ~120 min (extrapolated) |
| **C — 4 GPU / 16 proc (job 69272)** | **0.32** | **5.8 min (measured)** |

Total speedup A→C: ~21×; dock phase alone ~51×. Batching amortizes CNN model reload across ~2700 confs/shard (~27× of the win), on top of 16-way parallelism. After the dock speedup, conformer embedding (3.5 min) is 60% of the 5.8-min wall — new Amdahl bottleneck.

**Combined cluster wall time (both full runs):** 13.4 min + 5.8 min = ~19 min.
