# 6TD3 / CDK12-DDB1 — oracle validation + discrimination run
**Date:** 2026-06-18

## Question

Can the docking score tell real CDK12-DDB1 molecular glue degraders apart from realistic fake molecules?

## Context & Summary

**Context** — Entry 001 showed that the CRBN system hits a discrimination ceiling: the docking score is equally good at placing real glues and random warhead-bearing molecules, because GSPT1 cooperativity comes from a large protein–protein surface that docking can't resolve. The structural hypothesis for why the CRBN oracle fails is that the glue contact with the neosubstrate is not ligand-mediated — gluing GSPT1 to CRBN requires the whole recruited surface, not just the small molecule's arm. CDK12-DDB1 is structurally different: CR8's exposed arm reaches directly into the DDB1 interface, and the glue contact is made by the small molecule itself, not the protein surface. If that hypothesis is correct, the CDK12-DDB1 system should discriminate cleanly.

**Summary** — We first validated that blind docking can recover the native CR8 crystal pose (confirming we don't need anchoring for this system). Then we docked 160 real CDK12-DDB1 glues and 248 purine-armed decoys (real warhead, random arm), scored each against CDK12 alone (Tier 1) and CDK12+DDB1 (Tier 2), and measured the DDB1 neosubstrate differential. If real glues' arms contact DDB1 and decoy arms don't, the differential should separate them decisively.

## Answer

Decisive discrimination: 85.6% of real glues show a strong DDB1 bonus versus only 7.3% of decoys — a 78-percentage-point gap that CRBN never produced. Blind docking also recovers the native CR8 pose at 1.23 Å ranked #1, confirming we don't need a specialized anchoring protocol. The oracle rewards a productive DDB1-contacting arm, not just warhead presence. This is the right testbed for RGFN.

## Relevance to our Publication

This entry provides the first validated oracle for the RGFN pipeline. For NeurIPS, reviewers will ask whether the oracle actually discriminates — this entry answers that directly with a 78-percentage-point gap on 408 molecules. Paired with entry 001, it also supports the claim that our system selection was principled: we chose CDK12-DDB1 because its glue interface is ligand-mediated, and the discrimination confirms that structural reasoning was correct. Entry 003 formalizes this into a head-to-head comparison for the methods section.

## Next Experiments

**Refining for publication** — Ablation: compare the DDB1 neosubstrate differential against Tier 2 absolute score alone. NeurIPS reviewers will ask whether the differential is necessary or whether Tier 2 Vina alone achieves similar discrimination. Entry 003 includes the supporting numbers; a dedicated ablation entry would isolate this as a standalone result.

**Next steps in project** — Run RGFN with this validated oracle and evaluate the generated molecules. Also test whether the oracle generalizes to additional CDK12-DDB1 glues or other molecular glue systems (MolGlueDB has candidates). The compute pipeline needed for fast oracle evaluation is benchmarked in entry 004.

# Re-creation

## Relevant Files

Scripts:
- `./research/preprocessing/clean_6td3.py` — carves 6TD3 copy 1 into Tier 1 (CDK12), Tier 2 (CDK12+DDB1), and Tier 3 (+cyclinK) receptor structures; extracts native CR8 pose
- `./research/preprocessing/docking_6td3/redock_cr8.py` — blind-redocks CR8, minimizes native pose in-place, computes Tier 1 and Tier 2 scores for native pose validation
- `./research/preprocessing/docking_6td3/make_decoys_cdk.py` — generates purine-armed decoys (CDK12 ATP-hinge warhead + random drug-like arm)
- `./research/preprocessing/docking_6td3/dock_cluster.py` — batched multi-GPU box-docking driver; docks Tier 2, selects best-CNN pose, scores that pose against Tier 1 for the differential
- `./research/preprocessing/docking_6td3/submit_dock_6td3.sh` — Slurm submission script for Balam debug_full_node (job 69271)
- `./research/preprocessing/compare_systems.py` — prints known-vs-decoy discrimination metrics for each system and cross-system

Models:
- `./research/preprocessing/docking_6td3/6TD3_tier1.pdbqt` — CDK12 alone (Tier 1); used to score baseline pocket binding without DDB1
- `./research/preprocessing/docking_6td3/6TD3_tier2.pdbqt` — CDK12+DDB1 (Tier 2); the docking target for the discrimination run
- `./research/preprocessing/docking_6td3/crystal_RC8.pdb` — native CR8 pose; used as the autobox ligand and as the redocking validation target
- `./models/6TD3_tier1_CDK12.pdb`, `./models/6TD3_tier2_CDK12_DDB1.pdb` — source PDB files before pdbqt conversion

Datasets:
- `./research/preprocessing/test-data/DDB1_CDK12_Glues.csv` — 175 real CDK12-CCNK/DDB1 glues (161 unique, 123 purine-based); known+ positives for discrimination run (160 docked)

Results:
- `./research/preprocessing/docking_6td3/known_results.csv` — per-molecule docking scores for known glues (job 69271)
- `./research/preprocessing/docking_6td3/decoy_cdk_results.csv` — per-molecule docking scores for decoys (job 69271)

Job Logs:
- `/scratch/markymoo/rgfn_runs/dock_6td3_69271/` — full per-shard output directory (Balam scratch)
- `/scratch/markymoo/rgfn_runs/dock6td3-69271.out` — SLURM job log with timing

## Relevant Versions

```
106a4e6 Add 6TD3/CR8 glue docking oracle + 5HXB cross-system comparison
7afd7f1 Add docking pre-processing pipeline, inference, and run scripts
```

Relevant commit: `106a4e6` — 6TD3 docking scripts (`clean_6td3.py`, `redock_cr8.py`, `dock_cluster.py`, `make_decoys_cdk.py`) and result CSVs committed here.

## Relevant Resources

**Sources**
- 6TD3 crystal structure: Słabicki et al., *Nature* 2020 — DDB1·CDK12–cyclinK·CR8 ternary complex (doi:10.1038/s41586-020-2133-z)

**Packages**
- gnina v1.3.2 (CNN-rescored docking) — `docking_6td3/`; launched via `/scratch/markymoo/gnina/run_gnina.sh`
- RDKit — conformer generation and decoy arm synthesis (`make_decoys_cdk.py`)
- OpenBabel — pdbqt conversion (`clean_6td3.py`)

## Method

1. **Structure prep** — `research/preprocessing/clean_6td3.py` carves 6TD3 copy 1: `models/6TD3_tier1_CDK12.pdb` (CDK12 only), `models/6TD3_tier2_CDK12_DDB1.pdb` (CDK12+DDB1), optional Tier 3 (+cyclinK); native CR8 extracted to `crystal_RC8.pdb`. Receptors → pdbqt via obabel.

2. **Validation** — `docking_6td3/redock_cr8.py`: blind-redock CR8 with autobox on crystal pose (`--autobox_ligand crystal_RC8.pdb --autobox_add 4`). Confirm native-pose recovery. Score native pose against Tier 1 and Tier 2 to verify DDB1 cooperativity is captured.

3. **Discrimination run** — `docking_6td3/dock_cluster.py` on Balam `debug_full_node` (job 69271, 13 min): dock 160 real glues + 248 decoys to Tier 2. Take best-CNN pose per molecule. Score that pose against Tier 1 via `--score_only` → DDB1 differential = Tier2 − Tier1. Sharded across 16 workers / 4 GPUs.

## Results

**CR8 native-pose validation:**

| config | RMSD to crystal | Vina (kcal/mol) | CNNaff | DDB1 bonus |
|---|---|---|---|---|
| Blind redock (rank 1) | 1.23 Å | −10.68 | 0.99 | — |
| Native in-place minimize | — | −10.56 | 0.985 | — |
| Tier 1 (CDK12 only) | — | −8.39 | — | — |
| Tier 2 (CDK12+DDB1) | — | −10.68 | — | −2.16 to −2.29 |

**Discrimination run (job 69271, n=160 known, n=248 decoy):**

| metric | known | decoy | gap |
|---|---|---|---|
| **frac DDB1 dVina < −1.5** | **85.6%** | **7.3%** | **+78 pts** |
| median DDB1 dVina (kcal/mol) | −2.20 | −0.60 | −1.60 |
| median Tier 2 Vina (kcal/mol) | −10.15 | −7.96 | −2.19 |
| median Tier 2 CNNaff | 7.82 | 6.70 | 1.13 |
| frac Tier 2 Vina < −10 | 55.6% | 3.6% | +52 pts |
