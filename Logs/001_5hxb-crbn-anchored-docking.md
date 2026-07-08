# 5HXB / CRBN — warhead-anchored docking oracle + decoy control
**Date:** 2026-06-11 → 2026-06-18

## Question

Can docking accurately distinguish real CRBN molecular glue degraders from realistic fake molecules?

## Context & Summary

**Context** — This is the project's first oracle validation experiment. Before RGFN can generate useful glue candidates, we need to confirm that our scoring function actually rewards the right molecular property — that it can tell real glues apart from decoys (molecules with the correct warhead but a random arm not selected for glue activity). The CRBN/GSPT1 system (PDB: 5HXB) was the natural starting point: it's one of the best-characterized molecular glue systems, with a known glue (CC-885) and a well-studied binding mechanism. The key structural challenge is that CRBN's warhead (glutarimide) binds in a deep tri-Trp cage that blind docking rarely penetrates — off-the-shelf docking never sampled the native pose in any configuration we tried.

**Summary** — We developed a two-pass warhead-anchored docking protocol: pin the glutarimide to its crystal cage coordinates, then sample and relax the free arm. We docked ~188 real CRBN/GSPT1 glues and 260 warhead-bearing decoys (real warhead, random drug-like arm) using this protocol. For each molecule, we scored it against the E3 pocket alone (Tier 1, CRBN only) and the full ternary complex with neosubstrate (Tier 2, CRBN+GSPT1), measuring the neosubstrate differential to see whether real glues show more cooperativity than decoys.

## Answer

The anchored docking works mechanically — it places 95% of known glues into the CRBN cage with near-native accuracy, recovering CC-885's native pose to −13.55 kcal/mol. But it cannot discriminate real glues from decoys: the Vina score distributions largely overlap (median gap 0.85 kcal/mol), and the neosubstrate bonus is nearly identical for real glues and decoys. The structural reason: pinning the glutarimide warhead in the cage automatically positions every arm near the GSPT1 surface, regardless of whether the arm makes a productive glue contact. GSPT1 cooperativity in this system lives in a large protein–protein interface, not in a ligand-mediated contact the docking score can resolve.

## Relevance to our Publication

This entry is the negative control that justifies our system selection. For NeurIPS, reviewers will ask why we didn't use CRBN — the most studied molecular glue system. Entries 001–003 together provide a direct structural answer: CRBN's geometry makes the differential a nonspecific artifact rather than a glue signal, and we show this with a side-by-side comparison on identical methodology. Documenting the failure and its structural explanation makes the system choice principled rather than arbitrary.

## Next Experiments

**Refining for publication** — A molecular-weight sensitivity check on the CRBN known-glue dataset (filtering to MW < ~600 Da) to confirm the discrimination ceiling isn't partly a dataset artifact: `CRBN_GSPT1_Glues.csv` includes compounds up to MW ~1009, some of which are likely bivalent/PROTAC-range and won't behave as glues under docking.

**Next steps in project** — Pivot to a system where the neosubstrate contact is ligand-mediated and sits inside the docking box: 6TD3/CDK12-DDB1, where CR8's arm directly contacts DDB1 (entry 002). Run both systems head-to-head on identical methodology to confirm the structural explanation (entry 003).

# Re-creation

## Relevant Files

Scripts:
- `./research/preprocessing/clean.py` — carves 5HXB copy 1 into Tier 1 (CRBN only) and Tier 2 (CRBN+GSPT1) receptor structures; retains Zn, removes glue; outputs to `models/`
- `./docking_gnina/anchor_dock.py` — sequential per-molecule warhead-anchored docking (Pass A and B); used in this entry, later consolidated into `dock_cluster_crbn.py`
- `./docking_gnina/dock_cluster_crbn.py` — batched multi-GPU warhead-anchored docking driver (supersedes `anchor_dock.py`)
- `./docking_gnina/make_decoys.py` — generates decoys: IMiD glutarimide scaffold + random drug-like arms via amide/urea/sulfonamide/reductive-amination coupling
- `./research/preprocessing/compare_systems.py` — prints known-vs-decoy discrimination metrics for each system and cross-system

Models:
- `./docking_gnina/5HXB_tier2.pdbqt` — CRBN+GSPT1 ternary receptor (Tier 2); used to score neosubstrate cooperativity
- `./docking_gnina/5HXB_tier1_CRBN.pdbqt` — CRBN pocket alone (Tier 1); baseline against which the differential is computed
- `./docking_gnina/crystal_85C.pdb` — reference CC-885 native pose; used to validate the anchored approach recovers the crystal geometry

Datasets:
- `./research/preprocessing/test-data/Enamine_CRBN_Molecular_Glue_Library_..._4560cmpds_*.smiles` — purchasable scaffold library (potential glues, not validated degraders); used as known+ positives in this entry; superseded by the curated `CRBN_GSPT1_Glues.csv` in entry 003

Results:
- `./docking_gnina/batch_results_passB.csv` — known-glue docking scores, Pass B (validated pipeline)
- `./docking_gnina/batch_results.csv` — known-glue docking scores, Pass A (baseline)
- `./docking_gnina/decoy_results_passB.csv` — decoy docking scores, Pass B
- `./docking_gnina/decoy_results.csv` — decoy docking scores, Pass A

## Relevant Versions

```
106a4e6 Add 6TD3/CR8 glue docking oracle + 5HXB cross-system comparison
7afd7f1 Add docking pre-processing pipeline, inference, and run scripts
```

Relevant commit: `106a4e6` — CRBN anchored-docking scripts (`anchor_dock.py`, `dock_cluster_crbn.py`, `make_decoys.py`) and receptor files committed here as part of the same PR as 6TD3 work.

## Relevant Resources

**Sources**
- 5HXB crystal structure: Matyskiela et al., *Science* 2016 — CRBN·DDB1·GSPT1·CC-885 ternary complex

**Packages**
- gnina v1.3.2 (CNN-rescored docking) — `docking_gnina/`; launched via `/scratch/markymoo/gnina/run_gnina.sh`
- RDKit — conformer generation (`make_decoys.py`)
- OpenBabel — pdbqt conversion (`clean.py`)

## Method

1. **Structure prep** — `research/preprocessing/clean.py` carves 5HXB copy 1: `models/5HXB_tier1_CRBN.pdb` (CRBN only), `models/5HXB_tier2_CRBN_GSPT1.pdb` (CRBN+GSPT1). Zn retained, CC-885 removed. Receptors → pdbqt via `obabel ... -xr -p 7.4`.

2. **Sampling failure (blind docking)** — Blind docking with Vina, gnina, and gnina with exhaustiveness 64 + CNN never samples CC-885's deep tri-Trp-cage pose (top score ~−8 kcal/mol, CNN ~0.5; all displaced from native). Native pose minimizes in place to −13.8 / CNN 0.98 — sampling failure, not scoring failure.

3. **Warhead-anchored Pass B** — `docking_gnina/anchor_dock.py`: pin glutarimide to crystal cage coordinates, sample/relax only the arm via gnina `--minimize`, select clash-aware best pose with warhead-escape filter (poses >2.5 Å from cage rejected). Pass A (hard-frozen warhead): recovers native but 29–42% of molecules yield no clash-free pose. Pass B (flexible anchor, 1.5 Å wiggle + tether filter): CC-885 → −13.55 / CNN 0.97 (≈ native gold); fit-rate jumps to 94% known / 80% decoy.

4. **Decoy control** — `docking_gnina/make_decoys.py` builds 260 IMiD-scaffold decoys. Known positives: 10% sample of purchasable Enamine CRBN library. Both sets docked with Pass B; distributions compared via `compare_systems.py`.

## Results

**CC-885 native-pose recovery:**

| config | Vina (kcal/mol) | CNNaff |
|---|---|---|
| Crystal pose (in-place min) | −13.8 | 0.98 |
| Pass B anchor dock | −13.55 | 0.97 |
| Blind docking (best of exh 64) | ~−8 | ~0.5 |

**Known-glue vs decoy discrimination (Pass B):**

| metric | known (library sample) | decoy (n=260) | gap |
|---|---|---|---|
| fit rate | 94% | 80% | +14 pts |
| median Vina (kcal/mol) | −6.70 | −5.86 | −0.85 |
| median CNNaff | 6.21 | 6.09 | +0.12 |
| frac Vina < −10 | 15% | 7% | +8 pts |

Vina-physics channel sharpened ~65% Pass A→B, but medians did not close → confirmed ceiling, not sampling noise. See entry 003 for the explicit neosubstrate-differential comparison using the same methodology as 6TD3.
