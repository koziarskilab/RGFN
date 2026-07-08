# data/validation-molecules/

Curated sets of **real, validated molecular glues** used as *inputs* — the known
positives against which we validate the docking oracle (the "knowns" in the
known-vs-decoy discrimination of `experiments/oracle_validation/`). These are
inputs to the pipeline, not test fixtures or generated outputs.

| File | Set | Used by |
|---|---|---|
| `DDB1_CDK12_Glues.csv` | 175 real CDK12-CCNK/DDB1 glues (161 unique, 123 purine-based) — the 6TD3 KNOWN+ positives | `experiments/oracle_validation/docking_6td3/dock_cluster.py` |
| `CRBN_GSPT1_Glues.csv` | 200 real CRBN/GSPT1 glues (199 active, 188 glutarimide-bearing) — the 5HXB/CRBN KNOWN+ positives | `experiments/oracle_validation/docking_crbn/dock_cluster_crbn.py` |

The superseded purchasable scaffold library
(`Enamine_CRBN_Molecular_Glue_Library_*.smiles`, large) is git-ignored; the two
curated CSVs above are force-tracked (they override the global `*.csv` ignore).
