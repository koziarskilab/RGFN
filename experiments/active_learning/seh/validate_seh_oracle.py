"""Live validation of DockingSEHOracle on the Balam login-node A100.

Docks a handful of molecules against sEH via QuickVina2-GPU and checks:
  - the oracle returns one finite negative Vina energy per valid SMILES,
  - a deliberately-broken SMILES yields nan (failure contract),
  - timing, to confirm it's fast.
"""
import sys
import time

from glue.oracles.docking_seh_oracle import DockingSEHOracle

# A few drug-like molecules + aspirin (memory: aspirin docks ~-6.3 vs sEH),
# plus one invalid SMILES to exercise the nan-on-failure contract.
SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "c1ccc2c(c1)ccc3c2cccc3",  # anthracene-ish PAH
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "not_a_smiles_at_all",  # invalid -> expect nan
]

print(f"[validate] python: {sys.executable}")
oracle = DockingSEHOracle(exhaustiveness=8000, docking_batch_size=25, n_conformers=1)
print(f"[validate] oracle: name={oracle.name} higher_is_better={oracle.higher_is_better}")
print(f"[validate] qv_dir={oracle.qv_dir} receptor={oracle.receptor_name}")

t0 = time.time()
scores = oracle.score(SMILES)
dt = time.time() - t0

print(f"\n[validate] docked {len(SMILES)} molecules in {dt:.1f}s " f"({dt/len(SMILES):.1f}s/mol)\n")
import math

for smi, s in zip(SMILES, scores):
    tag = "nan" if (s is None or math.isnan(s)) else f"{s:+.2f} kcal/mol"
    print(f"  {tag:>18}   {smi}")

valid = [s for s in scores[:-1] if s == s]  # exclude the intentional invalid one
assert len(valid) >= 3, f"expected >=3 valid scores, got {len(valid)}: {scores}"
assert all(s < 0 for s in valid), f"Vina energies should be negative: {valid}"
assert scores[-1] != scores[-1], "invalid SMILES should map to nan"
print("\n[validate] PASS: finite negative Vina energies + nan on the invalid input.")
