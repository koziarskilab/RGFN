#!/usr/bin/env python3
"""(Re)build the seed datasets D_0 for the 6TD3 active-learning example.

- seed_6td3.csv : real validated-docking labels (ddb1_dvina differential, Vina
  Tier2-Tier1) for known glues + decoys, pulled from
  experiments/oracle_validation/docking_6td3/. Vina T2-T1 is the best of six candidate
  signals (Logs 006/007); more negative = better glue.
- seed_mock.csv : a handful of drug-like SMILES scored by MockGlueOracle, for the
  local CPU smoke test.

Run from the repo root:  python experiments/active_learning_6td3/build_seeds.py
"""
import csv
import os

from glue.oracles.mock_oracle import MockGlueOracle

HERE = os.path.dirname(os.path.abspath(__file__))
DOCK_DIR = os.path.join(HERE, "..", "..", "oracle_validation", "docking_6td3")


def build_6td3():
    out = os.path.join(HERE, "seed_6td3.csv")
    rows = []
    for fn in ("known_results.csv", "decoy_cdk_results.csv"):
        path = os.path.join(DOCK_DIR, fn)
        if not os.path.exists(path):
            print(f"  (skip, missing) {path}")
            continue
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r.get("status") != "ok":
                    continue
                # ddb1_dvina = Vina(Tier2) - Vina(Tier1), the VALIDATED differential
                # (Log 002); more negative = better glue. NOT ddb1_dcnnaff.
                smi, diff = r.get("smiles", "").strip(), r.get("ddb1_dvina", "").strip()
                if not smi or not diff:
                    continue
                try:
                    rows.append((smi, float(diff), r.get("set", "")))
                except ValueError:
                    continue
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "label", "set"])
        w.writerows(rows)
    print(f"seed_6td3.csv: {len(rows)} rows -> {out}")


def build_mock():
    smis = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Clc1ccccc1C2=NCC(=O)Nc3ccc(cc23)N",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
        "CC(C)NCC(O)COc1cccc2ccccc12",
        "O=C(O)c1ccccc1Nc1ccccc1Cl",
        "CCOC(=O)c1ccccc1",
        "c1ccc2c(c1)ccc1ccccc12",
        "OCC1OC(O)C(O)C(O)C1O",
        "CC1=CC(=O)CC(C)(C)C1",
        "NS(=O)(=O)c1ccc(N)cc1",
        "CCN(CC)CCOC(=O)c1ccc(N)cc1",
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    ]
    labels = MockGlueOracle().score(smis)
    out = os.path.join(HERE, "seed_mock.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "label"])
        for s, l in zip(smis, labels):
            if l == l:  # drop NaN
                w.writerow([s, round(l, 5)])
    print(f"seed_mock.csv -> {out}")


if __name__ == "__main__":
    build_6td3()
    build_mock()
