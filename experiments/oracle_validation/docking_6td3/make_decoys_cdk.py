#!/usr/bin/env python3
"""Generate physically-realistic 'fake' CDK12-glue decoys as a discrimination control.

Mirrors make_decoys.py (CRBN) but for the CR8/cyclinK system: the conserved ATP-pocket binder
here is the 2,6,9-trisubstituted PURINE (the kinase-hinge warhead), and the glue activity comes
from the C6 arm reaching out to contact DDB1. We keep a CR8-like purine core with a free C6
amine handle and cap it with random drug-like reagents (the SAME reaction/reagent set used for
the CRBN decoys). The arms are NOT selected to reach DDB1, so this is a fair baseline: if these
score (and especially gain a DDB1 bonus) like the real glues, the proxy is just reading ATP-pocket
binding; if the real glues win on the Tier2-Tier1 differential, the proxy rewards a productive arm.

Output: decoys_cdk.smiles (SMILES<TAB>ID, header) -- every molecule keeps the purine warhead.
"""
import os
import random
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "docking_crbn"))
import make_decoys as MD  # reuse REACTIONS + REAGENTS (amide/sulfon/urea/redam)

PURINE = Chem.MolFromSmarts("c1nc2c(n1)ncnc2")  # fused purine bicycle
PURINE2 = Chem.MolFromSmarts("c1[nX3]c2c(n1)ncnc2")
ALLOWED = set("C H N O S F Cl".split())

# CR8-like purine cores with a FREE C6 aromatic-amine handle (the DDB1-facing position)
SCAFFOLDS = {
    "cr8": "CC[C@@H](CO)Nc1nc(N)c2ncn(C(C)C)c2n1",  # C2=(R)-aminobutanol, C6=NH2, N9=iPr  (CR8 minus its arm)
    "me9": "CC[C@@H](CO)Nc1nc(N)c2ncn(C)c2n1",  # N9=methyl variant
    "pip": "CC(=O)N1CCN(c2nc(N)c3ncn(C(C)C)c3n2)CC1",  # C2=acetylpiperazine (LL-K12 series), C6=NH2
    "amp": "Nc1nc(N)c2ncn(C(C)C)c2n1",  # simple 2,6-diamino-9-iPr purine
}


def has_purine(m):
    return m.HasSubstructMatch(PURINE) or m.HasSubstructMatch(PURINE2)


def valid(mol):
    if mol is None or not has_purine(mol):
        return False
    if not all(a.GetSymbol() in ALLOWED for a in mol.GetAtoms()):
        return False
    return 250.0 <= Descriptors.MolWt(mol) <= 650.0


def main():
    products = {}
    for sname, ssmi in SCAFFOLDS.items():
        scaf = Chem.MolFromSmiles(ssmi)
        if scaf is None or not scaf.HasSubstructMatch(Chem.MolFromSmarts("[c][NX3;H2]")):
            print(f"  WARN scaffold {sname} has no aromatic-amine handle, skipping")
            continue
        for rxn_name, smarts in MD.REACTIONS.items():
            rxn = AllChem.ReactionFromSmarts(smarts)
            for rsmi in MD.REAGENTS[rxn_name]:
                reag = Chem.MolFromSmiles(rsmi)
                if reag is None:
                    continue
                for prod in rxn.RunReactants((scaf, reag)):
                    m = prod[0]
                    try:
                        Chem.SanitizeMol(m)
                    except Exception:
                        continue
                    if not valid(m):
                        continue
                    can = Chem.MolToSmiles(m)
                    if can not in products:
                        products[can] = f"DECOY_{sname}_{rxn_name}_{len(products):04d}"

    items = list(products.items())
    random.Random(7).shuffle(items)
    out = os.path.join(HERE, "decoys_cdk.smiles")
    with open(out, "w") as fh:
        fh.write("SMILES\tID\n")
        for can, did in items:
            fh.write(f"{can}\t{did}\n")
    mws = sorted(Descriptors.MolWt(Chem.MolFromSmiles(c)) for c, _ in items)
    print(f"generated {len(items)} unique CDK-glue decoys -> {os.path.basename(out)}")
    print(f"MW: min {mws[0]:.0f}  med {mws[len(mws)//2]:.0f}  max {mws[-1]:.0f}")


if __name__ == "__main__":
    main()
