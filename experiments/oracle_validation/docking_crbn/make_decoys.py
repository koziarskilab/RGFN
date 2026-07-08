#!/usr/bin/env python3
"""Generate physically-realistic 'random' glutarimide molecules as a baseline control.

Mirrors how real CRBN-glue libraries (and RGFN) are built: a fixed IMiD warhead +
a variable arm grown off the aniline handle. We cap the aniline of several IMiD
scaffolds with random drug-like reagents via realistic reactions (amide / urea /
sulfonamide / reductive amination). Arms are NOT selected for glue activity, so the
set is a fair decoy baseline: if these score like the known glues, the proxy is just
reading the warhead; if the known glues win, the proxy rewards productive arm contacts.

Output: decoys.smiles  (SMILES<TAB>ID, with header) -- every molecule is
glutarimide-eligible by construction.
"""
import random

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

GLUT = Chem.MolFromSmarts("C1CCC(=O)NC1=O")
ALLOWED = set("C H N O S F Cl".split())

# IMiD scaffolds with a free aniline NH2 handle (the standard derivatization point)
SCAFFOLDS = {
    "pom": "Nc1cccc2c1C(=O)N(C1CCC(=O)NC1=O)C2=O",  # pomalidomide (4-amino)
    "pom5": "Nc1ccc2c(c1)C(=O)N(C1CCC(=O)NC1=O)C2=O",  # 5-amino isomer
    "len": "O=C1CCC(N2Cc3cccc(N)c3C2=O)C(=O)N1",  # lenalidomide (4-amino)
    "len5": "O=C1CCC(N2Cc3ccc(N)cc3C2=O)C(=O)N1",  # 5-amino isomer
    "anil": "Nc1ccc(NC2CCC(=O)NC2=O)cc1",  # 3-(4-aminoanilino)glutarimide
}

# Reaction templates: aniline + reagent -> capped product
REACTIONS = {
    "amide": "[c:6][NX3;H2:1].[CX3:2](=[O:3])[OX2H1]>>[c:6][NX3;H1:1][C:2]=[O:3]",
    "sulfon": "[c:6][NX3;H2:1].[S:2](=[O:3])(=[O:4])[Cl]>>[c:6][NX3;H1:1][S:2](=[O:3])(=[O:4])",
    "urea": "[c:6][NX3;H2:1].[N:2]=[C:3]=[O:4]>>[c:6][NX3;H1:1][C:3](=[O:4])[NX3;H1:2]",
    "redam": "[c:6][NX3;H2:1].[CX3;H1:2]=[O:3]>>[c:6][NX3;H1:1][CX4;H2:2]",
}

REAGENTS = {
    "amide": [
        "CC(=O)O",
        "CCC(=O)O",
        "CC(C)C(=O)O",
        "CC(C)(C)C(=O)O",
        "OC(=O)C1CC1",
        "OC(=O)C1CCCCC1",
        "OC(=O)c1ccccc1",
        "OC(=O)c1ccc(F)cc1",
        "OC(=O)c1ccc(OC)cc1",
        "OC(=O)c1cccnc1",
        "OC(=O)c1ccncc1",
        "OC(=O)c1ccco1",
        "OC(=O)c1cccs1",
        "OC(=O)Cc1ccccc1",
        "Cn1cc(C(=O)O)cn1",
        "OC(=O)C1CCN(C)CC1",
        "OC(=O)c1ccccc1Cl",
        "OC(=O)CCc1ccccc1",
    ],
    "sulfon": [
        "CS(=O)(=O)Cl",
        "CCS(=O)(=O)Cl",
        "ClS(=O)(=O)c1ccccc1",
        "Cc1ccc(S(=O)(=O)Cl)cc1",
        "ClS(=O)(=O)C1CC1",
        "ClS(=O)(=O)c1cccnc1",
        "ClS(=O)(=O)c1cccs1",
        "ClS(=O)(=O)c1ccc(F)cc1",
        "Cn1cc(S(=O)(=O)Cl)cn1",
        "CN(C)S(=O)(=O)Cl",
        "ClS(=O)(=O)CC(C)C",
    ],
    "urea": [
        "CN=C=O",
        "CCN=C=O",
        "CC(C)N=C=O",
        "O=C=NC1CC1",
        "O=C=NC1CCCCC1",
        "O=C=Nc1ccccc1",
        "O=C=NCc1ccccc1",
        "O=C=Nc1ccc(F)cc1",
        "O=C=Nc1ccccc1C",
        "O=C=NCC1CCCO1",
        "O=C=Nc1cccnc1",
    ],
    "redam": [
        "CCC=O",
        "CC(C)C=O",
        "O=CC1CC1",
        "O=CC1CCCCC1",
        "O=Cc1ccccc1",
        "O=Cc1ccc(F)cc1",
        "O=Cc1cccnc1",
        "O=Cc1ccco1",
        "O=Cc1cccs1",
        "O=CCc1ccccc1",
        "O=CC1CCOCC1",
        "O=CC1CCN(C)CC1",
    ],
}


def valid(mol):
    if mol is None or not mol.HasSubstructMatch(GLUT):
        return False
    if not all(a.GetSymbol() in ALLOWED for a in mol.GetAtoms()):
        return False
    mw = Descriptors.MolWt(mol)
    return 250.0 <= mw <= 650.0


def main():
    products = {}  # canonical_smiles -> id
    for sname, ssmi in SCAFFOLDS.items():
        scaf = Chem.MolFromSmiles(ssmi)
        for rxn_name, smarts in REACTIONS.items():
            rxn = AllChem.ReactionFromSmarts(smarts)
            for rsmi in REAGENTS[rxn_name]:
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
    with open("decoys.smiles", "w") as fh:
        fh.write("SMILES\tID\n")
        for can, did in items:
            fh.write(f"{can}\t{did}\n")
    print(f"generated {len(items)} unique decoys -> decoys.smiles")
    # quick property summary
    mws = [Descriptors.MolWt(Chem.MolFromSmiles(c)) for c, _ in items]
    print(f"MW: min {min(mws):.0f}  med {sorted(mws)[len(mws)//2]:.0f}  max {max(mws):.0f}")
    by_rxn = {}
    for _, did in items:
        by_rxn[did.split("_")[2]] = by_rxn.get(did.split("_")[2], 0) + 1
    print("by reaction:", by_rxn)


if __name__ == "__main__":
    main()
