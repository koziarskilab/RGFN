#!/usr/bin/env python3
"""Clean the 6TD3 molecular-glue complex into docking-ready receptor tiers + the native ligand.

6TD3 is DDB1 bound to the CR8-engaged CDK12-cyclinK complex (Slabicki et al., Nature 2020):
CR8 (ligand ``RC8``) sits in the CDK12 ATP pocket and its solvent-exposed pyridyl-phenyl arm
protrudes to contact DDB1 -- the small molecule itself forms the glue interface. Unlike the
CRBN/GSPT1 system, the glue contact is ligand-mediated, so a CDK12+DDB1 receptor should
actually reward a productive arm.

The asymmetric unit holds THREE copies; we keep copy 1 (chains A/B/C) and carve receptors:

    Tier 1 : CDK12 alone               -> 6TD3_tier1_CDK12.pdb       (ATP-pocket affinity only)
    Tier 2 : CDK12 + DDB1              -> 6TD3_tier2_CDK12_DDB1.pdb  (glue-relevant composite)
    Tier 3 : CDK12 + DDB1 + cyclin K  -> 6TD3_tier3_full.pdb        (full native context)

The native CR8 is always removed from the receptors and written separately as the redock
reference (crystal_RC8.pdb). Phospho-Thr (TPO) on CDK12's activation loop is a modified
residue and is RETAINED as part of the protein.

Chain map (auth ids), copy 1:
    entity 1  DDB1                      copy1=A   (copy2=D, copy3=G)
    entity 2  CDK12 kinase domain       copy1=B   (carries TPO + RC8)
    entity 3  cyclin K                  copy1=C
    entity 4  CR8 glue (RC8)            on the CDK12 chain (B), resseq 1101

Reuses the dependency-free mmCIF parser / PDB writer from clean.py.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clean  # reuse parse_atom_site, pdb_atom_record, _blank, MODEL_NUM machinery

INPUT_CIF = os.path.join(HERE, "..", "..", "data", "models", "6TD3.cif")
OUT_DIR = os.path.join(HERE, "..", "..", "data", "models")

# Copy 1 chains (auth ids). Switch to D/E/F or G/H/I for the other copies.
CDK12_CHAIN = "B"
DDB1_CHAIN = "A"
CYCK_CHAIN = "C"

GLUE_RESNAME = "RC8"  # co-crystallised CR8 glue -- dropped from receptors, written separately
MODEL_NUM = "1"

# (label, output filename, set of auth chains)
TIERS = [
    ("Tier 1: CDK12", "6TD3_tier1_CDK12.pdb", {CDK12_CHAIN}),
    ("Tier 2: CDK12 + DDB1", "6TD3_tier2_CDK12_DDB1.pdb", {CDK12_CHAIN, DDB1_CHAIN}),
    (
        "Tier 3: CDK12 + DDB1 + cyclinK",
        "6TD3_tier3_full.pdb",
        {CDK12_CHAIN, DDB1_CHAIN, CYCK_CHAIN},
    ),
]

LIGAND_OUT = "crystal_RC8.pdb"
KEEP_HYDROGENS = True


def g(row, cols, key):
    return row[cols[key]]


def keep_receptor(row, cols, chains):
    if g(row, cols, "pdbx_PDB_model_num") != MODEL_NUM:
        return False
    if g(row, cols, "auth_asym_id") not in chains:
        return False
    if g(row, cols, "auth_comp_id") == GLUE_RESNAME:  # never dock against the bound glue
        return False
    if not KEEP_HYDROGENS and g(row, cols, "type_symbol").strip().upper() == "H":
        return False
    altloc = clean._blank(g(row, cols, "label_alt_id"))
    return altloc in ("", "A")


def keep_ligand(row, cols):
    if g(row, cols, "pdbx_PDB_model_num") != MODEL_NUM:
        return False
    if g(row, cols, "auth_comp_id") != GLUE_RESNAME:
        return False
    if g(row, cols, "auth_asym_id") != CDK12_CHAIN:  # the copy-1 CR8 only
        return False
    altloc = clean._blank(g(row, cols, "label_alt_id"))
    return altloc in ("", "A")


def write_pdb(out_path, kept_rows, cols):
    serial = 0
    prev_chain = None
    lines = []
    for row in kept_rows:
        chain = g(row, cols, "auth_asym_id")
        if prev_chain is not None and chain != prev_chain:
            lines.append("TER")
        serial += 1
        element = g(row, cols, "type_symbol")
        lines.append(clean.pdb_atom_record(serial, row, cols, element))
        prev_chain = chain
    lines.append("TER")
    lines.append("END")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return serial


def main():
    if not os.path.isfile(INPUT_CIF):
        sys.exit("Input not found: %s" % INPUT_CIF)
    cols, rows = clean.parse_atom_site(INPUT_CIF)
    print("Parsed %d atom rows from 6TD3.cif" % len(rows))
    print(
        "Config: CDK12=%s  DDB1=%s  cyclinK=%s  drop_glue=%s\n"
        % (CDK12_CHAIN, DDB1_CHAIN, CYCK_CHAIN, GLUE_RESNAME)
    )

    for label, fname, chains in TIERS:
        kept = [r for r in rows if keep_receptor(r, cols, chains)]
        n = write_pdb(os.path.join(OUT_DIR, fname), kept, cols)
        print("%-32s chains=%-8s -> %s  (%d atoms)" % (label, ",".join(sorted(chains)), fname, n))

    lig = [r for r in rows if keep_ligand(r, cols)]
    n = write_pdb(os.path.join(OUT_DIR, LIGAND_OUT), lig, cols)
    print("%-32s chain=%-9s -> %s  (%d atoms)" % ("Native ligand CR8", CDK12_CHAIN, LIGAND_OUT, n))


if __name__ == "__main__":
    main()
