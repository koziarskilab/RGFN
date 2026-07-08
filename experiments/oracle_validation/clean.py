#!/usr/bin/env python3
"""Clean the 5HXB ternary-complex structure into two docking-ready receptor tiers.

5HXB is the CRBN-DDB1-GSPT1 ternary complex (PDB 5HXB) captured with the
molecular glue CC-885 (ligand residue ``85C``). The asymmetric unit holds *two*
copies of the complex; we keep one self-contained copy (copy 1) and carve it
into two receptors for docking molecular glues with AutoDock Vina:

    Tier 1 : CRBN alone               -> 5HXB_tier1_CRBN.pdb
    Tier 2 : CRBN + GSPT1 (only)      -> 5HXB_tier2_CRBN_GSPT1.pdb

The structural Zn(2+) ion on CRBN is retained in both tiers (it stabilises the
CULT/thalidomide-binding domain that forms the glue pocket). The co-crystallised
glue ``85C`` is always removed -- it sits in the pocket we want to dock into.

Chain map for 5HXB (auth ids, i.e. the chain letters Vina/PDB use):

    entity 1  GSPT1 (eRF3a / ERF3A)   copy1=A   copy2=X
    entity 2  DDB1                     copy1=B   copy2=Y
    entity 3  CRBN (cereblon)          copy1=C   copy2=Z
    entity 4  Zn(2+) ion               on the CRBN chain (C / Z)
    entity 5  glue CC-885 (85C)        on the CRBN chain (C / Z)

This script is dependency-free: it parses the mmCIF ``_atom_site`` loop directly
and writes standard PDB records, so it runs under a bare Python install.
"""

from __future__ import annotations

import os
import shlex
import sys

# --------------------------------------------------------------------------- #
# Configuration -- edit here to retarget chains / copies / what is kept.
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_CIF = os.path.join(HERE, "..", "..", "data", "models", "5HXB.cif")
OUT_DIR = os.path.join(HERE, "..", "..", "data", "models")

# Copy 1 chains (auth ids). Switch to X/Y/Z to use the second copy instead.
CRBN_CHAIN = "C"
GSPT1_CHAIN = "A"

GLUE_RESNAME = "85C"  # co-crystallised molecular glue -- always dropped
ZN_RESNAME = "ZN"  # structural zinc on the CRBN chain
KEEP_ZN = True  # retain the structural Zn(2+) ion in both tiers
KEEP_HYDROGENS = True  # receptor-prep tools (prepare_receptor) re-add polar H

MODEL_NUM = "1"  # X-ray: a single model; keep model 1 only

# (record name, output filename, set of auth chains to include)
TIERS = [
    ("Tier 1: CRBN", "5HXB_tier1_CRBN.pdb", {CRBN_CHAIN}),
    ("Tier 2: CRBN + GSPT1", "5HXB_tier2_CRBN_GSPT1.pdb", {CRBN_CHAIN, GSPT1_CHAIN}),
]

# --------------------------------------------------------------------------- #
# mmCIF atom_site parsing
# --------------------------------------------------------------------------- #


def parse_atom_site(path):
    """Yield one dict per ATOM/HETATM row of the mmCIF ``_atom_site`` loop.

    Returns (columns, rows) where ``columns`` maps the ``_atom_site.<field>``
    name (without prefix) to its index in each row.
    """
    with open(path) as fh:
        lines = fh.readlines()

    # Locate the loop_ whose first data names are _atom_site.*
    columns = {}
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() == "loop_":
            # collect the column header names that follow
            j = i + 1
            header = []
            while j < n and lines[j].lstrip().startswith("_"):
                header.append(lines[j].strip())
                j += 1
            if header and header[0].startswith("_atom_site."):
                for idx, name in enumerate(header):
                    columns[name.split(".", 1)[1]] = idx
                data_start = j
                break
            i = j
        else:
            i += 1
    else:
        raise RuntimeError("No _atom_site loop found in %s" % path)

    rows = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            break  # end of the loop block
        if stripped.startswith("_") or stripped == "loop_":
            break
        if not (stripped.startswith("ATOM") or stripped.startswith("HETATM")):
            break
        # shlex handles quoted atom names such as "O5'" correctly
        rows.append(shlex.split(stripped))

    return columns, rows


# --------------------------------------------------------------------------- #
# PDB writing
# --------------------------------------------------------------------------- #


def _blank(value):
    """mmCIF uses '?' and '.' for absent values; map them to ''."""
    return "" if value in ("?", ".") else value


def _format_atom_name(name, element):
    """Place the atom name in PDB columns 13-16 with element-aware alignment."""
    name = name.strip()
    if len(name) >= 4:
        return name[:4]
    # 1-letter elements get a leading space (e.g. " CA " carbon-alpha vs "CA  " calcium)
    if len(element.strip()) == 1:
        return (" " + name).ljust(4)
    return name.ljust(4)


def pdb_atom_record(serial, row, cols, element):
    g = lambda key: row[cols[key]]

    record = "ATOM  " if g("group_PDB") == "ATOM" else "HETATM"
    name = _format_atom_name(g("auth_atom_id"), element)
    resname = g("auth_comp_id")[:3].rjust(3)
    chain = (g("auth_asym_id") or " ")[0]
    resseq = int(g("auth_seq_id"))
    icode = _blank(g("pdbx_PDB_ins_code"))[:1]
    x, y, z = float(g("Cartn_x")), float(g("Cartn_y")), float(g("Cartn_z"))
    occ = float(g("occupancy"))
    bfac = float(g("B_iso_or_equiv"))
    elem = element.strip().rjust(2)

    return (
        "{rec:<6}{serial:>5} {name:<4}{alt:1}{resn:>3} {chain:1}{resseq:>4}{icode:1}   "
        "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}          {elem:>2}".format(
            rec=record,
            serial=serial % 100000,
            name=name,
            alt=" ",
            resn=resname,
            chain=chain,
            resseq=resseq,
            icode=icode,
            x=x,
            y=y,
            z=z,
            occ=occ,
            b=bfac,
            elem=elem,
        )
    )


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


def keep_row(row, cols, chains):
    g = lambda key: row[cols[key]]

    if g("pdbx_PDB_model_num") != MODEL_NUM:
        return False
    if g("auth_asym_id") not in chains:
        return False

    comp = g("auth_comp_id")
    if comp == GLUE_RESNAME:  # never dock against the bound glue
        return False
    if comp == ZN_RESNAME and not KEEP_ZN:
        return False
    if not KEEP_HYDROGENS and g("type_symbol").strip().upper() == "H":
        return False

    # collapse alternate locations: keep the primary conformer only
    altloc = _blank(g("label_alt_id"))
    if altloc not in ("", "A"):
        return False

    return True


def write_tier(label, out_path, chains, cols, rows):
    serial = 0
    prev_chain = None
    out_lines = []
    kept = 0
    for row in rows:
        if not keep_row(row, cols, chains):
            continue
        chain = row[cols["auth_asym_id"]]
        if prev_chain is not None and chain != prev_chain:
            out_lines.append("TER")
        serial += 1
        kept += 1
        element = row[cols["type_symbol"]]
        out_lines.append(pdb_atom_record(serial, row, cols, element))
        prev_chain = chain
    out_lines.append("TER")
    out_lines.append("END")

    with open(out_path, "w") as fh:
        fh.write("\n".join(out_lines) + "\n")

    return kept


def main():
    if not os.path.isfile(INPUT_CIF):
        sys.exit("Input not found: %s" % INPUT_CIF)

    cols, rows = parse_atom_site(INPUT_CIF)
    required = [
        "group_PDB",
        "type_symbol",
        "auth_atom_id",
        "label_alt_id",
        "auth_comp_id",
        "auth_asym_id",
        "auth_seq_id",
        "pdbx_PDB_ins_code",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "occupancy",
        "B_iso_or_equiv",
        "pdbx_PDB_model_num",
    ]
    missing = [c for c in required if c not in cols]
    if missing:
        sys.exit("mmCIF is missing expected columns: %s" % ", ".join(missing))

    print("Parsed %d atom rows from %s" % (len(rows), os.path.basename(INPUT_CIF)))
    print(
        "Config: CRBN=%s  GSPT1=%s  keep_Zn=%s  drop_glue=%s\n"
        % (CRBN_CHAIN, GSPT1_CHAIN, KEEP_ZN, GLUE_RESNAME)
    )

    for label, fname, chains in TIERS:
        out_path = os.path.join(OUT_DIR, fname)
        kept = write_tier(label, out_path, chains, cols, rows)
        print(
            "%-24s chains=%-8s -> %s  (%d atoms)" % (label, ",".join(sorted(chains)), fname, kept)
        )


if __name__ == "__main__":
    main()
