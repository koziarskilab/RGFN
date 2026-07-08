#!/usr/bin/env python3
"""Central validation for the 6TD3 / CR8 glue system: can gnina SAMPLE the native pose?

CRBN failed here -- blind docking never sampled CC-885's deep tri-Trp-cage insertion. CR8 binds
the CDK12 ATP pocket (deep, druggable), so we expect blind/box docking to recover the native pose
without the warhead-anchoring trick. Two checks against the CDK12+DDB1 (Tier 2) receptor:

  1. NATIVE recognition: minimize the crystal CR8 pose in place -> does the score recognise it?
  2. BLIND redock:       build CR8 from SMILES, dock into a box around the pocket -> can it find
                         the native pose? (best-pose in-place RMSD to crystal, no superposition)
"""
import os
import subprocess

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
GNINA = os.environ.get(
    "GNINA", "/scratch/markymoo/gnina/run_gnina.sh"
)  # gnina launcher; override via $GNINA
RECEPTOR = os.path.join(HERE, "6TD3_tier2.pdbqt")
CRYSTAL = os.path.join(HERE, "crystal_RC8.pdb")
# CR8: (2R)-2-({9-isopropyl-6-[(4-(pyridin-2-yl)benzyl)amino]-9H-purin-2-yl}amino)butan-1-ol
CR8 = "CC[C@@H](CO)Nc1nc(NCc2ccc(-c3ccccn3)cc2)c2ncn(C(C)C)c2n1"


def crystal_ref():
    """Native CR8 heavy-atom pose with correct bond orders (template from SMILES).

    RDKit's PDB bond perception mis-valences CR8, so go through obabel (cleaner connectivity)
    and re-impose the SMILES template's bond orders.
    """
    nat_sdf = os.path.join(HERE, "crystal_RC8.sdf")
    if not os.path.exists(nat_sdf):
        subprocess.run(["obabel", CRYSTAL, "-O", nat_sdf], capture_output=True, text=True)
    tmpl = Chem.MolFromSmiles(CR8)
    cry = Chem.SDMolSupplier(nat_sdf, removeHs=True, sanitize=True)[0]
    return AllChem.AssignBondOrdersFromTemplate(tmpl, cry)


def embed(smiles, seed=0xC0FFEE):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def run_gnina(args):
    subprocess.run([GNINA] + args, capture_output=True, text=True, check=True)


def read_poses(sdf):
    out = []
    for m in Chem.SDMolSupplier(sdf, removeHs=True, sanitize=True):
        if m is None:
            continue
        gp = lambda k: float(m.GetProp(k)) if m.HasProp(k) else float("nan")
        out.append(
            {
                "mol": m,
                "vina": gp("minimizedAffinity"),
                "cnnsc": gp("CNNscore"),
                "cnnaff": gp("CNNaffinity"),
            }
        )
    return out


def rms_to_native(pose, ref):
    try:
        return rdMolAlign.CalcRMS(pose, ref)  # in-place, symmetry-minimised, NO superposition
    except Exception:
        return float("nan")


def main():
    ref = crystal_ref()
    print(f"native CR8: {ref.GetNumAtoms()} heavy atoms\n")

    # 1) native recognition -- minimize crystal pose in place
    nat_sdf = os.path.join(HERE, "crystal_RC8.sdf")
    subprocess.run(["obabel", CRYSTAL, "-O", nat_sdf], capture_output=True, text=True)
    nat_min = os.path.join(HERE, "_native_min.sdf")
    run_gnina(["-r", RECEPTOR, "-l", nat_sdf, "--minimize", "-o", nat_min])
    nm = read_poses(nat_min)
    if nm:
        p = nm[0]
        print(
            f"[native in-place min]  Vina {p['vina']:7.2f} | CNNscore {p['cnnsc']:.3f} | "
            f"CNNaff {p['cnnaff']:.3f} | drift {rms_to_native(p['mol'], ref):.2f} A"
        )

    # 2) blind redock from a fresh conformer, box around the pocket
    lig = embed(CR8)
    lig_sdf = os.path.join(HERE, "_cr8_embed.sdf")
    w = Chem.SDWriter(lig_sdf)
    w.write(lig)
    w.close()
    dock = os.path.join(HERE, "_cr8_docked.sdf")
    run_gnina(
        [
            "-r",
            RECEPTOR,
            "-l",
            lig_sdf,
            "--autobox_ligand",
            CRYSTAL,
            "--autobox_add",
            "4",
            "--exhaustiveness",
            "32",
            "--num_modes",
            "20",
            "--seed",
            "42",
            "-o",
            dock,
        ]
    )
    poses = read_poses(dock)
    poses_by_rms = sorted(poses, key=lambda p: rms_to_native(p["mol"], ref))
    print(f"\n[blind redock]  {len(poses)} poses")
    print(f"{'rank':>4} {'Vina':>8} {'CNNscore':>9} {'CNNaff':>8} {'RMSD_native':>12}")
    for i, p in enumerate(poses[:10], 1):  # ranked by gnina (Vina) order
        print(
            f"{i:>4} {p['vina']:>8.2f} {p['cnnsc']:>9.3f} {p['cnnaff']:>8.3f} "
            f"{rms_to_native(p['mol'], ref):>12.2f}"
        )
    if poses_by_rms:
        b = poses_by_rms[0]
        print(
            f"\nBEST-RMSD pose: {rms_to_native(b['mol'], ref):.2f} A  "
            f"(Vina {b['vina']:.2f}, CNNaff {b['cnnaff']:.3f})"
        )
        print(
            "=> SAMPLING WORKS"
            if rms_to_native(b["mol"], ref) < 2.0
            else "=> sampling still struggles (best > 2 A)"
        )


if __name__ == "__main__":
    main()
