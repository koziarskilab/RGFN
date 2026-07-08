#!/usr/bin/env python3
"""Warhead-anchored CRBN docking oracle prototype.

Every CRBN molecular glue shares the glutarimide (IMiD) warhead, which binds the
same tri-tryptophan cage. Instead of a blind global search (which never samples
the deep cage insertion), we PIN the glutarimide onto its known crystal pose and
only let the variable arm relax. This removes the sampling bottleneck and yields
orientation-correct, native-consistent scores.

Pipeline per molecule (SMILES):
  1. confirm a glutarimide substructure (guaranteed by warhead-constrained gen)
  2. RDKit ConstrainedEmbed: build a 3D conformer with the glutarimide atoms
     locked onto the crystal cage coordinates (extracted from crystal_85C.pdb)
  3. write SDF -> gnina --local_only (local opt + CNN rescore, no global search)
  4. report minimizedAffinity / CNNscore / CNNaffinity + warhead drift
"""
import os
import subprocess

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GNINA = os.environ.get(
    "GNINA", "/scratch/markymoo/gnina/run_gnina.sh"
)  # gnina launcher; override via $GNINA
RECEPTOR = os.path.join(HERE, "5HXB_tier2.pdbqt")
CRYSTAL = os.path.join(HERE, "crystal_85C.pdb")
CC885 = "c1c(CNC(=O)Nc2cc(Cl)c(C)cc2)cc2c(c1)C(=O)N(C2)C1C(=O)NC(=O)CC1"
GLUT_SMARTS = Chem.MolFromSmarts("C1CCC(=O)NC1=O")  # glutarimide (piperidine-2,6-dione)

# Pass B (flexible anchor): the warhead is no longer hard-frozen. It is started in the
# crystal cage (rigid alignment) and allowed to WIGGLE within MAX_DISPL during co-relaxation
# and gnina minimization; any output pose whose warhead escapes the cage beyond DRIFT_MAX is
# rejected (the tether). ANCHOR_PASS=A restores the original hard-freeze behaviour.
PASS = os.environ.get("ANCHOR_PASS", "B").upper()
DRIFT_MAX = 2.5  # A; output poses whose warhead drifts past this are off-tether -> rejected
RESTRAINT_K = 25.0  # kcal/mol/A^2 flat-bottom spring holding the warhead near the cage
MAX_DISPL = 1.5  # A; warhead is free to move within this radius (the "wiggle")


def crystal_glutarimide_core():
    """Glutarimide submol carrying the crystal 3D coordinates (the cage anchor)."""
    tmpl = Chem.MolFromSmiles(CC885)
    cry = Chem.MolFromPDBFile(CRYSTAL, removeHs=True, sanitize=False)
    cry = AllChem.AssignBondOrdersFromTemplate(tmpl, cry)
    match = set(cry.GetSubstructMatch(GLUT_SMARTS))
    core = Chem.RWMol(cry)
    for idx in sorted(set(range(cry.GetNumAtoms())) - match, reverse=True):
        core.RemoveAtom(idx)
    core = core.GetMol()
    Chem.SanitizeMol(core)
    return core


def anchored_conformers(smiles, core, n_confs=20, seed=0xC0FFEE, relax=True):
    """Ensemble of 3D conformers with the glutarimide pinned to the crystal cage.

    coordMap locks the warhead atoms to crystal coords during embedding while the
    arm is sampled freely -> a conformer ensemble that all share the native anchor.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if not mol.HasSubstructMatch(GLUT_SMARTS):
        raise ValueError("no glutarimide warhead -> not anchorable")
    mm, cm = mol.GetSubstructMatch(GLUT_SMARTS), core.GetSubstructMatch(GLUT_SMARTS)
    cconf = core.GetConformer()
    coord_map = {mm[i]: cconf.GetAtomPosition(cm[i]) for i in range(len(cm))}
    cids = AllChem.EmbedMultipleConfs(
        mol, numConfs=n_confs, coordMap=coord_map, randomSeed=seed, useRandomCoords=True
    )
    if not cids:  # fall back to single constrained embed if multi-conf embedding fails
        AllChem.ConstrainedEmbed(mol, core, randomseed=seed)
    if (
        relax
    ):  # MMFF-relax the arm with the warhead atoms held fixed -> removes intramolecular strain
        props = AllChem.MMFFGetMoleculeProperties(mol)
        for c in mol.GetConformers():
            ff = (
                AllChem.MMFFGetMoleculeForceField(mol, props, confId=c.GetId())
                if props
                else AllChem.UFFGetMoleculeForceField(mol, confId=c.GetId())
            )
            if ff is None:
                continue
            for idx in mm:  # pin every warhead atom to the cage
                ff.AddFixedPoint(idx)
            ff.Minimize(maxIts=300)
    return mol


def anchored_conformers_flex(
    smiles,
    core,
    n_confs=100,
    seed=0xC0FFEE,
    restraint_k=RESTRAINT_K,
    max_displ=MAX_DISPL,
    embed=os.environ.get("ANCHOR_EMBED", "coordmap"),
):
    """Pass B: flexible-anchor conformer ensemble.

    Unlike pass A (warhead pinned to exact crystal coords, arm relaxed against a HARD-frozen
    warhead), here the warhead is allowed to WIGGLE:
      1. embed diverse conformers with the warhead placed in the cage --
         'coordmap' : ETKDG with the warhead atoms constrained to the crystal coords (cage-aware
                      arm sampling, like pass A) -- default, best for small/rigid IMiDs;
         'free'     : free ETKDGv3 then rigid-align the glutarimide onto the cage.
      2. MMFF co-relax the WHOLE molecule with only a soft flat-bottom positional restraint on the
         warhead atoms -> warhead + arm relieve strain TOGETHER, warhead free within max_displ.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if not mol.HasSubstructMatch(GLUT_SMARTS):
        raise ValueError("no glutarimide warhead -> not anchorable")
    mm, cm = mol.GetSubstructMatch(GLUT_SMARTS), core.GetSubstructMatch(GLUT_SMARTS)
    cconf = core.GetConformer()
    atom_map = [(mm[i], cm[i]) for i in range(len(cm))]  # (mol idx, core idx)

    if embed == "coordmap":  # warhead constrained to cage coords during embedding (kwarg form)
        coord_map = {mm[i]: cconf.GetAtomPosition(cm[i]) for i in range(len(cm))}
        cids = list(
            AllChem.EmbedMultipleConfs(
                mol, numConfs=n_confs, coordMap=coord_map, randomSeed=seed, useRandomCoords=True
            )
        )
    else:  # free ETKDGv3 embed, aligned to the cage below
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 1
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))
    if not cids:  # degenerate fallback
        AllChem.ConstrainedEmbed(mol, core, randomseed=seed)
        cids = [c.GetId() for c in mol.GetConformers()]

    props = AllChem.MMFFGetMoleculeProperties(mol)
    for cid in cids:
        if embed != "coordmap":  # 'free' confs aren't in the cage frame yet
            AllChem.AlignMol(mol, core, prbCid=cid, atomMap=atom_map)
        ff = (
            AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if props
            else AllChem.UFFGetMoleculeForceField(mol, confId=cid)
        )
        if ff is None:
            continue
        add_pc = ff.MMFFAddPositionConstraint if props else ff.UFFAddPositionConstraint
        for idx in mm:  # soft tether: free within max_displ, spring beyond
            add_pc(idx, max_displ, restraint_k)
        ff.Minimize(maxIts=500)
    return mol


def build_conformers(smiles, core, n_confs=100, seed=0xC0FFEE):
    """Dispatch to the active anchoring strategy (ANCHOR_PASS env: 'A' hard-freeze, 'B' flexible)."""
    if PASS == "A":
        return anchored_conformers(smiles, core, n_confs=n_confs, seed=seed, relax=True)
    return anchored_conformers_flex(smiles, core, n_confs=n_confs, seed=seed)


def pose_drift(mol, core):
    """Symmetry-aware warhead RMSD of a (possibly unsanitized) gnina output pose vs the cage."""
    m = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    return warhead_drift(m, core, 0)


def warhead_drift(mol, core, conf_id=0):
    """Symmetry-aware RMSD of the molecule's glutarimide vs the crystal anchor."""
    cm = core.GetSubstructMatch(GLUT_SMARTS)
    Q = np.array([list(core.GetConformer().GetAtomPosition(i)) for i in cm])
    mconf = mol.GetConformer(conf_id)
    best = 1e9
    for mm in mol.GetSubstructMatches(GLUT_SMARTS, uniquify=False):  # all symmetric maps
        P = np.array([list(mconf.GetAtomPosition(i)) for i in mm])
        best = min(best, float(np.sqrt(((P - Q) ** 2).sum(1).mean())))
    return best


def gnina_local(sdf_in, sdf_out, core=None, drift_max=DRIFT_MAX, select="vina"):
    """Local opt + CNN rescore of EVERY conformer from the anchored start (no global search).

    select='vina'  -> pick the pose with the lowest (best) minimizedAffinity. This is
                      CLASH-AWARE: a sterically bad arm has a large positive Vina energy,
                      so we never report a clashing pose just because its CNN looks fine.
    select='cnnaff'-> pick the pose with the highest CNNaffinity (original behaviour).

    Tether (pass B): when `core` is given, every output pose's warhead drift from the crystal
    cage is measured and poses past `drift_max` are rejected before selection -- so we never
    accept a pose whose warhead wiggled its way out of the cage. Falls back to the full pool
    only if NO pose stays on-tether (so the caller still gets a record to flag).
    Returns the chosen pose's record plus diagnostics (n_poses, n_tethered, n_nonclash).
    """
    subprocess.run(
        [GNINA, "-r", RECEPTOR, "-l", sdf_in, "--minimize", "-o", sdf_out],
        capture_output=True,
        text=True,
        check=True,
    )
    poses = []
    for m in Chem.SDMolSupplier(sdf_out, removeHs=False, sanitize=False):
        if m is None:
            continue
        g = lambda k: float(m.GetProp(k)) if m.HasProp(k) else float("nan")
        rec = {
            "minimizedAffinity": g("minimizedAffinity"),
            "CNNscore": g("CNNscore"),
            "CNNaffinity": g("CNNaffinity"),
            "mol": m,
            "drift": float("nan"),
        }
        if core is not None:
            try:
                rec["drift"] = pose_drift(m, core)
            except Exception:
                pass
        poses.append(rec)
    if not poses:
        return None
    on_tether = [p for p in poses if p["drift"] <= drift_max] if core is not None else poses
    pool = on_tether if on_tether else poses
    key = (lambda r: r["minimizedAffinity"]) if select == "vina" else (lambda r: -r["CNNaffinity"])
    best = min(pool, key=key)
    best["n_poses"] = len(poses)
    best["n_tethered"] = len(on_tether)
    best["n_nonclash"] = sum(1 for p in pool if p["minimizedAffinity"] < 0)
    best["on_tether"] = bool(on_tether)
    return best


def score(name, smiles, core, n_confs=100):
    mol = build_conformers(smiles, core, n_confs=n_confs)
    sdf_in = os.path.join(HERE, f"_anchor_{name}.sdf")
    sdf_out = os.path.join(HERE, f"_anchor_{name}_min.sdf")
    w = Chem.SDWriter(sdf_in)
    for c in mol.GetConformers():
        w.write(mol, confId=c.GetId())
    w.close()
    res = gnina_local(sdf_in, sdf_out, core=core)
    print(
        f"{name:>14} | {res['n_poses']:3d} confs {res['n_tethered']:3d} tether "
        f"{res['n_nonclash']:3d} fit | out-drift {res['drift']:.2f} A | "
        f"minAff {res['minimizedAffinity']:7.2f} | "
        f"CNNscore {res['CNNscore']:.3f} | CNNaff {res['CNNaffinity']:.3f}"
    )
    return res


if __name__ == "__main__":
    core = crystal_glutarimide_core()
    print(f"PASS {PASS} | anchor: crystal glutarimide ({core.GetNumAtoms()} atoms) -> CRBN cage\n")
    tests = [
        ("CC-885", CC885),  # native: must recover deep well
        ("lenalidomide", "O=C1CCC(N2Cc3cccc(N)c3C2=O)C(=O)N1"),  # different warhead-bearing glue
        ("pomalidomide", "Nc1cccc2c1C(=O)N(C1CCC(=O)NC1=O)C2=O"),  # IMiD we blind-docked earlier
    ]
    print(
        f"{'molecule':>14} | {'confs/tether/fit':>20} | {'out-drift':>9} | "
        f"{'Vina min':>10} | {'CNN pose':>10} | {'CNN aff':>8}"
    )
    print("-" * 92)
    for name, smi in tests:
        try:
            score(name, smi, core)
        except Exception as e:
            print(f"{name:>14} | FAILED: {e}")
