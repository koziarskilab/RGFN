#!/usr/bin/env python3
"""6TD3/CR8 redock that RETAINS EVERY POSE, for the pose-selection ablation.

This is a variant of ``docking_6td3/dock_cluster.py`` with one change: instead of
collapsing each molecule to a single "best" pose (max CNNscore) before the Tier-1
rescore, it keeps **all** ``num_modes`` docked poses and ``--score_only``s *every*
pose against Tier 1. The output is therefore a PER-POSE CSV, so the pose-selection
rule (CNNscore vs. Vina) can be applied offline by ``analyze_pose_selection.py``
without re-docking.

Why: ``Logs/006`` ranked the six scoring signals (Vina/CNN x Tier1/Tier2/diff)
but held pose selection FIXED at max-CNNscore. The open question is whether
picking the top-Vina pose instead would degrade the validated Vina dT2-T1
differential (AUROC 0.946). Answering it needs both candidate poses' Tier-1
scores for the same molecule — which the production pipeline never stores.

Everything else is identical to dock_cluster.py on purpose (same gnina flags,
embedding, autobox, exhaustiveness, num_modes, seed, sharding) so the CNN-selected
arm of this run reproduces the production pipeline as an in-run control. The two
arms (CNN-selected vs. Vina-selected) come from the SAME docked poses, so the
comparison is paired and immune to docking stochasticity between jobs.

Outputs go to $OUTDIR (scratch); $HOME is read-only on compute nodes.
"""
import csv
import multiprocessing as mp
import os
import subprocess
import time

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
DOCK6TD3 = os.path.join(
    HERE, "..", "..", "oracle_validation", "docking_6td3"
)  # reuse the prepared receptors/inputs
GNINA = os.environ.get(
    "GNINA", "/scratch/markymoo/gnina/run_gnina.sh"
)  # gnina launcher; override via $GNINA
TIER2 = os.path.join(DOCK6TD3, "6TD3_tier2.pdbqt")
TIER1 = os.path.join(DOCK6TD3, "6TD3_tier1.pdbqt")
CRYSTAL = os.path.join(DOCK6TD3, "crystal_RC8.pdb")
KNOWN_CSV = os.path.join(
    HERE, "..", "..", "..", "data", "validation-molecules", "DDB1_CDK12_Glues.csv"
)
DECOY_SMI = os.path.join(DOCK6TD3, "decoys_cdk.smiles")

OUTDIR = os.environ.get("OUTDIR", os.path.join(HERE, "cluster_out"))
WORK = os.environ.get("WORK", "/scratch/markymoo/dock_6td3_allposes")
N_GPU = int(os.environ.get("N_GPU", "4"))
PROCS_PER_GPU = int(os.environ.get("PROCS_PER_GPU", "4"))
WORKERS = N_GPU * PROCS_PER_GPU
TOTAL_CPU = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 128))
CPU_PER = max(1, TOTAL_CPU // WORKERS)
EXH = int(os.environ.get("EXH", "16"))
NUM_MODES = int(os.environ.get("NUM_MODES", "9"))
SEED = 42


def largest_frag(m):
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return max(fr, key=lambda f: f.GetNumHeavyAtoms()) if len(fr) > 1 else m


def load_all():
    """Return [(idx, cid, set_label, canonical_smiles)] for known + decoy, deduped within set.

    Identical loader to dock_cluster.load_all so the molecule set matches entry 002.
    """
    mols, idx = [], 0
    seen = set()
    for r in csv.DictReader(open(KNOWN_CSV)):
        m = Chem.MolFromSmiles(r["SMILES"])
        if m is None:
            continue
        smi = Chem.MolToSmiles(largest_frag(m))
        if ("known", smi) in seen:
            continue
        seen.add(("known", smi))
        idx += 1
        mols.append((idx, r.get("DATAID") or r.get("Name") or str(idx), "known", smi))
    with open(DECOY_SMI) as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split("\t")
            m = Chem.MolFromSmiles(p[0])
            if m is None:
                continue
            smi = Chem.MolToSmiles(largest_frag(m))
            if ("decoy", smi) in seen:
                continue
            seen.add(("decoy", smi))
            idx += 1
            mols.append((idx, p[1] if len(p) > 1 else str(idx), "decoy", smi))
    cap = int(os.environ.get("MAX", "0"))
    if cap:  # smoke test: keep a few from each set
        k = [m for m in mols if m[2] == "known"][:cap]
        d = [m for m in mols if m[2] == "decoy"][:cap]
        mols = k + d
    return mols


def embed(task):
    idx, cid, setl, smi = task
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        if AllChem.EmbedMolecule(m, randomSeed=SEED) != 0:
            AllChem.EmbedMolecule(m, randomSeed=SEED, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(m)
        m.SetProp("_Name", str(idx))
        return (idx, Chem.MolToMolBlock(m))
    except Exception:
        return (idx, None)


def _poses(sdf):
    """idx -> list of pose dicts in gnina output order (rank 1..N = gnina's own ranking)."""
    groups = {}
    for m in Chem.SDMolSupplier(sdf, removeHs=False, sanitize=False):
        if m is None:
            continue
        idx = m.GetProp("_Name") if m.HasProp("_Name") else None
        g = lambda k: float(m.GetProp(k)) if m.HasProp(k) else float("nan")
        groups.setdefault(idx, []).append(
            {
                "mol": m,
                "vina": g("minimizedAffinity"),
                "cnnsc": g("CNNscore"),
                "cnnaff": g("CNNaffinity"),
            }
        )
    return groups


def _score_only_stream(receptor, sdf, gpu_env):
    """Return ordered list of (affinity, cnnaff) for each ligand in `sdf`."""
    r = subprocess.run(
        [GNINA, "-r", receptor, "-l", sdf, "--score_only"],
        capture_output=True,
        text=True,
        check=True,
        env=gpu_env,
    )
    affs, caffs = [], []
    for line in r.stdout.splitlines():
        if line.startswith("Affinity:"):
            affs.append(float(line.split()[1]))
        elif line.startswith("CNNaffinity:"):
            caffs.append(float(line.split()[1]))
    return list(zip(affs, caffs))


def dock_shard(arg):
    """Dock one shard into Tier 2, then Tier-1 --score_only EVERY pose (not just the best).

    Emits one row per (molecule, pose). The Tier-1 scores are matched back to poses
    by writing all poses to a single SDF in a recorded (idx, pose_rank) order and
    relying on --score_only preserving input order.
    """
    shard_id, gpu, idxs, meta = arg
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    shard_sdf = os.path.join(WORK, f"shard{shard_id:02d}.sdf")
    docked = os.path.join(WORK, f"shard{shard_id:02d}_docked.sdf")
    allposes_sdf = os.path.join(WORK, f"shard{shard_id:02d}_allposes.sdf")
    rows = []
    try:
        subprocess.run(
            [
                GNINA,
                "-r",
                TIER2,
                "-l",
                shard_sdf,
                "--autobox_ligand",
                CRYSTAL,
                "--autobox_add",
                "4",
                "--exhaustiveness",
                str(EXH),
                "--num_modes",
                str(NUM_MODES),
                "--cpu",
                str(CPU_PER),
                "--seed",
                str(SEED),
                "-o",
                docked,
            ],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        groups = _poses(docked)

        # Write EVERY pose to one SDF, recording (idx, pose_rank, t2 metrics) in write order.
        order = []  # parallel to the SDF; (idx, pose_rank, vina_t2, cnnsc_t2, cnnaff_t2)
        w = Chem.SDWriter(allposes_sdf)
        for idx in idxs:
            ps = groups.get(str(idx))
            if not ps:
                continue
            for rank, p in enumerate(ps, start=1):  # gnina output order = its own ranking
                w.write(p["mol"])
                order.append((idx, rank, p["vina"], p["cnnsc"], p["cnnaff"]))
        w.close()

        t1 = _score_only_stream(TIER1, allposes_sdf, env) if order else []

        for k, (idx, rank, v2, sc2, c2) in enumerate(order):
            cid, setl, smi = meta[idx]
            v1, c1 = t1[k] if k < len(t1) else (float("nan"), float("nan"))
            rows.append(
                [
                    idx,
                    cid,
                    setl,
                    smi,
                    "ok",
                    len(groups.get(str(idx), [])),
                    rank,
                    f"{v2:.3f}",
                    f"{sc2:.4f}",
                    f"{c2:.4f}",
                    f"{v1:.3f}",
                    f"{c1:.4f}",
                ]
            )
        # record molecules that produced no pose (one row, pose_rank 0)
        got = {idx for idx, *_ in order}
        for idx in idxs:
            if idx not in got:
                cid, setl, smi = meta[idx]
                rows.append([idx, cid, setl, smi, "no_pose", 0, 0, "", "", "", "", ""])
        return rows
    except Exception as e:
        for idx in idxs:
            cid, setl, smi = meta[idx]
            rows.append([idx, cid, setl, smi, f"shard_fail:{type(e).__name__}", 0, 0, *[""] * 5])
        return rows


HEADER = [
    "idx",
    "id",
    "set",
    "smiles",
    "status",
    "n_poses",
    "pose_rank",  # 1 = gnina's top-ranked pose
    "vina_t2",
    "cnnsc_t2",
    "cnnaff_t2",
    "vina_t1",
    "cnnaff_t1",
]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    t0 = time.time()
    mols = load_all()
    nk = sum(1 for x in mols if x[2] == "known")
    nd = len(mols) - nk
    print(f"loaded {len(mols)} molecules (known={nk}, decoy={nd})", flush=True)
    print(
        f"node: WORKERS={WORKERS} ({N_GPU} GPU x {PROCS_PER_GPU}), CPU_PER={CPU_PER}, "
        f"EXH={EXH}, modes={NUM_MODES}  [ALL POSES retained]",
        flush=True,
    )

    print("phase 1: embedding (parallel)...", flush=True)
    with mp.Pool(min(TOTAL_CPU, 64)) as pool:
        embs = pool.map(embed, mols)
    blocks = {idx: blk for idx, blk in embs if blk is not None}
    meta = {idx: (cid, setl, smi) for (idx, cid, setl, smi) in mols}
    print(
        f"phase 1 done: {len(blocks)}/{len(mols)} embedded ({(time.time()-t0)/60:.1f} min)",
        flush=True,
    )

    # shard round-robin and write shard SDFs
    shards = [[] for _ in range(WORKERS)]
    for j, idx in enumerate(sorted(blocks)):
        shards[j % WORKERS].append(idx)
    for s, idxs in enumerate(shards):
        with open(os.path.join(WORK, f"shard{s:02d}.sdf"), "w") as fh:
            fh.write("".join(blocks[idx] + "$$$$\n" for idx in idxs))
    args = [(s, s % N_GPU, idxs, meta) for s, idxs in enumerate(shards) if idxs]

    print(f"phase 2: docking {len(args)} shards across {N_GPU} GPUs (all poses)...", flush=True)
    t1 = time.time()
    with mp.Pool(WORKERS) as pool:
        results = pool.map(dock_shard, args)
    allrows = [r for shard in results for r in shard]
    print(f"phase 2 done ({(time.time()-t1)/60:.1f} min)", flush=True)

    # split by set -> per-pose CSVs
    for setl, fname in (("known", "known_allposes.csv"), ("decoy", "decoy_allposes.csv")):
        rows = sorted((r for r in allrows if r[2] == setl), key=lambda r: (r[0], r[6]))
        with open(os.path.join(OUTDIR, fname), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(HEADER)
            w.writerows(rows)
        n_mol = len({r[0] for r in rows if r[4] == "ok"})
        n_pose = sum(1 for r in rows if r[4] == "ok")
        print(f"  {setl}: {n_mol} molecules / {n_pose} poses -> {fname}", flush=True)
    print(f"DONE total {(time.time()-t0)/60:.1f} min -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
