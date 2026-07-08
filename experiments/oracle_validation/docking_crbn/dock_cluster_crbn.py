#!/usr/bin/env python3
"""High-throughput CRBN/5HXB warhead-anchored glue docking for a multi-GPU debug node.

Mirrors the 6TD3 cluster run (dock_6td3/dock_cluster.py) so the two systems are directly
comparable, but uses the warhead-ANCHORED Pass-B pipeline (CRBN blind docking can't sample the
tri-Trp cage, so we pin the glutarimide and let the arm relax). Per molecule:
  1. build N flexible-anchor conformers (glutarimide pinned to the crystal cage, arm sampled),
  2. gnina --minimize all confs against Tier 2 (CRBN + GSPT1), pick the best ON-TETHER pose by
     clash-aware Vina,
  3. score that SAME pose against Tier 1 (CRBN only) -> GSPT1 differential (analog of 6TD3's DDB1
     bonus): how much the GSPT1 neosubstrate contact stabilises the pose.

Sharded across 4 GPUs (CUDA_VISIBLE_DEVICES per worker). Per-shard CSVs are checkpointed to
$OUTDIR as each shard finishes (survives a walltime kill); merged + split by set at the end.
Outputs -> $OUTDIR (scratch); $HOME is read-only on compute nodes.
"""
import csv
import multiprocessing as mp
import os
import subprocess
import sys
import time

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import anchor_dock as A  # build_conformers, crystal_glutarimide_core, pose_drift, GLUT_SMARTS, DRIFT_MAX, GNINA

GNINA = A.GNINA
TIER2 = os.path.join(HERE, "5HXB_tier2.pdbqt")  # CRBN + GSPT1
TIER1 = os.path.join(HERE, "5HXB_tier1_CRBN.pdbqt")  # CRBN only
KNOWN_CSV = os.path.join(
    HERE, "..", "..", "..", "data", "validation-molecules", "CRBN_GSPT1_Glues.csv"
)
DECOY_SMI = os.path.join(HERE, "decoys.smiles")

OUTDIR = os.environ.get("OUTDIR", os.path.join(HERE, "cluster_out_crbn"))
WORK = os.environ.get("WORK", "/scratch/markymoo/dock_crbn_cluster")
N_GPU = int(os.environ.get("N_GPU", "4"))
PROCS_PER_GPU = int(os.environ.get("PROCS_PER_GPU", "4"))
WORKERS = N_GPU * PROCS_PER_GPU
N_CONFS = int(os.environ.get("N_CONFS", "100"))
DRIFT_MAX = A.DRIFT_MAX


def largest_frag(m):
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return max(fr, key=lambda f: f.GetNumHeavyAtoms()) if len(fr) > 1 else m


def load_all():
    """[(idx, cid, set, smiles)] for glutarimide-bearing known glues + glutarimide decoys."""
    mols, idx, seen = [], 0, set()
    for r in csv.DictReader(open(KNOWN_CSV)):
        m = Chem.MolFromSmiles(r["SMILES"])
        if m is None:
            continue
        parent = largest_frag(m)
        if not parent.HasSubstructMatch(A.GLUT_SMARTS):  # anchored method needs the warhead
            continue
        smi = Chem.MolToSmiles(parent)
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
    if cap:
        mols = [m for m in mols if m[2] == "known"][:cap] + [m for m in mols if m[2] == "decoy"][
            :cap
        ]
    return mols


_CORE = None


def _init():
    global _CORE
    _CORE = A.crystal_glutarimide_core()


def embed(task):
    """Build N anchored flex confs, write one SDF (all confs titled with idx)."""
    idx, cid, setl, smi = task
    sdf = os.path.join(WORK, f"m{idx:05d}.sdf")
    try:
        mol = A.build_conformers(smi, _CORE, n_confs=N_CONFS)
        w = Chem.SDWriter(sdf)
        for c in mol.GetConformers():
            mol.SetProp("_Name", str(idx))
            w.write(mol, confId=c.GetId())
        w.close()
        return (idx, mol.GetNumConformers(), "embedded")
    except Exception as e:
        return (idx, 0, f"embed_fail:{type(e).__name__}")


def _grouped_poses(sdf, core):
    groups = {}
    for m in Chem.SDMolSupplier(sdf, removeHs=False, sanitize=False):
        if m is None:
            continue
        idx = m.GetProp("_Name") if m.HasProp("_Name") else None
        g = lambda k: float(m.GetProp(k)) if m.HasProp(k) else float("nan")
        try:
            drift = A.pose_drift(m, core)
        except Exception:
            drift = float("nan")
        groups.setdefault(idx, []).append(
            {
                "mol": m,
                "vina": g("minimizedAffinity"),
                "cnnsc": g("CNNscore"),
                "cnnaff": g("CNNaffinity"),
                "drift": drift,
            }
        )
    return groups


def _score_only_stream(receptor, sdf, env):
    r = subprocess.run(
        [GNINA, "-r", receptor, "-l", sdf, "--score_only"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    affs, caffs = [], []
    for line in r.stdout.splitlines():
        if line.startswith("Affinity:"):
            affs.append(float(line.split()[1]))
        elif line.startswith("CNNaffinity:"):
            caffs.append(float(line.split()[1]))
    return list(zip(affs, caffs))


def dock_shard(arg):
    shard_id, gpu, idxs, meta = arg
    _init()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    shard_sdf = os.path.join(WORK, f"shard{shard_id:02d}.sdf")
    minimized = os.path.join(WORK, f"shard{shard_id:02d}_min.sdf")
    best_sdf = os.path.join(WORK, f"shard{shard_id:02d}_best.sdf")
    out_csv = os.path.join(OUTDIR, f"shard{shard_id:02d}.csv")
    rows = []
    try:
        subprocess.run(
            [GNINA, "-r", TIER2, "-l", shard_sdf, "--minimize", "-o", minimized],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        groups = _grouped_poses(minimized, _CORE)
        order, best_mols, best_t2, n_diag = [], [], {}, {}
        for idx in idxs:
            ps = groups.get(str(idx))
            if not ps:
                continue
            on_tether = [p for p in ps if p["drift"] <= DRIFT_MAX]
            pool = on_tether if on_tether else ps
            b = min(pool, key=lambda p: p["vina"])  # clash-aware best on-tether pose
            n_diag[idx] = (len(ps), len(on_tether))
            valid = bool(on_tether) and b["vina"] < 0
            if valid:
                order.append(idx)
                best_mols.append(b["mol"])
                best_t2[idx] = (b["vina"], b["cnnsc"], b["cnnaff"])
            else:
                cid, setl, smi = meta[idx]
                rows.append(
                    [
                        idx,
                        cid,
                        setl,
                        smi,
                        "no_valid_pose",
                        len(ps),
                        len(on_tether),
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
        if best_mols:
            w = Chem.SDWriter(best_sdf)
            for mm in best_mols:
                w.write(mm)
            w.close()
            t1 = _score_only_stream(TIER1, best_sdf, env)
        else:
            t1 = []
        for i, idx in enumerate(order):
            cid, setl, smi = meta[idx]
            v2, sc2, c2 = best_t2[idx]
            v1, c1 = t1[i] if i < len(t1) else (float("nan"), float("nan"))
            np_, nt_ = n_diag[idx]
            rows.append(
                [
                    idx,
                    cid,
                    setl,
                    smi,
                    "ok",
                    np_,
                    nt_,
                    f"{v2:.3f}",
                    f"{sc2:.4f}",
                    f"{c2:.4f}",
                    f"{v1:.3f}",
                    f"{c1:.4f}",
                    f"{v2 - v1:.3f}",
                    f"{c2 - c1:.4f}",
                ]
            )
        for idx in idxs:  # embed failures (no SDF / no group)
            if str(idx) not in groups and idx not in {r[0] for r in rows}:
                cid, setl, smi = meta[idx]
                rows.append([idx, cid, setl, smi, "no_confs", 0, 0, "", "", "", "", "", "", ""])
    except Exception as e:
        for idx in idxs:
            cid, setl, smi = meta[idx]
            rows.append([idx, cid, setl, smi, f"shard_fail:{type(e).__name__}", 0, 0, *[""] * 7])
    with open(out_csv, "w", newline="") as fh:  # per-shard checkpoint
        csv.writer(fh).writerows(rows)
    return rows


HEADER = [
    "idx",
    "id",
    "set",
    "smiles",
    "status",
    "n_confs",
    "n_tethered",
    "vina_t2",
    "cnnsc_t2",
    "cnnaff_t2",
    "vina_t1",
    "cnnaff_t1",
    "gspt1_dvina",
    "gspt1_dcnnaff",
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
        f"node: WORKERS={WORKERS} ({N_GPU} GPU x {PROCS_PER_GPU}), N_CONFS={N_CONFS}, "
        f"DRIFT_MAX={DRIFT_MAX}, PASS={A.PASS}, embed={os.environ.get('ANCHOR_EMBED','coordmap')}",
        flush=True,
    )

    print("phase 1: building anchored conformers (parallel)...", flush=True)
    with mp.Pool(min(WORKERS * 4, 64), initializer=_init) as pool:
        embs = pool.map(embed, mols)
    n_emb = sum(1 for _, _, s in embs if s == "embedded")
    meta = {idx: (cid, setl, smi) for (idx, cid, setl, smi) in mols}
    print(f"phase 1 done: {n_emb}/{len(mols)} embedded ({(time.time()-t0)/60:.1f} min)", flush=True)

    good = [idx for idx, _, s in embs if s == "embedded"]
    shards = [[] for _ in range(WORKERS)]
    for j, idx in enumerate(sorted(good)):
        shards[j % WORKERS].append(idx)
    for s, idxs in enumerate(shards):
        with open(os.path.join(WORK, f"shard{s:02d}.sdf"), "w") as out:
            for idx in idxs:
                out.write(open(os.path.join(WORK, f"m{idx:05d}.sdf")).read())
    args = [(s, s % N_GPU, idxs, meta) for s, idxs in enumerate(shards) if idxs]

    print(f"phase 2: minimize {len(args)} shards across {N_GPU} GPUs...", flush=True)
    t1 = time.time()
    with mp.Pool(WORKERS) as pool:
        results = pool.map(dock_shard, args)
    allrows = [r for shard in results for r in shard]
    print(f"phase 2 done ({(time.time()-t1)/60:.1f} min)", flush=True)

    for setl, fname in (("known", "known_crbn_results.csv"), ("decoy", "decoy_crbn_results.csv")):
        srows = sorted((r for r in allrows if r[2] == setl), key=lambda r: r[0])
        with open(os.path.join(OUTDIR, fname), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(HEADER)
            w.writerows(srows)
        ok = sum(1 for r in srows if r[4] == "ok")
        print(f"  {setl}: {ok}/{len(srows)} ok -> {fname}", flush=True)
    print(f"DONE total {(time.time()-t0)/60:.1f} min -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
