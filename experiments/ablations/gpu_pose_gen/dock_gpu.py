"""Re-dock the entry-006 molecule set with the GPU pose-generation oracle.

Runs the SAME 408 molecules (160 known glues + 248 decoys) that entry 006 scored
with CPU gnina search, but through :class:`Docking6TD3GpuOracle` — QuickVina2-GPU
generates the Tier-2 poses, gnina ``--score_only`` does the CNN selection and both
tier scores, and the differential is ``Vina(Tier2) - Vina(Tier1)``. Output rows
carry both the GPU differential and the entry-006 reference so
``analyze_gpu_pose_gen.py`` can (a) re-rank known-vs-decoy discrimination against
the 0.946 baseline and (b) correlate per-molecule against the gnina-search dvina.

Inputs (reused from entry 006, identical molecules):
    experiments/oracle_validation/docking_6td3/known_results.csv   (160 known)
    experiments/oracle_validation/docking_6td3/decoy_cdk_results.csv (248 decoy)

Chunked + incremental + resumable: writes each chunk's rows immediately and skips
ids already present in the output CSV, so a re-run continues where it stopped.

Env (Balam compute node): module load cuda/11.8.0; conda activate rgfn;
LD_LIBRARY_PATH += $SCRATCH/vina_gpu/boost/lib; GNINA set. See submit script.
"""
import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF_DIR = HERE.parents[1] / "oracle_validation" / "docking_6td3"
KNOWN_CSV = REF_DIR / "known_results.csv"
DECOY_CSV = REF_DIR / "decoy_cdk_results.csv"

HEADER = [
    "id",
    "set",
    "smiles",
    "status",
    "n_poses",
    "vina_t2",
    "vina_t1",
    "dvina",
    "cnnsc_t2",
    "qv2_t2",  # this oracle (GPU)
    "ref_dvina",
    "ref_vina_t2",
    "ref_vina_t1",
    "ref_cnnsc_t2",  # entry-006 (gnina search)
]


def load_reference():
    """Return [(id, set, smiles, ref_dvina, ref_vina_t2, ref_vina_t1, ref_cnnsc_t2)]
    for the status==ok rows of both 006 result CSVs (the canonical 408 molecules)."""
    rows = []
    for path in (KNOWN_CSV, DECOY_CSV):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r.get("status") != "ok":
                    continue
                rows.append(
                    (
                        r["id"],
                        r["set"],
                        r["smiles"],
                        float(r["ddb1_dvina"]),
                        float(r["vina_t2"]),
                        float(r["vina_t1"]),
                        float(r["cnnsc_t2"]),
                    )
                )
    return rows


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="results CSV (incremental/resumable)")
    ap.add_argument("--chunk", type=int, default=32, help="molecules per docking chunk")
    ap.add_argument("--num-modes", type=int, default=9, help="QV2 poses per search")
    ap.add_argument("--exhaustiveness", type=int, default=8000, help="QV2 thread/search effort")
    ap.add_argument("--timing-csv", default=None, help="optional substep-timing CSV")
    ap.add_argument("--limit", type=int, default=0, help="cap molecules (smoke); 0 = all")
    ap.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="process at most N chunks "
        "this invocation then exit (0 = all); resume-friendly way to stay "
        "under the login-node per-process CPU-time limit",
    )
    args = ap.parse_args()

    from glue.oracles import Docking6TD3GpuOracle

    ref = load_reference()
    if args.limit:
        ref = ref[: args.limit]
    print(
        f"[load] {len(ref)} molecules "
        f"({sum(s=='known' for _, s, *_ in ref)} known / "
        f"{sum(s=='decoy' for _, s, *_ in ref)} decoy)",
        flush=True,
    )

    out = Path(args.out)
    done = set()
    if out.exists():
        with open(out) as fh:
            done = {r["id"] for r in csv.DictReader(fh)}
        print(f"[resume] {len(done)} ids already in {out.name}; skipping them", flush=True)
    else:
        with open(out, "w", newline="") as fh:
            csv.writer(fh).writerow(HEADER)

    oracle = Docking6TD3GpuOracle(num_modes=args.num_modes, exhaustiveness=args.exhaustiveness)
    if args.timing_csv:
        oracle.enable_step_timing(args.timing_csv)

    todo = [row for row in ref if row[0] not in done]
    print(f"[dock] {len(todo)} molecules remaining, chunk={args.chunk}", flush=True)
    n_done = 0
    for ci, chunk in enumerate(chunks(todo, args.chunk)):
        if args.max_chunks and ci >= args.max_chunks:
            print(
                f"[stop] hit --max-chunks={args.max_chunks}; "
                f"{len(todo) - n_done} molecules left for a resume run",
                flush=True,
            )
            break
        smiles = [r[2] for r in chunk]
        det = oracle.score_detailed(smiles)
        with open(out, "a", newline="") as fh:
            w = csv.writer(fh)
            for row, d in zip(chunk, det):
                _id, setl, smi, rdv, rvt2, rvt1, rcs = row
                w.writerow(
                    [
                        _id,
                        setl,
                        smi,
                        d["status"],
                        d["n_poses"],
                        f"{d['vina_t2']:.4f}",
                        f"{d['vina_t1']:.4f}",
                        f"{d['dvina']:.4f}",
                        f"{d['cnnsc_t2']:.4f}",
                        f"{d['qv2_t2']:.4f}",
                        f"{rdv:.4f}",
                        f"{rvt2:.4f}",
                        f"{rvt1:.4f}",
                        f"{rcs:.4f}",
                    ]
                )
        n_done += len(chunk)
        n_ok = sum(d["status"] == "ok" for d in det)
        print(
            f"[chunk {ci}] {n_done}/{len(todo)} done ({n_ok}/{len(chunk)} ok this chunk)",
            flush=True,
        )

    print(f"[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
