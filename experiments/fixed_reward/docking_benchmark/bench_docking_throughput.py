#!/usr/bin/env python
"""Docking-throughput benchmark for per-step fixed-reward docking (Phase 0).

Question this answers
---------------------
Can the baseline generators (FragGFN / RxnFlow / SCENT) — which run in their own
conda envs and reach docking only by shelling out to ``scripts/score_batch.py`` in
the ``rgfn`` env — use docking as a **fixed reward called every training step**?
A single GFN run docks ~100 molecules/step for a few hundred steps. Whether that is
feasible turns on ONE unknown: the **cold cross-env per-call overhead** of
``score_batch.py`` (import ``glue``/``dgl``/``torch`` + build ``DockingMoleculeProxy``
+ init the GPU/OpenCL context), paid once per step on top of the actual docking.

What it measures (reusing existing oracles — no new docking code)
----------------------------------------------------------------
For each oracle (``docking_seh`` single-target, ``docking_6td3_gpu`` two-tier
differential — the GPU-pose + CPU-gnina pipeline of Logs/013/014):

  * **warm / in-env**: import ``glue`` once, build the oracle once, dock batches of a
    few sizes; the per-molecule dock rate with the proxy + GPU already hot (this is
    what a *persistent scoring server* would deliver). Uses the oracle's own
    ``enable_step_timing`` for the substep breakdown.
  * **cold / cross-env**: for each batch size, spawn
    ``conda run -n rgfn python scripts/score_batch.py ...`` as a fresh subprocess and
    time the whole call — exactly the per-step cost a baseline pays. An OLS fit of
    ``total(B) = startup + B·per_mol`` across batch sizes isolates the fixed startup
    from the per-molecule dock rate.

Then it prints a projection table: for a full single-shot run of ``I`` iterations ×
``M`` molecules/iter, the wall-clock under (a) **per-step cold subprocess** =
``I·(startup + M·per_mol_cold)`` vs (b) **persistent warm server** =
``startup + I·M·per_mol_warm`` — flagged against the SLURM walltime cap so the bridge
architecture and iteration budget fall straight out of the numbers.

Run (rgfn env, GPU node — heavy gnina CPU is killed by the login CPU cap):
    python experiments/fixed_reward/docking_benchmark/bench_docking_throughput.py \
        --oracles docking_seh docking_6td3_gpu \
        --smiles-csv experiments/active_learning/6td3/seed_6td3.csv \
        --n 200 --batch-sizes 8 32 100 --repeats 2 \
        --out-dir "$SCRATCH/rgfn_runs/docking_benchmark/<jobid>"
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]

# oracle name -> constructor kwargs used by BOTH the warm (in-env) build and the cold
# (score_batch.py --oracle-arg) call, so the two paths dock identically. Defaults on
# the classes already match the production reward; kept explicit for provenance.
ORACLE_ARGS: Dict[str, Dict] = {
    "docking_seh": {},  # DockingSEHOracle defaults (receptor_name=sEH, exh=8000, bs=25)
    "docking_6td3_gpu": {},  # Docking6TD3GpuOracle defaults (num_modes=9, exh=8000, bs=25)
}


# --------------------------------------------------------------------- SMILES io
def read_smiles(path: Path, n: int) -> List[str]:
    """Read up to ``n`` SMILES from a CSV with a ``smiles`` column (or 1-per-line)."""
    lines = path.read_text().splitlines()
    out: List[str] = []
    if lines and "smiles" in lines[0].lower() and "," in lines[0]:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            col = "smiles" if "smiles" in reader.fieldnames else reader.fieldnames[0]
            for row in reader:
                s = (row.get(col) or "").strip()
                if s:
                    out.append(s)
                if len(out) >= n:
                    break
    else:
        for ln in lines:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                out.append(ln.split()[0])
            if len(out) >= n:
                break
    return out


def _ols(xs: List[float], ys: List[float]) -> "tuple[float, float]":
    """Ordinary least squares -> (intercept=startup, slope=per_mol)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0:
        return my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
    return my - slope * mx, slope


# --------------------------------------------------------------------- warm mode
def bench_warm(oracle_name: str, smiles: List[str], batch_sizes: List[int], out_dir: Path) -> Dict:
    """In-env: build the oracle once (warm proxy + GPU), time a batch of each size."""
    import glue  # noqa: F401  (registers oracles)
    from glue.oracles import Docking6TD3GpuOracle, DockingSEHOracle

    cls = {"docking_seh": DockingSEHOracle, "docking_6td3_gpu": Docking6TD3GpuOracle}[oracle_name]
    oracle = cls(**ORACLE_ARGS.get(oracle_name, {}))
    timer_csv = out_dir / f"warm_{oracle_name}_substeps.csv"
    if hasattr(oracle, "enable_step_timing"):
        oracle.enable_step_timing(timer_csv)

    # Warm-up: build proxy + init GPU/OpenCL on a tiny batch (not timed).
    print(f"[warm:{oracle_name}] warming up (proxy build + GPU init)...", flush=True)
    _ = oracle.score(smiles[: min(4, len(smiles))])

    points = []
    for b in batch_sizes:
        batch = smiles[:b]
        t0 = time.perf_counter()
        scores = oracle.score(batch)
        dt = time.perf_counter() - t0
        ok = sum(1 for s in scores if s is not None and s == s)
        per_mol = dt / len(batch)
        points.append({"batch": len(batch), "seconds": dt, "per_mol": per_mol, "n_ok": ok})
        print(
            f"[warm:{oracle_name}] batch={len(batch):>3} {dt:7.1f}s "
            f"{per_mol:5.2f}s/mol  ok={ok}/{len(batch)}",
            flush=True,
        )
    # Warm per-mol: prefer the largest batch (best amortized).
    per_mol_warm = points[-1]["per_mol"] if points else float("nan")
    return {"points": points, "per_mol_warm": per_mol_warm, "substep_csv": str(timer_csv)}


# --------------------------------------------------------------------- cold mode
def bench_cold(
    oracle_name: str, smiles: List[str], batch_sizes: List[int], repeats: int, out_dir: Path
) -> Dict:
    """Cross-env: spawn a fresh ``conda run -n rgfn score_batch.py`` per batch, time it."""
    tmp = out_dir / "cold_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    oracle_args: List[str] = []
    for k, v in ORACLE_ARGS.get(oracle_name, {}).items():
        oracle_args += ["--oracle-arg", f"{k}={v}"]

    xs: List[float] = []
    ys: List[float] = []
    points = []
    for b in batch_sizes:
        smi_path = tmp / f"batch_{b}.smi"
        smi_path.write_text("\n".join(smiles[:b]) + "\n")
        for r in range(repeats):
            out_csv = tmp / f"labels_{oracle_name}_{b}_{r}.csv"
            cmd = [
                "conda", "run", "--no-capture-output", "-n", "rgfn",
                "python", "scripts/score_batch.py",
                "--oracle", oracle_name,
                "--in", str(smi_path),
                "--out", str(out_csv),
                *oracle_args,
            ]  # fmt: skip
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
            dt = time.perf_counter() - t0
            n_ok = _parse_scored(proc.stdout)
            if proc.returncode != 0:
                print(
                    f"[cold:{oracle_name}] batch={b} rep={r} FAILED rc={proc.returncode}\n"
                    f"{proc.stdout[-800:]}\n{proc.stderr[-800:]}",
                    flush=True,
                )
                continue
            xs.append(float(b))
            ys.append(dt)
            points.append({"batch": b, "rep": r, "seconds": dt, "n_ok": n_ok})
            print(
                f"[cold:{oracle_name}] batch={b:>3} rep={r} {dt:7.1f}s  "
                f"({dt / b:5.2f}s/mol incl startup)  ok={n_ok}/{b}",
                flush=True,
            )
    startup, per_mol_cold = _ols(xs, ys) if len(set(xs)) >= 2 else (float("nan"), float("nan"))
    print(
        f"[cold:{oracle_name}] OLS fit: startup={startup:.1f}s  per_mol={per_mol_cold:.2f}s/mol",
        flush=True,
    )
    return {"points": points, "startup_s": startup, "per_mol_cold": per_mol_cold}


def _parse_scored(stdout: str) -> Optional[int]:
    """Pull the '[score_batch] scored N/M successfully' count from stdout."""
    for line in stdout.splitlines():
        if "scored" in line and "successfully" in line:
            try:
                return int(line.split("scored")[1].split("/")[0].strip())
            except (ValueError, IndexError):
                return None
    return None


# ------------------------------------------------------------------- projection
def project(name: str, warm: Dict, cold: Dict, iters: List[int], mols_per_iter: int) -> List[Dict]:
    """Project full-run wall-clock under per-step-cold vs persistent-warm-server."""
    per_mol_warm = warm.get("per_mol_warm", float("nan"))
    startup = cold.get("startup_s", float("nan"))
    per_mol_cold = cold.get("per_mol_cold", float("nan"))
    rows = []
    print(f"\n=== projection: {name} ({mols_per_iter} mol/iter) ===", flush=True)
    print(
        f"{'iters':>6} | {'cold per-step (h)':>18} | {'warm server (h)':>16} | fits 3-day?",
        flush=True,
    )
    for I in iters:
        cold_h = I * (startup + mols_per_iter * per_mol_cold) / 3600.0
        warm_h = (startup + I * mols_per_iter * per_mol_warm) / 3600.0
        fits = "yes" if cold_h <= 72 else "NO"
        rows.append(
            {"iters": I, "cold_hours": cold_h, "warm_hours": warm_h, "cold_fits_3day": fits}
        )
        print(f"{I:>6} | {cold_h:>18.1f} | {warm_h:>16.1f} | {fits}", flush=True)
    return rows


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracles", nargs="+", default=["docking_seh", "docking_6td3_gpu"])
    ap.add_argument("--smiles-csv", default="experiments/active_learning/6td3/seed_6td3.csv")
    ap.add_argument("--n", type=int, default=200, help="max SMILES to load")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 32, 100])
    ap.add_argument("--repeats", type=int, default=2, help="cold-mode repeats per batch size")
    ap.add_argument("--mode", choices=["both", "warm", "cold"], default="both")
    ap.add_argument("--iters", nargs="+", type=int, default=[400, 1000])
    ap.add_argument(
        "--mols-per-iter", type=int, default=100, help="~train_forward+replay trajectories"
    )
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    smiles = read_smiles(Path(args.smiles_csv), max(args.n, max(args.batch_sizes)))
    if len(smiles) < max(args.batch_sizes):
        sys.exit(
            f"need >= {max(args.batch_sizes)} SMILES, got {len(smiles)} from {args.smiles_csv}"
        )
    print(
        f"[bench] {len(smiles)} SMILES from {args.smiles_csv}; oracles={args.oracles}", flush=True
    )

    results: Dict[str, Dict] = {}
    for name in args.oracles:
        print(f"\n########## {name} ##########", flush=True)
        warm = bench_warm(name, smiles, args.batch_sizes, out_dir) if args.mode != "cold" else {}
        cold = (
            bench_cold(name, smiles, args.batch_sizes, args.repeats, out_dir)
            if args.mode != "warm"
            else {}
        )
        proj = project(name, warm, cold, args.iters, args.mols_per_iter) if warm and cold else []
        results[name] = {"warm": warm, "cold": cold, "projection": proj}

    out_json = out_dir / "benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[bench] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
