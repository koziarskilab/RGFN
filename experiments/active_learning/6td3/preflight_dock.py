"""Pre-flight GPU-docking gate: prove THIS node can actually generate poses before
the active-learning loop spends ~an hour training a generator it can't score.

Why this is needed on top of the OpenCL health check (Logs/014, jobs 69481/69511):
balam009 *passes* the raw ``clCreateContext`` probe (``opencl_healthcheck.c``) yet
QuickVina2-GPU produces **zero poses** for every molecule in real docking — a
subtler post-outage GPU degradation than balam008's hard ``clCreateContext=-5``.
The OpenCL gate can't see it; only an actual dock can. So we dock a couple of the
validated seed molecules here, at job startup, and bail immediately (before the
62-minute round-1 GFN training) if the node can't pose them.

Exit 0 if at least one seed molecule docks (node is usable); exit 42 with a clear
message otherwise (add the node to the submit script's ``--exclude`` and resubmit).
Run under the same env as the loop (``source ~/bin/rgfn-smoke-env.sh`` or the
submit script's module/conda setup).
"""

import sys

from glue.oracles.docking_gpu_differential_oracle import Docking6TD3GpuOracle

# Two validated seed molecules (from seed_6td3.csv / entry 002): known to dock and
# score on a healthy node (entry 013 / the Trillium H100 smoke test in Logs/014).
PROBE_SMILES = [
    "C[C@H](C(=O)Nc1nc2ccccc2[nH]1)N1Cc2ccccc2C1=O",  # ref dvina ~ -2.20
    "C=Cc1ccc(CNc2nc(N[C@H](CC)CO)nc3c2ncn3C(C)C)cc1",  # ref dvina ~ -1.26
]


def main() -> int:
    oracle = Docking6TD3GpuOracle(num_modes=9, exhaustiveness=8000, docking_batch_size=8, n_gpu=1)
    print(
        f"[preflight] docking {len(PROBE_SMILES)} seed molecules to test the node ...", flush=True
    )
    details = oracle.score_detailed(PROBE_SMILES)
    posed = 0
    for smi, d in zip(PROBE_SMILES, details):
        n_poses = d.get("n_poses") or 0
        print(
            f"[preflight]   status={d.get('status')} n_poses={n_poses} dvina={d.get('dvina')}",
            flush=True,
        )
        if n_poses > 0:
            posed += 1
    if posed == 0:
        print(
            "[preflight] FATAL: QuickVina2-GPU produced NO poses for any seed molecule on "
            f"this node -- it is degraded for docking despite passing the OpenCL probe "
            f"(cf. balam009, Logs/014). Add this node to the submit script's "
            f"'#SBATCH --exclude' and resubmit.",
            flush=True,
        )
        return 42
    print(
        f"[preflight] OK: {posed}/{len(PROBE_SMILES)} seed molecules docked -- node is usable.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
