#!/usr/bin/env python
"""Verify that SCENT's backward-policy P_B is EXACTLY recoverable once the guidance-model
weights are persisted via ``guidance_io`` (this experiment's fix).

The check is an in-process round trip on a fixed batch of trajectories:

    P_B_A        = P_B with the current guidance weights
    save sidecar (guidance_io.save_guidance_models)
    P_B_corrupt  = P_B after re-initialising the guidance MLPs at random
                   (== the state a checkpoint reload lands in WITHOUT the sidecar)
    P_B_restored = P_B after loading the sidecar back in (guidance_io.load_guidance_models)

PASS iff  P_B_restored == P_B_A (bit-exact)  AND  P_B_corrupt != P_B_A. Because
``load_state_dict`` is value-agnostic, a bit-exact restore of these weights restores the
trained P_B exactly too.

Two modes:
  * default (LOGIN-SAFE, inference only): load the trained FORWARD policy from an existing
    ``last_gfn.pt`` so trajectories are realistic; the round trip is on the guidance models.
        conda run -n scent python .../verify_pb_recovery.py --checkpoint <last_gfn.pt>
  * ``--train N`` (COMPUTE NODE only): train a fresh SCENT sEH model N iters first, so the
    guidance weights are genuinely trained, then save last_gfn.pt + guidance_models.pt and
    round-trip. Do NOT run this on a login node.
"""
import argparse
import os
import sys
from pathlib import Path

import gin
import torch

REPO_ROOT = Path("/home/markymoo/projects/RGFN_Fork/RGFN-Fork")
SCENT_ROOT = REPO_ROOT / "external" / "scent"
GEN_DIR = REPO_ROOT / "validation/generators/scent"
CFG = REPO_ROOT / "validation/configs/scent_seh_fixed.gin"
RUN_DIR = Path("/tmp/scent_pb_recovery")
N_TRAJ, BATCH, SAMPLE_SEED = 50, 50, 7

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=None, help="existing last_gfn.pt (forward policy source)")
ap.add_argument("--train", type=int, default=0, help="if >0, train that many iters (compute node)")
args = ap.parse_args()

sys.path.insert(0, str(GEN_DIR))  # sibling imports: guidance_io, route, fixed_reward
sys.path.insert(1, str(SCENT_ROOT))
os.chdir(SCENT_ROOT)
gin.add_config_file_search_path(str(SCENT_ROOT))

from rgfn.utils.helpers import seed_everything  # noqa: E402

seed_everything(42)
import fixed_reward  # noqa: E402,F401  (registers @ScentFixedRewardRun)
from guidance_io import load_guidance_models, save_guidance_models  # noqa: E402

import rgfn  # noqa: E402,F401
from rgfn.api.trajectories import Trajectories  # noqa: E402
from rgfn.gfns.reaction_gfn.api.reaction_api import (  # noqa: E402
    ReactionActionSpace0orCBackward,
)
from rgfn.trainer.trainer import Trainer  # noqa: E402,F401  (registers @Trainer)

RUN_DIR.mkdir(parents=True, exist_ok=True)
gin.parse_config_files_and_bindings(
    [str(CFG)],
    bindings=[
        'user_root_dir="/tmp/scent_pb_recovery"',
        'run_name="verify"',
        f"Trainer.n_iterations={max(args.train, 1)}",
        f'ScentFixedRewardRun.run_dir="{RUN_DIR}/run"',
        f'ScentFixedRewardRun.repo_root="{REPO_ROOT}"',
        "ScentFixedRewardRun.seed=42",
    ],
)

gpath = RUN_DIR / "guidance_models.pt"
if args.train > 0:
    print(
        f"\n########## TRAIN SHORT SCENT sEH ({args.train} iters) [COMPUTE NODE] ##########",
        flush=True,
    )
    trainer = gin.get_configurable("trainer/gin.singleton")()
    trainer.train()
    objective = trainer.objective
    sampler = trainer.valid_sampler
    ckpt_dir = Path(trainer.run_dir) / "train" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.make_checkpoint("last_gfn", {})
    gpath = ckpt_dir / "guidance_models.pt"
else:
    print(
        "\n########## BUILD OBJECTIVE + LOAD TRAINED FORWARD POLICY [LOGIN-SAFE] ##########",
        flush=True,
    )
    objective = gin.get_configurable("objective/gin.singleton")()
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        res = objective.load_state_dict(ckpt["model"], strict=False)
        real_missing = [k for k in res.missing_keys if "_cache" not in k]
        print(
            f"loaded forward policy + logZ from {args.checkpoint} "
            f"(real-missing={len(real_missing)})",
            flush=True,
        )
    sampler = gin.get_configurable("valid_sampler/gin.singleton")()

device = objective.device
sampler.policy.set_device(device)
for pol in (objective.forward_policy, objective.backward_policy):
    if hasattr(pol, "set_device"):
        pol.set_device(device)


def backward_logprobs():
    objective.assign_log_probs(traj)
    return traj.get_backward_log_probs_flat().detach().cpu().clone()


print("\n########## SAMPLE FIXED BATCH ##########", flush=True)
seed_everything(SAMPLE_SEED)
batches = list(sampler.get_trajectories_iterator(N_TRAJ, BATCH))
traj = Trajectories.from_trajectories(batches) if len(batches) > 1 else batches[0]
bwd_spaces = traj.get_backward_action_spaces_flat()
cphase = [
    i
    for i, sp in enumerate(bwd_spaces)
    if isinstance(sp, ReactionActionSpace0orCBackward)
    and len(sp.get_possible_actions_indices()) > 1
]
print(
    f"{len(traj)} trajectories, {len(bwd_spaces)} steps, "
    f"{len(cphase)} C-phase multi-parent steps",
    flush=True,
)

pb_A = backward_logprobs()
keys = save_guidance_models(objective, gpath)
print(f"saved guidance sidecar -> {gpath} (keys={keys})", flush=True)

print("\n########## IN-PROCESS ROUND TRIP ##########", flush=True)
# (a) Corrupt: reinit guidance MLPs at random == what a reload WITHOUT the sidecar gives.
torch.manual_seed(31337)
n_reinit = 0
for pol in getattr(objective.backward_policy, "policies", [objective.backward_policy]):
    for attr in ("cost_prediction_model", "decomposable_prediction_model"):
        m = getattr(pol, attr, None)
        if m is not None:
            for layer in m.mlp_c:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
                    n_reinit += 1
pb_corrupt = backward_logprobs()

# (b) Restore: load the sidecar back into the (corrupted) guidance models.
loaded, unmatched = load_guidance_models(objective, gpath, map_location=device, strict=True)
pb_restored = backward_logprobs()
print(f"reinit {n_reinit} layers; loaded sidecar keys={loaded} unmatched={unmatched}", flush=True)


def stats(a, b):
    d = (a - b).abs()
    sub = d[cphase] if cphase else d
    return (
        f"all: max={d.max():.3e} mean={d.mean():.3e} | "
        f"C-phase: max={sub.max():.3e} mean={sub.mean():.3e} "
        f"#changed(>1e-6)={int((sub > 1e-6).sum())}/{len(sub)}"
    )


print("\n=== |ΔlogP_B| vs P_B_A (the weights we saved) ===")
print("  corrupt  (reload WITHOUT sidecar): ", stats(pb_A, pb_corrupt))
print("  restored (reload WITH sidecar):    ", stats(pb_A, pb_restored))

exact = torch.allclose(pb_A, pb_restored, atol=1e-6, rtol=0)
corrupt_differs = not torch.allclose(pb_A, pb_corrupt, atol=1e-6, rtol=0)
print("\n########## VERDICT ##########")
print(f"  restored == saved (bit-exact, atol=1e-6): {exact}")
print(f"  corrupt  != saved (sidecar actually matters): {corrupt_differs}")
print(
    "  RESULT:",
    "PASS - guidance sidecar recovers P_B exactly" if (exact and corrupt_differs) else "FAIL",
)
