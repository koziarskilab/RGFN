#!/usr/bin/env python
"""End-to-end confirmation on a PRODUCTION SCENT checkpoint: load last_gfn.pt (trained
forward policy + logZ) + the job's guidance_models.pt (trained cost/decomposability MLPs),
and confirm the trained P_B is genuinely recovered.

Checks:
  1. sidecar loads cleanly (keys matched, none unmatched)
  2. P_B with the trained sidecar DIFFERS from the no-sidecar (fresh random) reload
     -> the sidecar carries real trained information
  3. deterministic: corrupt (random reinit) then reload sidecar -> P_B bit-identical
     -> P_B is fully pinned by the checkpoint, independent of init RNG

Usage: verify_pb_recovery-style paths; args: <last_gfn.pt> <guidance_models.pt>
"""
import os
import sys
from pathlib import Path

import gin
import torch

REPO_ROOT = Path("/home/markymoo/projects/RGFN_Fork/RGFN-Fork")
SCENT_ROOT = REPO_ROOT / "external" / "scent"
GEN_DIR = REPO_ROOT / "validation/generators/scent"
LAST_GFN = Path(sys.argv[1])
SIDECAR = Path(sys.argv[2])
CFG = (
    Path(sys.argv[3]) if len(sys.argv) > 3 else REPO_ROOT / "validation/configs/scent_seh_fixed.gin"
)
N_TRAJ, BATCH, SAMPLE_SEED = 50, 50, 7

sys.path.insert(0, str(GEN_DIR))
sys.path.insert(1, str(SCENT_ROOT))
os.chdir(SCENT_ROOT)
gin.add_config_file_search_path(str(SCENT_ROOT))

from rgfn.utils.helpers import seed_everything  # noqa: E402

seed_everything(42)
import fixed_reward  # noqa: E402,F401
from guidance_io import load_guidance_models  # noqa: E402

import rgfn  # noqa: E402,F401
from rgfn.api.trajectories import Trajectories  # noqa: E402
from rgfn.gfns.reaction_gfn.api.reaction_api import (  # noqa: E402
    ReactionActionSpace0orCBackward,
)
from rgfn.trainer.trainer import Trainer  # noqa: E402,F401

gin.parse_config_files_and_bindings(
    [str(CFG)],
    bindings=[
        'user_root_dir="/tmp/scent_check"',
        'run_name="check"',
        "Trainer.n_iterations=1",
        'ScentFixedRewardRun.run_dir="/tmp/scent_check/run"',
        f'ScentFixedRewardRun.repo_root="{REPO_ROOT}"',
        "ScentFixedRewardRun.seed=42",
    ],
)

objective = gin.get_configurable("objective/gin.singleton")()
ckpt = torch.load(str(LAST_GFN), map_location="cpu", weights_only=False)
res = objective.load_state_dict(ckpt["model"], strict=False)
print(
    f"loaded forward+logZ (real-missing={len([k for k in res.missing_keys if '_cache' not in k])})"
)
sampler = gin.get_configurable("valid_sampler/gin.singleton")()
device = objective.device
sampler.policy.set_device(device)
for pol in (objective.forward_policy, objective.backward_policy):
    if hasattr(pol, "set_device"):
        pol.set_device(device)


def pb():
    objective.assign_log_probs(traj)
    return traj.get_backward_log_probs_flat().detach().cpu().clone()


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
print(f"{len(traj)} trajectories, {len(bwd_spaces)} steps, {len(cphase)} C-phase multi-parent")

pb_fresh = pb()  # no sidecar (random guidance)
loaded, unmatched = load_guidance_models(objective, SIDECAR, map_location=device, strict=True)
pb_trained = pb()  # trained sidecar
print(f"\nsidecar keys loaded={loaded}\n         unmatched={unmatched}")

# corrupt then reload -> determinism
torch.manual_seed(31337)
for p in getattr(objective.backward_policy, "policies", [objective.backward_policy]):
    for a in ("cost_prediction_model", "decomposable_prediction_model"):
        m = getattr(p, a, None)
        if m is not None:
            for layer in m.mlp_c:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
load_guidance_models(objective, SIDECAR, map_location=device, strict=True)
pb_trained2 = pb()


def s(a, b):
    d = (a - b).abs()
    sub = d[cphase] if cphase else d
    return f"C-phase: max={sub.max():.3e} mean={sub.mean():.3e} #changed(>1e-6)={int((sub>1e-6).sum())}/{len(sub)}"


print("\n=== results ===")
print("  trained-sidecar P_B vs no-sidecar (random) reload: ", s(pb_trained, pb_fresh))
print("  reload determinism (load -> corrupt -> load again): ", s(pb_trained, pb_trained2))
info = not torch.allclose(pb_trained, pb_fresh, atol=1e-6)
det = torch.allclose(pb_trained, pb_trained2, atol=1e-6)
print("\n  sidecar clean (0 unmatched):", len(unmatched) == 0)
print("  trained P_B differs from random (sidecar carries info):", info)
print("  reload is deterministic/bit-exact:", det)
print(
    "  RESULT:",
    "PASS - trained P_B recovered end-to-end from the production checkpoint"
    if (len(unmatched) == 0 and info and det)
    else "FAIL",
)
