#!/bin/bash
# setup_fraggfn.sh — OURS. Install the FragGFN baseline generator for validation.
#
# FragGFN (fragment-based GFlowNet, the non-synthesizable baseline from the RGFN
# paper) is implemented with Recursion's `gflownet` (FragMolBuildingEnvContext).
# The adapter in validation/generators/fraggfn/ is a THIN wrapper; the heavy
# upstream code is installed HERE (cloned under external/, not vendored). See
# validation/generators/README.md for the entrant table and the boundary rule.
#
# Why a DEDICATED conda env (`fraggfn`) instead of reusing `rgfn`:
#   Recursion's gflownet (pinned commit below) hard-pins  python>=3.10,<3.11 +
#   torch==2.1.2 + torch-geometric==2.4.0. The `rgfn` env is python 3.11 +
#   torch 2.3.0 — an unresolvable conflict. FragGFN therefore runs in its own
#   env and reaches the SHARED docking oracle across the env boundary via
#   `scripts/score_batch.py` run under `rgfn` (see the fraggfn README). This is
#   exactly the "different env per benchmarked generator, one shared scoring
#   standard" model.
#
# Run from the repo root (login or compute node with conda + CUDA 11.8):
#   bash external/setup_fraggfn.sh
#
# Idempotent: re-running skips the clone/env if already present.

set -euo pipefail

ENV_NAME="${FRAGGFN_ENV:-fraggfn}"
PYVER=3.10
# gflownet upstream — pinned for reproducibility (matches the README + adapter).
GFLOWNET_ORG=recursionpharma
GFLOWNET_REPO=gflownet
GFLOWNET_COMMIT=da999404e997a302a81773eb183b1da6ec2a4449
CLONE_DIR="external/${GFLOWNET_REPO}"

# CUDA-11.8 wheels (Balam loads cuda/11.8.0; matches the rgfn / QuickVina2-GPU stack).
TORCH_VER=2.1.2
TORCH_INDEX="https://download.pytorch.org/whl/cu118"
PYG_WHL="https://data.pyg.org/whl/torch-${TORCH_VER}+cu118.html"

echo "[setup_fraggfn] env=${ENV_NAME} python=${PYVER} torch=${TORCH_VER}+cu118"

# --- 1. Clone Recursion's gflownet at the pinned commit (under external/). ------
if [ ! -d "${CLONE_DIR}" ]; then
    echo "[setup_fraggfn] cloning ${GFLOWNET_ORG}/${GFLOWNET_REPO}@${GFLOWNET_COMMIT:0:7}"
    git -C external clone "https://github.com/${GFLOWNET_ORG}/${GFLOWNET_REPO}" "${GFLOWNET_REPO}"
    git -C "${CLONE_DIR}" checkout "${GFLOWNET_COMMIT}"
else
    echo "[setup_fraggfn] ${CLONE_DIR} already present — skipping clone"
fi

# --- 2. Create the dedicated conda env (python 3.10). --------------------------
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[setup_fraggfn] creating conda env ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" "python=${PYVER}"
else
    echo "[setup_fraggfn] conda env ${ENV_NAME} already exists — skipping create"
fi

# --- 3. Install torch + the pyg C++ extension wheels FIRST. --------------------
# Order matters: torch-scatter/sparse/cluster must come from the pyg wheel index
# (prebuilt against torch 2.1.2+cu118) BEFORE `pip install -e gflownet`, otherwise
# pip tries to compile them from source (needs nvcc, slow/fails).
echo "[setup_fraggfn] installing torch ${TORCH_VER}+cu118"
conda run -n "${ENV_NAME}" pip install "torch==${TORCH_VER}" --index-url "${TORCH_INDEX}"
echo "[setup_fraggfn] installing pyg extension wheels"
conda run -n "${ENV_NAME}" pip install \
    torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 -f "${PYG_WHL}"

# --- 4. Install gflownet (pulls torch-geometric==2.4.0 + the rest from PyPI). --
echo "[setup_fraggfn] installing gflownet (editable)"
conda run -n "${ENV_NAME}" pip install -e "${CLONE_DIR}"

# torch 2.1.2 was built against numpy 1.x; a transitive dep (scipy) pulls numpy 2.x
# which triggers an ABI mismatch ("_ARRAY_API not found"). Pin numpy<2 to be safe.
echo "[setup_fraggfn] pinning numpy<2 (torch 2.1.2 ABI)"
conda run -n "${ENV_NAME}" pip install "numpy<2"

# --- 5. Import smoke test. -----------------------------------------------------
echo "[setup_fraggfn] verifying install"
conda run -n "${ENV_NAME}" python - <<'PY'
import torch
import gflownet
from gflownet import GFNTask, ObjectProperties, LogScalar
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow
from gflownet.online_trainer import StandardOnlineTrainer
print("[setup_fraggfn] OK  torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("[setup_fraggfn] FRAGMENTS:", len(bengio2021flow.FRAGMENTS),
      "FRAGMENTS_18:", len(bengio2021flow.FRAGMENTS_18))
PY

echo "[setup_fraggfn] done. Activate with:  conda activate ${ENV_NAME}"
