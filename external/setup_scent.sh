#!/bin/bash
# setup_scent.sh — OURS. Install the SCENT baseline generator for validation.
#
# SCENT ([gainski2025scent], "Scalable and Cost-Efficient de Novo Template-Based
# Molecular Generation", arXiv:2506.19865) is our COST-AWARE baseline. It is a
# fork of RGFN from the same lab that adds, on top of RGFN's synthesizable
# reaction-template action space: Recursive Cost Guidance (synthesis cost from
# building-block prices + reaction yields), an Exploitation Penalty, and a
# Dynamic Library. The adapter in validation/generators/scent/ is a THIN wrapper;
# the heavy upstream code is installed HERE (cloned under external/, not
# vendored). See validation/generators/README.md for the entrant table and the
# boundary rule, and validation/generators/scent/README.md for the design.
#
# Why a DEDICATED conda env (`scent`) instead of reusing `rgfn`:
#   SCENT's python package is *literally named* `rgfn` (it is an RGFN fork), so it
#   would SHADOW our own `rgfn/` package if installed into the `rgfn` env. This is
#   a NAMESPACE clash, where FragGFN had a VERSION clash — but the outcome is the
#   same two-env pattern: SCENT runs in its own env and reaches the SHARED docking
#   oracle across the env boundary via `scripts/score_batch.py` run under `rgfn`
#   (see the scent README). The stacks are otherwise identical (py3.11.8 /
#   torch 2.3.0 / dgl 2.2.1 / gin), so this env is a near-twin of `rgfn`.
#
# Run from the repo root (login or compute node with conda + CUDA 11.8):
#   bash external/setup_scent.sh
#
# Idempotent: re-running skips the clone/env if already present.

set -euo pipefail

ENV_NAME="${SCENT_ENV:-scent}"
PYVER=3.11.8
# SCENT upstream — pinned for reproducibility (matches the README + adapter).
SCENT_ORG=koziarskilab
SCENT_REPO=SCENT
SCENT_COMMIT=af1fee53786cf814204013ee0e1d13003a9b24f8
CLONE_DIR="external/scent"

# CUDA-11.8 wheels (Balam/Trillium load cuda/11.8.0; matches the rgfn /
# QuickVina2-GPU stack). Mirrors SCENT's README install exactly.
TORCH_VER=2.3.0
TORCH_INDEX="https://download.pytorch.org/whl/cu118"
DGL_VER="2.2.1+cu118"
DGL_FIND="https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html"

echo "[setup_scent] env=${ENV_NAME} python=${PYVER} torch=${TORCH_VER}+cu118 dgl=${DGL_VER}"

# --- 1. Clone SCENT at the pinned commit (under external/, git-ignored). --------
if [ ! -d "${CLONE_DIR}" ]; then
    echo "[setup_scent] cloning ${SCENT_ORG}/${SCENT_REPO}@${SCENT_COMMIT:0:7}"
    git -C external clone "https://github.com/${SCENT_ORG}/${SCENT_REPO}" scent
    git -C "${CLONE_DIR}" checkout "${SCENT_COMMIT}"
else
    echo "[setup_scent] ${CLONE_DIR} already present — skipping clone"
fi

# --- 2. Create the dedicated conda env (python 3.11.8). ------------------------
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[setup_scent] creating conda env ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" "python=${PYVER}"
else
    echo "[setup_scent] conda env ${ENV_NAME} already exists — skipping create"
fi

# --- 3. Install torch + dgl (CUDA 11.8), then SCENT (editable). -----------------
# Order follows SCENT's README: torch, then the matching dgl wheel, then -e .
echo "[setup_scent] installing torch ${TORCH_VER}+cu118"
conda run -n "${ENV_NAME}" pip install "torch==${TORCH_VER}" --index-url "${TORCH_INDEX}"
echo "[setup_scent] installing dgl ${DGL_VER}"
conda run -n "${ENV_NAME}" pip install "dgl==${DGL_VER}" -f "${DGL_FIND}"
echo "[setup_scent] installing SCENT (editable)"
conda run -n "${ENV_NAME}" pip install -e "${CLONE_DIR}"

# SCENT's logger imports wandb, which still imports `pkg_resources` (setuptools).
# Python 3.11 + recent pip ship venvs WITHOUT setuptools, and setuptools>=81 removed
# pkg_resources -> `import rgfn` fails with ModuleNotFoundError. Pin setuptools<81.
echo "[setup_scent] installing setuptools<81 (wandb needs pkg_resources)"
conda run -n "${ENV_NAME}" pip install "setuptools<81"

# Some systems need libxrender for RDKit drawing (per SCENT README). Harmless if
# already present; only attempt via conda when the package manager is available.
echo "[setup_scent] ensuring xorg-libxrender (RDKit dep, per SCENT README)"
conda install -y -n "${ENV_NAME}" conda-forge::xorg-libxrender || \
    echo "[setup_scent] (skipped libxrender — install manually if RDKit complains)"

# --- 4. Import smoke test. -----------------------------------------------------
# Confirms SCENT's `rgfn` fork resolves (NOT our repo-local rgfn/) and exposes the
# exact symbols our proxy/loop bind to (MPNNet, mol2graph, the reaction action
# state machine, Trainer). Run from /tmp so the repo-local `rgfn/` cannot shadow
# the installed SCENT package on sys.path.
echo "[setup_scent] verifying install"
( cd /tmp && conda run -n "${ENV_NAME}" python - <<'PY'
import rgfn, torch
from rgfn.gfns.reaction_gfn.proxies.seh_proxy import MPNNet, NUM_ATOMIC_NUMBERS
from rgfn.gfns.reaction_gfn.policies.graph_transformer import mol2graph, mols2batch, _chunks
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionAction0, ReactionActionC, ReactionStateTerminal, ReactionStateEarlyTerminal,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase
from rgfn.trainer.trainer import Trainer
print("[setup_scent] OK  SCENT rgfn at", rgfn.__file__)
print("[setup_scent] torch", torch.__version__, "cuda?", torch.cuda.is_available(),
      "NUM_ATOMIC_NUMBERS", NUM_ATOMIC_NUMBERS)
PY
)

echo "[setup_scent] done. Activate with:  conda activate ${ENV_NAME}"
echo "[setup_scent] SMALL library + shipped cost/yield data: ${CLONE_DIR}/data/small/"
