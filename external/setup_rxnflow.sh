#!/bin/bash
# setup_rxnflow.sh — OURS. Install the RxnFlow baseline generator for validation.
#
# RxnFlow (reaction-template + building-block GFlowNet, the SYNTHESIZABLE peer to RGFN;
# `[seo2024rxnflow]`) is built on a bundled Recursion `gflownet`. The adapter in
# validation/generators/rxnflow/ is a THIN wrapper; the heavy upstream code is installed
# HERE (cloned under external/, not vendored). See validation/generators/README.md for
# the entrant table and the boundary rule.
#
# Why a DEDICATED conda env (`rxnflow`) instead of reusing `rgfn`/`fraggfn`:
#   RxnFlow pins  python>=3.12,<3.13 + torch==2.5.1 (cu121). The `rgfn` env is
#   python 3.11 + torch 2.3 (cu118); `fraggfn` is python 3.10 + torch 2.1.2 — both
#   unresolvable conflicts. RxnFlow therefore runs in its own env and reaches the
#   SHARED docking oracle across the env boundary via `scripts/score_batch.py` run
#   under `rgfn` (see the rxnflow README). This is exactly the "different env per
#   benchmarked generator, one shared scoring standard" model.
#
# NOTE (CUDA): RxnFlow uses cu121 torch wheels (self-contained CUDA runtime), while the
#   bridge subprocess uses the rgfn env's cu118 + the cuda/11.8.0 module. The two are
#   separate processes/envs and don't need to agree; the GPU driver must merely support
#   CUDA 12.1 (driver >= 525). Confirm on the Balam/Trillium node (REFACTOR_LOG).
#
# Run from the repo root (login or compute node with conda + a CUDA-12-capable driver):
#   bash external/setup_rxnflow.sh            # full library
#   bash external/setup_rxnflow.sh --smoke    # also prepare a tiny block subset for smoke
#
# Idempotent: re-running skips the clone/env if already present.

set -euo pipefail

ENV_NAME="${RXNFLOW_ENV:-rxnflow}"
PYVER=3.12
SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

# RxnFlow upstream — pinned for reproducibility (matches the README + adapter).
RXNFLOW_ORG=SeonghwanSeo
RXNFLOW_REPO=RxnFlow
RXNFLOW_COMMIT="${RXNFLOW_COMMIT:-main}"   # TODO: pin to a commit SHA once validated on Balam
CLONE_DIR="external/${RXNFLOW_REPO}"

# cu121 wheels (RxnFlow pins torch 2.5.1+cu121).
TORCH_VER=2.5.1
PYG_FIND_LINKS="https://data.pyg.org/whl/torch-${TORCH_VER}+cu121.html"

# Prepared env dir (building blocks + reaction templates) the configs point at.
ENV_DIR="data/models/rxnflow_env"
ENV_DIR_SMOKE="data/models/rxnflow_env_smoke"

echo "[setup_rxnflow] env=${ENV_NAME} python=${PYVER} torch=${TORCH_VER}+cu121 smoke=${SMOKE}"

# --- 1. Clone RxnFlow at the pinned commit (under external/, git-ignored). ------
if [ ! -d "${CLONE_DIR}" ]; then
    echo "[setup_rxnflow] cloning ${RXNFLOW_ORG}/${RXNFLOW_REPO}@${RXNFLOW_COMMIT}"
    git -C external clone "https://github.com/${RXNFLOW_ORG}/${RXNFLOW_REPO}" "${RXNFLOW_REPO}"
    git -C "${CLONE_DIR}" checkout "${RXNFLOW_COMMIT}"
else
    echo "[setup_rxnflow] ${CLONE_DIR} already present — skipping clone"
fi

# --- 2. Create the dedicated conda env (python 3.12). --------------------------
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[setup_rxnflow] creating conda env ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" "python=${PYVER}"
else
    echo "[setup_rxnflow] conda env ${ENV_NAME} already exists — skipping create"
fi

# --- 3. Install RxnFlow (+ bundled gflownet, torch, pyg via find-links). --------
# RxnFlow's pyproject pins torch 2.5.1; --find-links provides the prebuilt pyg
# extension wheels (torch-scatter/sparse/cluster) against torch 2.5.1+cu121 so pip
# does not try to compile them from source (needs nvcc, slow/fails).
echo "[setup_rxnflow] installing RxnFlow (editable) + deps"
conda run -n "${ENV_NAME}" pip install -e "${CLONE_DIR}" --find-links "${PYG_FIND_LINKS}"
# gdown is imported by rxnflow.utils.misc (data/weights download) but missing from its
# pyproject deps — install it explicitly so `import rxnflow` works.
conda run -n "${ENV_NAME}" pip install gdown

# --- 4. Prepare the env directory: building blocks + reaction templates. --------
# RxnFlow builds its synthetic action space from a building-block library + a reaction
# template set via data/scripts/b_create_env.py (see CLONE_DIR/data/README.md). We use
# the PUBLIC ZINCFrag set (Enamine REAL is request-only) + the paper's 71-template HB
# set (templates/hb_edited.txt). The clone bundles a 10k debug subset
# (zincfrag_10k.smi.gz) — enough for a first end-to-end run; swap in the full ZINCFrag
# (gdown zincfrag.smi.gz, ~200k) for the headline run.
REPO_ROOT="$PWD"
ABS_ENV_DIR="${REPO_ROOT}/${ENV_DIR}"
BLOCKS="${RXNFLOW_BLOCKS:-./building_blocks/zincfrag_10k.smi.gz}"   # repo-data-relative
# real.txt = RxnFlow's default 109-template Enamine-REAL set. We use it (not the paper's
# 71-template hb_edited.txt) because hb_edited on the SMALL 10k debug library is too
# sparse — many (template, reactant) states have no compatible second block, producing
# all-masked action categories -> NaN logprobs / non-finite TB loss (validated on Balam,
# Logs/016). real.txt trains stably on the debug library; switch to hb_edited only with
# the full ~200k ZINCFrag library (RXNFLOW_TEMPLATES=./templates/hb_edited.txt).
TEMPLATES="${RXNFLOW_TEMPLATES:-./templates/real.txt}"            # 109 templates (Enamine REAL)
CPU="${RXNFLOW_CPU:-8}"
mkdir -p "$(dirname "${ABS_ENV_DIR}")"
if [ ! -d "${ABS_ENV_DIR}" ]; then
    echo "[setup_rxnflow] building env dir -> ${ENV_DIR} (blocks=${BLOCKS} templates=${TEMPLATES})"
    ( cd "${CLONE_DIR}/data" && conda run -n "${ENV_NAME}" python scripts/b_create_env.py \
        -b "${BLOCKS}" -t "${TEMPLATES}" -o "${ABS_ENV_DIR}" --cpu "${CPU}" )
else
    echo "[setup_rxnflow] ${ENV_DIR} already present — skipping prep"
fi
# The smoke config reuses ${ENV_DIR} (its tiny budget comes from the loop knobs, not the
# block-library size), so no separate smoke env is built.

# --- 5. Import smoke test. -----------------------------------------------------
echo "[setup_rxnflow] verifying install"
conda run -n "${ENV_NAME}" python - <<'PY'
import torch
import rxnflow
from rxnflow.base import RxnFlowTrainer, BaseTask  # noqa: F401
from gflownet import ObjectProperties, LogScalar   # bundled gflownet  # noqa: F401
from gflownet.models import bengio2021flow          # shared proxy net  # noqa: F401
print("[setup_rxnflow] OK  torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("[setup_rxnflow] bengio2021flow FRAGMENTS:", len(bengio2021flow.FRAGMENTS))
PY

echo "[setup_rxnflow] done. Activate with:  conda activate ${ENV_NAME}"
echo "[setup_rxnflow] NOTE: finish the env-dir prep (step 4) per ${CLONE_DIR}/data/README.md before running."
