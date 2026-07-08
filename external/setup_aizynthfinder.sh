#!/bin/bash
# setup_aizynthfinder.sh — OURS. Install AiZynthFinder for post-hoc synthesizability scoring.
#
# AiZynthFinder (`[genheden2020aizynth]`) is the retrosynthesis tool RGFN, RxnFlow,
# and SCENT all use to report their "AiZynth success rate" (the fraction of generated
# molecules for which a full route to in-stock building blocks is found). We use it the
# same way: a VALIDATION-ONLY, post-hoc metric over a finished candidate dataset
# (`validation/harness/synthesizability.py`) — never in the training loop.
#
# Why a DEDICATED conda env (`aizynth`) instead of reusing `rgfn`:
#   AiZynthFinder pins its own onnx/onnxruntime + a large template-expansion model and
#   does not need (and must not drag in) the rgfn env's torch/dgl/docking stack. It runs
#   on CPU. This mirrors the "one env per benchmarked tool, one shared on-disk standard"
#   model already used for fraggfn/rxnflow: the evaluator reads the standard
#   candidate-dataset CSV/JSON directly, so it never imports `glue`.
#
# What it installs:
#   1. a conda env `aizynth` (python 3.10) with aizynthfinder + rdkit,
#   2. the STANDARD PUBLIC DATASET via `download_public_data` into
#      data/models/aizynthfinder/ (USPTO expansion templates + ZINC in-stock library +
#      USPTO filter model) and the matching config.yml the evaluator points at.
#
# Run from the repo root (login node is fine — CPU only):
#   bash external/setup_aizynthfinder.sh
#
# Idempotent: re-running skips the env create / data download if already present.

set -euo pipefail

ENV_NAME="${AIZYNTH_ENV:-aizynth}"
PYVER=3.10
# Prepared model+stock dir the synthesizability evaluator defaults to (--config).
DATA_DIR="data/models/aizynthfinder"
CONFIG="${DATA_DIR}/config.yml"

echo "[setup_aizynth] env=${ENV_NAME} python=${PYVER} data=${DATA_DIR}"

# --- 1. Create the dedicated conda env (python 3.10, CPU). ---------------------
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[setup_aizynth] creating conda env ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" "python=${PYVER}"
else
    echo "[setup_aizynth] conda env ${ENV_NAME} already exists — skipping create"
fi

# --- 2. Install aizynthfinder (pulls rdkit, onnxruntime, etc. from PyPI). ------
# Pin a known-good line; aizynthfinder>=4 exposes the pydantic `config.search` knobs
# and the extract_statistics() keys the evaluator reads.
echo "[setup_aizynth] installing aizynthfinder"
conda run -n "${ENV_NAME}" pip install "aizynthfinder>=4.3,<5"

# --- 3. Download the standard public dataset (templates + ZINC stock + config). -
mkdir -p "${DATA_DIR}"
if [ ! -f "${CONFIG}" ]; then
    echo "[setup_aizynth] downloading public dataset into ${DATA_DIR} (this is large)"
    conda run -n "${ENV_NAME}" download_public_data "${DATA_DIR}"
else
    echo "[setup_aizynth] ${CONFIG} already present — skipping download"
fi

# --- 4. Smoke test: config loads, a trivial target solves, SA score works. -----
echo "[setup_aizynth] verifying install"
conda run -n "${ENV_NAME}" python - "${CONFIG}" <<'PY'
import os, sys
config = sys.argv[1]

# SA score (RDKit contrib) — the cheap companion metric.
from rdkit import Chem
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer
sa = sascorer.calculateScore(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))  # aspirin
print(f"[setup_aizynth] SA(aspirin) = {sa:.2f}")

# AiZynthFinder — solve a trivial commercially-available target.
from aizynthfinder.aizynthfinder import AiZynthFinder
finder = AiZynthFinder(configfile=config)
for comp, key in ((finder.stock, "zinc"), (finder.expansion_policy, "uspto"),
                  (finder.filter_policy, "uspto")):
    items = list(getattr(comp, "items", []))
    if key in items:
        comp.select(key)
    elif items:
        comp.select(items[0])
finder.target_smiles = "Cc1ccc(C(=O)O)cc1"  # p-toluic acid (in ZINC)
finder.tree_search()
finder.build_routes()
stats = finder.extract_statistics()
print(f"[setup_aizynth] OK  solved={stats.get('is_solved')} "
      f"steps={stats.get('number_of_steps')}")
PY

echo "[setup_aizynth] done. Run synthesizability over a candidate dataset with:"
echo "  conda run -n ${ENV_NAME} python validation/harness/synthesizability.py \\"
echo "      --dataset <candidate_dataset_dir> --config ${CONFIG} --nproc 8"
