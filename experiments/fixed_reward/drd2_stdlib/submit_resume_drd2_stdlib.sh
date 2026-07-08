#!/bin/bash
#SBATCH --job-name=fr_rgfn_drd2_stdlib_resume
#SBATCH --time=24:00:00                         # ~1272 iters left of 5002 @ ~54s/iter ~= 19h
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# RESUME the RGFN DRD2-on-SMALL proxy fixed-reward run that hit the 3-day wall at iter
# 3730/5002 (job 69703, Logs/021). Loads the last checkpoint into the SAME run dir, finishes
# training, then samples + emits candidates. Proxy run (no docking) -> no boost/gnina/OpenCL
# env, but keeps the cuda module for dgl. Mirrors submit_resume_6td3.sh's resume logic.
# Submit: sbatch experiments/fixed_reward/drd2_stdlib/submit_resume_drd2_stdlib.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export WANDB_PROJECT=rgfn WANDB_MODE=offline
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb WANDB_DIR=$SCRATCH/wandb
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch PIP_CACHE_DIR=$SCRATCH/.cache/pip
FR_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$FR_ROOT_DIR"

# Auto-detect the newest drd2_stdlib run's checkpoint + derive its run_name.
CKPT=$(ls -t "$FR_ROOT_DIR"/fixed_reward/drd2_stdlib/*/train/checkpoints/last_gfn.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "FATAL: no drd2_stdlib last_gfn.pt found under $FR_ROOT_DIR"; exit 44; fi
RUN_DIR=$(dirname "$(dirname "$(dirname "$CKPT")")")
RUN_NAME=${RUN_DIR#"$FR_ROOT_DIR"/}
echo "resume checkpoint: $CKPT"; echo "resume run_name: $RUN_NAME"

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

# Strip lazily-populated '*_cache' buffers so upstream Trainer's STRICT load_state_dict
# accepts the checkpoint (same fix as submit_resume_6td3.sh; the cache is recomputable).
CLEAN=${CKPT%.pt}.resumeclean.pt
python - "$CKPT" "$CLEAN" <<'PYEOF'
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
d = torch.load(src, map_location="cpu")
dropped = [k for k in list(d["model"]) if k.endswith("_cache")]
for k in dropped:
    d["model"].pop(k)
torch.save(d, dst)
print(f"[clean-ckpt] dropped {dropped}\n[clean-ckpt] wrote {dst}", flush=True)
PYEOF

python scripts/fixed_reward.py \
        --cfg configs/glue/fixed_reward_drd2_stdlib.gin \
        --seed 42 --root-dir "$FR_ROOT_DIR" \
        --run-name "$RUN_NAME" --resume-from "$CLEAN"
echo "[fr_rgfn_drd2_stdlib_resume] done"
