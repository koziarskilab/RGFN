#!/bin/bash
#SBATCH --job-name=fr_rgfn_6td3_resume
#SBATCH --time=24:00:00                         # ~140 iters left (of 400) + emit; ~11h at 277s/iter, margin for contention
#SBATCH --partition=compute
#SBATCH --exclude=balam008
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# RESUME the RGFN 6TD3 differential fixed-reward run that hit the 20h wall at iter 260/400
# (job 69695, Logs/021). Loads the last checkpoint into the SAME run dir, trains the remaining
# iters to n_iterations, then samples + emits candidates. Uses scripts/fixed_reward.py's
# --run-name (reuse the existing run dir) + --resume-from (Trainer.resume_path).
# Submit: sbatch experiments/fixed_reward/6td3/submit_resume_6td3.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export WANDB_PROJECT=rgfn WANDB_MODE=offline
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb WANDB_DIR=$SCRATCH/wandb
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch PIP_CACHE_DIR=$SCRATCH/.cache/pip
FR_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$FR_ROOT_DIR"

# Auto-detect the newest 6TD3 run's checkpoint + derive its run_name (relative to FR_ROOT_DIR).
CKPT=$(ls -t "$FR_ROOT_DIR"/fixed_reward/6td3/*/train/checkpoints/last_gfn.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "FATAL: no 6td3 last_gfn.pt checkpoint found under $FR_ROOT_DIR"; exit 44; fi
RUN_DIR=$(dirname "$(dirname "$(dirname "$CKPT")")")   # .../fixed_reward/6td3/<timestamp>
RUN_NAME=${RUN_DIR#"$FR_ROOT_DIR"/}                    # fixed_reward/6td3/<timestamp>
echo "resume checkpoint: $CKPT"
echo "resume run_name:   $RUN_NAME"

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export GNINA=/scratch/markymoo/gnina/run_gnina.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

HC=$SCRATCH/vina_gpu/opencl_healthcheck
HC_OUT=$(CUDA_VISIBLE_DEVICES=0 "$HC" 2>&1)
grep -q "clCreateContext err=0" <<<"$HC_OUT" || { echo "FATAL: OpenCL dead on $(hostname)"; echo "$HC_OUT"; exit 42; }
echo "OpenCL health OK on $(hostname)"

# Strip lazily-populated '*_cache' buffers from the checkpoint's model state_dict: upstream
# Trainer.__init__ does a STRICT load_state_dict, but the forward policy's block-embedding
# cache (forward_policy.b_action_embedding_fn._cache) is populated during training and absent
# from a freshly-built model -> strict load rejects it as an "unexpected key". The cache is
# recomputable, so dropping it is safe. Write a cleaned checkpoint and resume from that (also
# handles re-resumes, since each interrupted job re-saves the dirty cache).
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
        --cfg configs/glue/fixed_reward_6td3.gin \
        --seed 42 --root-dir "$FR_ROOT_DIR" \
        --run-name "$RUN_NAME" --resume-from "$CLEAN"
echo "[fr_rgfn_6td3_resume] done"
