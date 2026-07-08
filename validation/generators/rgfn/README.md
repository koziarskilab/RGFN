# RGFN entrant (fixed-reward)

Unlike the other entrants, the RGFN benchmark entrant is **not** a thin adapter over an
external package — it *is* our shipped pipeline. So there is no adapter code here: the
RGFN entrant runs the production fixed-reward pipeline directly.

- **Generator:** `glue.fixed_reward.FixedRewardPipeline` (the single-shot, no-oracle
  counterpart of `glue.active_learning.ActiveLearningLoop`; see
  `glue/fixed_reward/pipeline.py`).
- **Entry point:** `scripts/fixed_reward.py`.
- **Config (matched four-way, sEH proxy):** `configs/glue/fixed_reward_seh_proxy.gin`
  — trains RGFN once against the frozen Bengio-2021 `SehMoleculeProxy` (the same reward
  generator the FragGFN / RxnFlow / SCENT fixed-reward entrants use), then emits the
  standard candidate dataset directly (it can import `glue`, so no ingest bridge needed).
- **Second RGFN test (paper reproduction, RGFN-only):**
  `configs/glue/fixed_reward_seh_docking.gin` — QuickVina2-GPU docking as the reward
  generator (the RGFN paper's "docking directly in the loop").

Run (rgfn env, from repo root):

    python scripts/fixed_reward.py --cfg configs/glue/fixed_reward_seh_proxy.gin \
        --seed 42 --root-dir $SCRATCH/rgfn_runs/experiments

Submit on Balam: `experiments/fixed_reward/seh_proxy/submit_fixed_seh_proxy.sh`.

The candidate dataset lands under `<run_dir>/fixed_reward/candidates/` and is scored for
synthesizability by `validation/harness/synthesizability.py` alongside the other
entrants; `validation/harness/aggregate_synthesizability.py` assembles the four-way table.
