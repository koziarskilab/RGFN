# SCENT — recovering the trained backward policy (P_B) from checkpoints

**Date:** 2026-07-07, ~4pm

### Question

Can we make SCENT's trained "backward policy" — the part of the model that decides, in
reverse, how a molecule was likely assembled — survive being saved to disk, so that we can
reload a finished SCENT run and analyse it faithfully later?

### Context & Summary

SCENT is our cost-aware, synthesizable baseline generator (entry 017), one of four in the
matched comparison. We want to run our hub-diversification / flow analysis (entry 022) on
*every* generator, not just RGFN. That analysis reconstructs, for each intermediate
molecule, how much "flow" the model routes through it — a calculation that needs both
directions of the model's policy: the forward policy (how it builds molecules) **and** the
backward policy (how it takes them apart).

A prior audit this session found a problem: SCENT's checkpoints (`last_gfn.pt`) save only
the forward policy plus one global constant, and **silently drop the entire trained
backward policy**. SCENT shapes its backward policy with two small learned networks — a
synthesis-**cost** model and a **decomposability** model (~263k weights each) — and because
of how they're stored in the code, the save routine skips them. Reloading a finished run
therefore gives you *random* backward-policy weights, which produce a plausible-looking but
wrong backward distribution (we measured it changing on 75% of the meaningful decisions
depending purely on the random seed). So the trained backward policy was, until now,
unrecoverable from any completed SCENT run.

This experiment implements the fix — save those two networks in a small sidecar file next
to the checkpoint — and verifies that, with the sidecar, the backward policy reloads
**exactly**. We also went back and flagged every pre-fix SCENT run on disk as
irrecoverable.

### Answer

The trained backward policy is now fully recoverable. With the sidecar loaded, the reloaded
backward log-probabilities match the originals bit-for-bit (max difference 0.0 across all
690 decision steps in the test batch); without it, a reload lands on random weights that
differ on 146 of the 195 meaningful (multi-parent) steps. The fix is a self-contained
save/load helper wired into our own SCENT adapter, so every future SCENT run emits the
sidecar automatically; the SCENT clone itself is untouched. Existing pre-fix checkpoints
remain irrecoverable and are now labelled as such on disk.

### Relevance to our Publication

SCENT is a baseline in our four-way generator comparison, and our flow / hub-diversification
analysis is a methodological contribution of the paper. Reviewers (e.g. NeurIPS) will expect
that analysis to apply uniformly across generators — being unable to run it on the
cost-aware baseline would be a visible gap. It's also a plain reproducibility point: a
checkpoint that can't reproduce its own sampling policy is a reproducibility bug, and we now
close it for all future runs and document it for the past ones.

### Next Experiments

**Refining for publication**
- **Submitted (2026-07-07):** all four SCENT fixed-reward re-runs with the patched adapter
  (seh 70066, drd2 70067, 6td3 70068, clpp 70069) so each produces a checkpoint + guidance
  sidecar. When they finish, confirm exact P_B recovery on the genuinely-trained guidance
  weights (`verify_pb_recovery.py --checkpoint <new last_gfn.pt>`).
- Run the balance-based flow analysis (entry 022 machinery) on a recovered SCENT model and
  compare its visit-count vs Trajectory-Balance flow agreement against RGFN's.

**Next steps in project**
- Fold SCENT into the cross-generator flow / hub-diversification comparison so the
  concurrency-vs-diversity Pareto story covers the cost-aware baseline, not just RGFN.

---

# Re-creation

### Relevant Files

Root: `./validation/generators/scent/`

**Scripts / code (ours — the fix)**
- `guidance_io.py` — **new.** Save/load helper for the backward-policy guidance models. Walks
  `objective.backward_policy.policies` and writes each learned sub-model's `state_dict` to a
  `guidance_models.pt` sidecar keyed by `policies.<i>.<attr>`; loads them back on reload.
  Exists because `JointlyGuidedBackwardPolicy` stores its sub-policies in a plain Python list,
  which `nn.Module.state_dict()` does not recurse into.
- `fixed_reward.py` — **modified.** `ScentFixedRewardRun` now calls `_save_guidance_models()`
  right after `trainer.train()`, writing the sidecar beside `last_gfn.pt`.
- `al_loop.py` — **modified.** `ScentActiveLearningLoop` saves the sidecar after each round's
  `trainer.train()` (kept in sync with the round's overwritten `last_gfn.pt`).
- `verify_pb_recovery.py` — **new.** In-process round-trip verifier: save → corrupt (random
  reinit, == a no-sidecar reload) → restore, asserting the restored P_B is bit-exact. Default
  mode is login-safe (loads a trained forward policy from an existing checkpoint, inference
  only); `--train N` trains a fresh short model first (compute node only).

**Reference (upstream SCENT clone — read, not modified)**
- `external/scent/rgfn/gfns/reaction_gfn/policies/jointly_biased_backward_policy.py` — the
  `self.policies = policies` plain-list attribute that causes the drop.
- `external/scent/rgfn/gfns/reaction_gfn/policies/{cost_biased_backward_policy.py,decomposability_guided_backward_policy.py}`
  and `.../guidance_models/{cost_models.py,decomposable_models.py}` — the two guidance MLPs
  (`SimpleCostModel.mlp_c`, `BinaryDecomposableModel.mlp_c`) and their own Adam optimizers.

**Checkpoints (inputs / on disk)** — root `/scratch/markymoo/rgfn_runs/experiments/`
- `fixed_reward/scent_seh/2026-07-02_10-23-09/train/checkpoints/last_gfn.pt` — the existing
  trained sEH checkpoint used as the forward-policy source for the login-safe verification.
- Five pre-fix checkpoint dirs (below) now each carry `BACKWARD_POLICY_NOT_SAVED.txt`.

**Job logs** — root `/scratch/markymoo/rgfn_runs/`
- Patched fixed-reward re-runs submitted 2026-07-07: **70066** (seh), **70067** (drd2),
  **70068** (6td3, docking), **70069** (clpp, docking). Outputs at `fr_scent_*-<jobid>.{out,err}`;
  each new run dir will carry `train/checkpoints/guidance_models.pt`.

### Relevant Versions

Files are **not yet committed** (working tree on branch `GPU-Dock`, base commit `133c788`):

```
 M validation/generators/scent/al_loop.py
 M validation/generators/scent/fixed_reward.py
?? validation/generators/scent/guidance_io.py
?? validation/generators/scent/verify_pb_recovery.py
```

Plus this log (`Logs/024_scent-backward-policy-recovery.md`) and the README index row.
`external/scent` is git-ignored (pinned clone from `external/setup_scent.sh`) and was
deliberately left unmodified. The five `BACKWARD_POLICY_NOT_SAVED.txt` markers live on
`$SCRATCH` (outside the repo).

**[TODO — add commit hash after committing the four `validation/generators/scent/` files + this log.]**

### Relevant Resources

**Sources**
- `[gainski2025scent]` (arXiv:2506.19865) — SCENT: cost-aware, decomposability-guided
  backward policy (the ĉ_B^S / ĉ_B^D guidance models).
- `[malkin2022trajectorybalance]` (arXiv:2201.13259) — Trajectory Balance; the flow-recovery
  identity `F(h) = R(x)·∏P_B(h|x) / ∏P_F(x|h)` that motivates needing a faithful P_B.

**Packages**
- PyTorch — `nn.Module.state_dict()` recursion semantics (recurses `_modules`, not plain-list
  attributes) are the root cause; used by `guidance_io.save/load_guidance_models`.
- SCENT (`scent` conda env, its `rgfn` fork) — the `Trainer`, `objective`, and
  `valid_sampler` singletons the verifier drives.

### Method

All SCENT-env commands prefixed with `LD_LIBRARY_PATH` set to the `scent` env's
`nvidia/*/lib` + `torch/lib` (dgl/graphbolt need the torch-bundled CUDA libs).

1. **Confirm the drop (audit, this session).** Loaded
   `.../scent_seh/.../last_gfn.pt` and printed `checkpoint['model']` keys: 131 keys =
   `logZ` (shape (16,)) + 130 `forward_policy.*`; **0** keys matching cost/decompos/backward.
   Confirmed live that `type(objective.backward_policy.policies) == list` and each guidance
   model (262,913 params) has `in_state_dict == False`.
2. **Implement the fix.** Added `guidance_io.py`; wired `_save_guidance_models()` into
   `fixed_reward.py` (after `train()`) and `al_loop.py` (after each round's `train()`),
   writing `train/checkpoints/guidance_models.pt`.
3. **Verify recovery (login-safe, inference only):**
   ```
   conda run -n scent python validation/generators/scent/verify_pb_recovery.py \
       --checkpoint /scratch/.../scent_seh/2026-07-02_10-23-09/train/checkpoints/last_gfn.pt
   ```
   Loads the trained forward policy, samples 50 trajectories, then round-trips the guidance
   models: P_B_A → save sidecar → random reinit (P_B_corrupt) → load sidecar (P_B_restored).
4. **Full patched re-runs — SUBMITTED (2026-07-07)**, all four SCENT fixed-reward systems
   (the pre-fix checkpoints below), each driving `run_scent_fixed.py` → the patched
   `fixed_reward.py` → emits `guidance_models.pt` beside `last_gfn.pt`:
   ```
   sbatch experiments/fixed_reward/scent_seh/submit_fixed_scent_seh.sh                                  # job 70066 (seh)
   sbatch experiments/fixed_reward/scent_drd2/submit_fixed_scent_drd2.sh                                # job 70067 (drd2)
   sbatch experiments/fixed_reward/baseline_docking/submit_scent_docking.sh validation/configs/scent_6td3_fixed.gin  # job 70068
   sbatch experiments/fixed_reward/baseline_docking/submit_scent_docking.sh validation/configs/scent_clpp_fixed.gin  # job 70069
   ```
   Only SCENT is affected (RGFN/FragGFN/RxnFlow save their backward policy normally). Submit
   scripts unchanged. After completion, confirm exact P_B recovery on the genuinely-trained
   guidance weights via `verify_pb_recovery.py --checkpoint <new last_gfn.pt>`.
5. **Flag pre-fix runs.** Wrote `BACKWARD_POLICY_NOT_SAVED.txt` into each of the five
   pre-fix SCENT checkpoint dirs (those with `last_gfn.pt` but no `guidance_models.pt`).

### Results

**Checkpoint contents (step 1).** `checkpoint['model']`: 131 keys — `logZ` (present, trained:
sum = 74.33 vs fresh-init 37.5) + `forward_policy.*` (130). Guidance models: 2 × 262,913
params, none in `state_dict`.

**Round-trip recovery (step 3).** Batch: 50 trajectories, 690 backward steps, 195 C-phase
multi-parent choices (the steps where the guidance models actually shape P_B).

| Reload condition | max \|ΔlogP_B\| (all steps) | C-phase: max / mean | C-phase steps changed (>1e-6) |
|---|---|---|---|
| **corrupt** (reload WITHOUT sidecar) | 3.155e-01 | 3.155e-01 / 6.58e-02 | **146 / 195** |
| **restored** (reload WITH sidecar) | **0.000e+00** | **0.0 / 0.0** | **0 / 195** |

Verdict: `restored == saved` bit-exact (atol 1e-6) = **True**; `corrupt != saved` = **True** →
**PASS**. Because `load_state_dict` is value-agnostic, an exact restore of these weights
restores the trained P_B exactly. (The corrupt-vs-saved numbers reproduce the acid test's
independent reseed finding — 146/195 C-phase steps depend on the unsaved weights.)

**Pre-fix runs flagged (step 5).** `BACKWARD_POLICY_NOT_SAVED.txt` written to:
`active_learning/scent_6td3/2026-06-30_19-46-47`, `fixed_reward/scent_6td3/2026-07-03_14-34-22`,
`fixed_reward/scent_seh/2026-07-02_10-23-09`, `fixed_reward/scent_clpp/2026-07-03_20-55-41`,
`fixed_reward/scent_drd2/2026-07-02_11-11-14` (each `.../train/checkpoints/`).
