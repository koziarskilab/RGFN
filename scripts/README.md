# scripts/

The **generic launch layer** — pipeline-wide entry points and nothing else.
These are **ours** (not upstream).

| Script | Purpose |
|---|---|
| `train.py` | Training wrapper — imports `glue` (so gin sees our components), then runs the upstream root `train.py`. Use for any config under `configs/glue/`. |
| `infer.py` | Inference wrapper — same idea, delegates to the root `infer.py`. |
| `active_learning.py` | Active-learning loop entry point — imports `glue`, parses an AL config, builds an `ActiveLearningLoop` and runs it. Drives *any* `configs/glue/*` AL config (6TD3, sEH, mock). |
| `submit.sh` | **Generic** Balam SLURM submit (training). Run-specific submit scripts live with their experiment, not here. |

## What belongs here — the generality test

`scripts/` holds only things that are **pipeline-wide**: they work for any system
or experiment. If a script is tied to one experiment/system, it lives **with that
experiment** under `experiments/<group>/<run>/` instead — not here. Precedents:

- `experiments/active_learning/seh/` — sEH seed-gen, live validation, compute-node submit.
- `experiments/active_learning/6td3/` — the 6TD3 mini-run submit script.
- `experiments/ablations/pose_selection/` — one ablation's scripts + submit.

This keeps `scripts/` small and meaningful: *"here is how you launch the pipeline."*
(There is no `hpc/` subdir — a single generic `submit.sh` doesn't need a folder.)

## Why the train/infer wrappers exist

Gin discovers components by class name, but only after the defining module is
imported. Upstream `train.py` imports only `rgfn`, so it cannot see anything in
`glue/`. These wrappers `import glue` first (registering our oracles, rewards,
samplers, proxies) and then delegate to the unmodified upstream entry point (via
`runpy`, so the upstream entry stays the source of truth).

Plain upstream RGFN configs can still be run with the root `train.py`/`infer.py`
directly; use these wrappers whenever the config references a `glue` component.
(`active_learning.py` is not a wrapper — it's our own driver for the multi-round
loop, which has no upstream equivalent.)
