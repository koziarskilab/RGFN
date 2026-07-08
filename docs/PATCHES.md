# Patches to upstream files

We keep `rgfn/` and `configs/` mergeable with upstream RGFN. New functionality is
added via the `glue/` package and `configs/glue/` overlays — **not** by editing
upstream. The exceptions below are small operational overrides we chose to apply
directly to upstream files. They are listed here so they are easy to find, review,
and re-apply after an upstream merge/rebase.

If you pull upstream changes and one of these reverts, re-apply it from here.

---

## 1. `rgfn/trainer/logger/wandb_logger.py` — drop `dir=` from `wandb.init`

**Change:** removed the `dir=self.logdir,` argument from the `wandb.init(...)`
call.

```diff
         return wandb.init(
-            dir=self.logdir,
             project=self.project_name,
             name=self.experiment_name,
             group=group,
```

**Why:** on Balam we run wandb in offline mode with cache/dir env vars
(`WANDB_DIR`, etc., set in `scripts/submit.sh`). Passing `dir=self.logdir`
conflicted with that setup.

**Status:** kept as a one-line patch (intentionally not refactored into a `glue/`
subclass — it's a single line and low-churn).

---

## 2. `configs/loggers/wandb.gin` — default to offline

**Change:**

```diff
-WandbLogger.mode = 'online'
+WandbLogger.mode = 'offline'
```

**Why:** Balam compute nodes have no/limited network; wandb runs offline and is
synced later.

**Note:** could alternatively live as a `configs/glue/` overlay. Left in place for
now because offline is the desired default for all our runs.

---

## 3. `configs/rgfn_seh_docking.gin` — cap docking-run iterations

**Change:** appended

```gin
# GPU docking is ~160s/iter (vs ~10s for the neural proxy), so the base config's
# 5002 iterations would take ~9 days. Cap at 400 (~18h) to complete within the
# 20h SLURM walltime in submit.sh. Override here only; proxy configs keep 5002.
Trainer.n_iterations = 400
```

**Why:** make a docking-oracle run fit inside the Balam walltime.

**Note:** this is a docking-specific config; the cleaner long-term home is a
`configs/glue/` overlay. Left in place to preserve current run behavior.

---

## Convention going forward

Prefer adding new behavior in `glue/` + `configs/glue/`. Only patch upstream when
there is no reasonable override point, and when you do, **add an entry here**.
