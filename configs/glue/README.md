# configs/glue/

Our gin configs. **All new configs go here**, not in the upstream `configs/` tree.

## The overlay pattern

Upstream configs (`configs/*.gin`, `configs/**/*.gin`) are treated as pristine.
To build on one, create a config here that `include`s the upstream base and then
overrides or adds bindings:

```gin
# configs/glue/rgfn_my_oracle.gin
include 'configs/rgfn_base.gin'
include 'configs/envs/reaction.gin'

# point the reward at one of our proxies (registered via glue.registry)
proxy/gin.singleton.constructor = @ExampleGlueProxy
train_proxy = @proxy/gin.singleton()
valid_proxy = %train_proxy
```

Run it with the wrapper that registers our components first:

```bash
python scripts/train.py --cfg configs/glue/rgfn_my_oracle.gin
```

## Note on existing upstream config edits

Three small operational overrides were applied directly to upstream files rather
than via this overlay (offline wandb, a docking walltime cap, and a wandb logger
one-liner). They are documented in `docs/PATCHES.md`. New deviations should
prefer this overlay directory so upstream stays mergeable.
