# data/synthetic/

Generated synthetic datasets — the **outputs** of sampling from trained RGFN runs
(molecules + their oracle scores), produced by tooling in `glue/datasets/`.

- Upstream input data stays at the top of `data/` (`chemistry.xlsx`, `targets/`).
- Generated artifacts go here and are **gitignored** (see `.gitignore`); commit
  only small curated samples or a manifest describing how to regenerate them.

Each generation run should record the config, checkpoint, and command used (an
`Logs/` entry is the right place for anything significant).
