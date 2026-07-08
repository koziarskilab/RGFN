"""Pose-selection ablation: does picking the top-Vina pose hurt the Vina dT2-T1
oracle, vs. the production rule (top-CNNscore pose)?

Reads the PER-POSE CSVs from ``dock_allposes.py`` (every docked pose with its
Tier-2 and Tier-1 scores), then for each molecule applies two pose-selection
rules to the SAME poses:

  - "cnn"  : pose = argmax CNNscore (cnnsc_t2)   <- PRODUCTION rule (dock_cluster.py)
  - "vina" : pose = argmin Vina    (vina_t2)     <- the ablation ("just take the top pose")

For each rule it forms the validated differential ``ddb1_dvina = vina_t2 - vina_t1``
on the selected pose and scores how well it separates known glues from decoys
(Cohen's d, AUROC, Mann-Whitney U) — the exact discrimination measures used in
``Logs/006`` (experiments/ablations/sixway/discrimination_stats.py).

Key reads:
  - The "cnn" arm should reproduce the entry-006 Vina dT2-T1 result (AUROC ~0.946)
    up to docking stochasticity, serving as an in-run sanity control.
  - "vina" - "cnn" AUROC is the headline: how much (if any) discrimination is lost
    by selecting the top-Vina pose instead of the most-native-like (CNN) pose.
  - pose_agreement: fraction of molecules where the two rules pick the SAME pose.
    If high, the choice barely matters; if low, pose selection is doing real work.

Inputs default to per-pose CSVs alongside this script; override the directory with
$DATA_DIR (e.g. point at the scratch $OUTDIR after a Balam run). Outputs (stats CSV
+ figure) go to $OUT_DIR, defaulting to $DATA_DIR -- because $HOME (and thus the
repo) is READ-ONLY on Balam compute nodes, so on-cluster runs must write to scratch.
No docking here.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", HERE))
# $HOME (the repo) is read-only on Balam compute nodes -> outputs default to DATA_DIR (scratch).
OUT_DIR = Path(os.environ.get("OUT_DIR", DATA_DIR))
KNOWN_CSV = DATA_DIR / "known_allposes.csv"
DECOY_CSV = DATA_DIR / "decoy_allposes.csv"

# Entry-006 production baseline for the Vina dT2-T1 differential (CNN-selected pose).
BASELINE_CNN_AUROC = 0.946

# Pose-selection rules: (name, column, pick) — "pick" is the per-molecule selector.
RULES = [
    ("cnn", "cnnsc_t2", "max"),  # production: most native-like pose
    ("vina", "vina_t2", "min"),  # ablation: best Vina-scored pose ("top pose")
]


def load_poses(path):
    """Per-pose rows with status ok and finite scores needed for selection + differential."""
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    for c in ("vina_t2", "vina_t1", "cnnsc_t2"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["vina_t2", "vina_t1", "cnnsc_t2"])


def select(df, column, pick):
    """Per molecule (idx) pick one pose by `column` (`max`/`min`); return one row per molecule
    with the differential on the selected pose."""
    grp = df.groupby("idx", sort=True)
    sel_idx = grp[column].idxmax() if pick == "max" else grp[column].idxmin()
    sel = df.loc[sel_idx].copy()
    sel["dvina"] = sel["vina_t2"] - sel["vina_t1"]
    return sel.set_index("idx")


def cohens_d(a, b):
    """Pooled-SD standardized mean difference, a vs b (matches discrimination_stats.py)."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp


def discrimination(known_dv, decoy_dv):
    """Cohen's d / AUROC / MWU p for a lower-is-better Vina differential (oriented so
    higher = better-for-binder, matching discrimination_stats.py)."""
    b_or, d_or = -known_dv, -decoy_dv  # negate: Vina differential is lower-is-better
    d = cohens_d(b_or, d_or)
    U, p = mannwhitneyu(b_or, d_or, alternative="greater")
    auroc = U / (len(b_or) * len(d_or))
    return float(d), float(auroc), float(p)


def main():
    known, decoy = load_poses(KNOWN_CSV), load_poses(DECOY_CSV)
    nk, nd = known["idx"].nunique(), decoy["idx"].nunique()
    print(f"6TD3 pose-selection ablation — {nk} known / {nd} decoy molecules")
    print(f"  per-pose rows: {len(known)} known / {len(decoy)} decoy\n")

    selections = {}  # rule -> (known_sel, decoy_sel)
    rows = []
    for name, col, pick in RULES:
        ks, ds = select(known, col, pick), select(decoy, col, pick)
        selections[name] = (ks, ds)
        d, auroc, p = discrimination(ks["dvina"].values, ds["dvina"].values)
        rows.append(
            {
                "rule": name,
                "select_by": f"{pick}({col})",
                "n_known": len(ks),
                "n_decoy": len(ds),
                "known_median_dvina": float(np.median(ks["dvina"])),
                "decoy_median_dvina": float(np.median(ds["dvina"])),
                "cohens_d": d,
                "auroc": auroc,
                "mwu_p": p,
            }
        )

    summary = pd.DataFrame(rows)
    auroc = {r["rule"]: r["auroc"] for r in rows}
    summary["auroc_vs_cnn"] = summary["auroc"] - auroc["cnn"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "pose_selection_stats.csv"
    summary.to_csv(out_csv, index=False)

    # pose agreement: same selected pose_rank under both rules, per molecule
    def agreement(sel_a, sel_b):
        common = sel_a.index.intersection(sel_b.index)
        same = sel_a.loc[common, "pose_rank"].values == sel_b.loc[common, "pose_rank"].values
        return float(np.mean(same)) if len(common) else float("nan")

    ck, cd = selections["cnn"]
    vk, vd = selections["vina"]
    agree_known, agree_decoy = agreement(ck, vk), agreement(cd, vd)

    # ---- report ----
    print(
        f"{'rule':<6}{'select_by':<16}{'known_med':>10}{'decoy_med':>10}"
        f"{'cohen_d':>9}{'AUROC':>8}{'vs_cnn':>8}"
    )
    for r in rows:
        print(
            f"{r['rule']:<6}{r['select_by']:<16}{r['known_median_dvina']:>10.2f}"
            f"{r['decoy_median_dvina']:>10.2f}{r['cohens_d']:>9.2f}{r['auroc']:>8.3f}"
            f"{auroc[r['rule']] - auroc['cnn']:>8.3f}"
        )
    print(f"\nentry-006 production baseline (CNN-selected Vina dT2-T1): AUROC {BASELINE_CNN_AUROC}")
    print(
        f"in-run CNN control reproduces: AUROC {auroc['cnn']:.3f} "
        f"(delta {auroc['cnn'] - BASELINE_CNN_AUROC:+.3f} vs published; docking is stochastic)"
    )
    print(
        f"\npose agreement (same pose chosen by both rules): "
        f"known {agree_known:.1%}, decoy {agree_decoy:.1%}"
    )
    print(f"\nSaved: {out_csv}")

    # ---- figure: known vs decoy dvina violins, one panel per rule ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, (name, _, _) in zip(axes, RULES):
        ks, ds = selections[name]
        data = [ds["dvina"].values, ks["dvina"].values]
        parts = ax.violinplot(data, showmedians=True)
        for i, arr in enumerate(data, start=1):
            ax.scatter(np.random.normal(i, 0.04, len(arr)), arr, s=6, alpha=0.3, color="k")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["decoy", "known"])
        ax.set_title(f"{name}-selected pose  (AUROC {auroc[name]:.3f})")
        ax.axhline(-1.5, ls="--", lw=0.8, color="grey")  # entry-006 Youden-ish cut
    axes[0].set_ylabel("Vina dT2-T1 (ddb1_dvina); lower = better glue")
    fig.suptitle("6TD3 pose-selection ablation: CNNscore vs. Vina pose choice")
    fig.tight_layout()
    out_png = OUT_DIR / "pose_selection_violins.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
