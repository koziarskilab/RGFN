"""Analyse the GPU pose-generation re-dock vs the entry-006 gnina-search baseline.

Two questions:
  1. DISCRIMINATION — does the GPU-search differential separate known glues from
     decoys as well as 006's CPU-search differential (AUROC 0.946, Cohen's d 2.38)?
     We recompute AUROC / Cohen's d / Mann-Whitney on the GPU ``dvina`` and, as an
     internal control, on the ``ref_dvina`` column (should reproduce ~0.946 on
     exactly these molecules).
  2. AGREEMENT — does the GPU differential track the gnina-search differential
     per molecule (Pearson/Spearman of dvina vs ref_dvina)? High discrimination
     with low per-molecule agreement would mean "different poses, same separation".

Reads the dock_gpu.py output CSV; writes a stats CSV and a scatter PNG. Pure
pandas/numpy/scipy/sklearn (matplotlib headless) — runs anywhere, no GPU.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

BASELINE_AUROC = 0.946  # entry-006 Vina dT2-T1 (CNN-selected pose)
BASELINE_D = 2.38


def cohens_d(binders, decoys):
    """Oriented so positive = binders score better. dvina is lower-is-better, so
    binders should have the more-negative mean; we negate to make 'better' = higher."""
    a, b = -np.asarray(binders), -np.asarray(decoys)  # higher = better
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else float("nan")


def discrimination(df, col):
    """AUROC / Cohen's d / MWU p for a lower-is-better column, known vs decoy."""
    k = df.loc[df["set"] == "known", col].to_numpy()
    d = df.loc[df["set"] == "decoy", col].to_numpy()
    y = np.r_[np.ones_like(k), np.zeros_like(d)]
    score = -np.r_[k, d]  # higher = better glue
    auroc = roc_auc_score(y, score)
    _, p = mannwhitneyu(-k, -d, alternative="greater")
    return {
        "n_known": len(k),
        "n_decoy": len(d),
        "known_median": float(np.median(k)),
        "decoy_median": float(np.median(d)),
        "cohens_d": cohens_d(k, d),
        "auroc": auroc,
        "mwu_p": p,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="dock_gpu.py output CSV")
    ap.add_argument(
        "--out-dir", default=None, help="where to write stats/plots (default: results dir)"
    )
    args = ap.parse_args()

    res = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else res.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(res)
    n_total = len(df)
    df = df[df["status"] == "ok"].copy()
    print(
        f"[load] {len(df)}/{n_total} ok rows "
        f"({(df['set']=='known').sum()} known / {(df['set']=='decoy').sum()} decoy)"
    )

    # 1. Discrimination: GPU dvina vs the 006 reference dvina (same molecules).
    rows = []
    for label, col in [
        ("GPU dvina (QV2 search)", "dvina"),
        ("ref dvina (gnina search, recomputed)", "ref_dvina"),
    ]:
        s = discrimination(df, col)
        s["metric"] = label
        rows.append(s)
        print(
            f"  {label:<40} AUROC={s['auroc']:.3f}  d={s['cohens_d']:.2f}  "
            f"median known/decoy={s['known_median']:.2f}/{s['decoy_median']:.2f}"
        )
    stats = pd.DataFrame(rows)[
        [
            "metric",
            "n_known",
            "n_decoy",
            "known_median",
            "decoy_median",
            "cohens_d",
            "auroc",
            "mwu_p",
        ]
    ]
    stats.to_csv(out_dir / "discrimination_stats.csv", index=False)

    gpu_auroc = float(stats.loc[stats["metric"].str.startswith("GPU"), "auroc"].iloc[0])
    ctrl_auroc = float(stats.loc[stats["metric"].str.startswith("ref"), "auroc"].iloc[0])

    # 2. Per-molecule agreement: GPU dvina vs ref dvina.
    pr, pp = pearsonr(df["dvina"], df["ref_dvina"])
    sr, sp = spearmanr(df["dvina"], df["ref_dvina"])
    bias = float((df["dvina"] - df["ref_dvina"]).mean())
    rmse = float(np.sqrt(((df["dvina"] - df["ref_dvina"]) ** 2).mean()))
    agree = pd.DataFrame(
        [
            {
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_r": sr,
                "spearman_p": sp,
                "mean_bias_gpu_minus_ref": bias,
                "rmse": rmse,
                "n": len(df),
            }
        ]
    )
    agree.to_csv(out_dir / "agreement_stats.csv", index=False)

    # scatter: GPU vs ref differential, colored by set.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for setl, c in [("decoy", "#cc6677"), ("known", "#4477aa")]:
        sub = df[df["set"] == setl]
        ax.scatter(sub["ref_dvina"], sub["dvina"], s=14, alpha=0.6, c=c, label=setl)
    lim = [
        min(df["ref_dvina"].min(), df["dvina"].min()) - 0.5,
        max(df["ref_dvina"].max(), df["dvina"].max()) + 0.5,
    ]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5, label="y = x")
    ax.axhline(-1.5, color="grey", lw=0.6, ls=":")
    ax.axvline(-1.5, color="grey", lw=0.6, ls=":")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("entry-006 dvina  (gnina CPU search)")
    ax.set_ylabel("GPU dvina  (QuickVina2-GPU search)")
    ax.set_title(
        f"DDB1 differential: GPU vs gnina-search\n"
        f"AUROC GPU={gpu_auroc:.3f} (ref ctrl {ctrl_auroc:.3f}, 006={BASELINE_AUROC})  "
        f"Pearson r={pr:.2f}"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "gpu_vs_gnina_scatter.png", dpi=140)

    # verdict line
    print(
        f"\n[discrimination] GPU AUROC={gpu_auroc:.3f}  vs  006 baseline {BASELINE_AUROC} "
        f"(internal ref control {ctrl_auroc:.3f})"
    )
    print(
        f"[agreement]      Pearson r={pr:.3f}  Spearman r={sr:.3f}  "
        f"bias(GPU-ref)={bias:+.3f}  RMSE={rmse:.3f} kcal/mol"
    )
    print(
        f"[wrote] {out_dir/'discrimination_stats.csv'}, {out_dir/'agreement_stats.csv'}, "
        f"{out_dir/'gpu_vs_gnina_scatter.png'}"
    )


if __name__ == "__main__":
    main()
