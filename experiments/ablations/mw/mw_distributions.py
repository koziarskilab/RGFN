"""
Step 1 of the MW control: does a size confound even exist?

Before MW-matching can matter, we have to show the known glues and decoys differ
in molecular weight in the first place. This script draws the known-vs-decoy MW
distributions (violin + jittered points, mirroring the Entry 006 figures) and
reports the size gap on a unit-free footing (Cohen's d, AUROC) plus Mann-Whitney
U and Kolmogorov-Smirnov tests.

Output:
  - mw_distribution.png    — violin of MW, decoys vs. binders.
  - mw_distribution.csv    — n, median, mean, sd, Cohen's d, AUROC, MWU/KS p.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mw_common as mw
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu


def cohens_d(a, b):
    """Pooled-SD standardized mean difference, a vs b."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp)


def main():
    known, decoy = mw.load(mw.KNOWN_CSV), mw.load(mw.DECOY_CSV)
    k_mw, d_mw = known["mw"].to_numpy(), decoy["mw"].to_numpy()
    n_known, n_decoy = len(k_mw), len(d_mw)

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 6))
    data = [d_mw, k_mw]
    colors = [mw.DECOY_COLOR, mw.BINDER_COLOR]
    positions = [1, 2]
    parts = ax.violinplot(
        data, positions=positions, showmedians=True, showextrema=False, widths=0.8
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
    parts["cmedians"].set_color("#111827")
    parts["cmedians"].set_linewidth(1.6)

    rng = np.random.default_rng(0)
    for pos, vals, color in zip(positions, data, colors):
        x = pos + (rng.random(len(vals)) - 0.5) * 0.18
        ax.scatter(x, vals, s=7, color=color, alpha=0.35, linewidths=0, zorder=3)
        med = float(np.median(vals))
        ax.annotate(
            f"med {med:.0f}",
            (pos, med),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#111827",
        )

    # Mark the decoy MW ceiling — knowns above it have no possible size match.
    d_max = float(d_mw.max())
    ax.axhline(d_max, color=mw.MW_ACCENT, linestyle="--", linewidth=1.2, alpha=0.8)
    n_above = int((k_mw > d_max).sum())
    ax.annotate(
        f"decoy MW ceiling {d_max:.0f}\n({n_above} knowns above → unmatchable)",
        (2.0, d_max),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        va="bottom",
        fontsize=8,
        color=mw.MW_ACCENT,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"decoys\n(n={n_decoy})", f"binders\n(n={n_known})"], fontsize=10)
    ax.set_ylabel("Molecular weight (Da)", fontsize=11)
    ax.set_title(
        "6TD3 / CDK12-DDB1 — molecular-weight distributions\n"
        "known glues are systematically heavier than decoys",
        fontsize=12,
        pad=8,
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()

    out_png = Path(__file__).parent / "mw_distribution.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")

    # ---- stats ------------------------------------------------------------
    # AUROC oriented as "binder heavier than decoy" (higher MW = more binder-like).
    auroc_mw = mw.auroc(k_mw, d_mw, "higher")
    d = cohens_d(known["mw"], decoy["mw"])
    _, p_mwu = mannwhitneyu(k_mw, d_mw, alternative="two-sided")
    ks_stat, p_ks = ks_2samp(k_mw, d_mw)

    import pandas as pd

    rows = [
        {
            "set": "known",
            "n": n_known,
            "median": float(np.median(k_mw)),
            "mean": float(k_mw.mean()),
            "sd": float(k_mw.std(ddof=1)),
            "min": float(k_mw.min()),
            "max": float(k_mw.max()),
        },
        {
            "set": "decoy",
            "n": n_decoy,
            "median": float(np.median(d_mw)),
            "mean": float(d_mw.mean()),
            "sd": float(d_mw.std(ddof=1)),
            "min": float(d_mw.min()),
            "max": float(d_mw.max()),
        },
        {
            "set": "known_minus_decoy",
            "n": np.nan,
            "median": float(np.median(k_mw) - np.median(d_mw)),
            "mean": float(k_mw.mean() - d_mw.mean()),
            "sd": np.nan,
            "min": np.nan,
            "max": np.nan,
            "cohens_d": d,
            "auroc": auroc_mw,
            "mwu_p": float(p_mwu),
            "ks_stat": float(ks_stat),
            "ks_p": float(p_ks),
        },
    ]
    out_csv = Path(__file__).parent / "mw_distribution.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print("6TD3 / CDK12-DDB1 — molecular-weight comparison")
    print(
        f"  known: n={n_known}  median {np.median(k_mw):.1f}  mean {k_mw.mean():.1f}"
        f"  sd {k_mw.std(ddof=1):.1f}  range {k_mw.min():.0f}-{k_mw.max():.0f}"
    )
    print(
        f"  decoy: n={n_decoy}  median {np.median(d_mw):.1f}  mean {d_mw.mean():.1f}"
        f"  sd {d_mw.std(ddof=1):.1f}  range {d_mw.min():.0f}-{d_mw.max():.0f}"
    )
    print(f"  size gap (known - decoy median): {np.median(k_mw) - np.median(d_mw):+.1f} Da")
    print(
        f"  Cohen's d {d:+.2f}   AUROC(MW) {auroc_mw:.3f}" f"   MWU p {p_mwu:.1e}   KS p {p_ks:.1e}"
    )
    print(f"  knowns above decoy MW ceiling ({d_max:.0f} Da): {n_above}")
    print(f"\nSaved: {out_png}\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
