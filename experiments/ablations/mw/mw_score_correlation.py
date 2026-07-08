"""
Step 2 of the MW control: which oracle signals actually track molecular weight?

A confound only bites a metric that (a) differs between the sets in MW — shown in
mw_distributions.py — and (b) is itself correlated with MW. This script regresses
each of the six oracle signals on MW (pooling known + decoy) and reports Spearman
and Pearson correlations, so we can see which signals are size-driven.

The expected, and decisive, pattern: the absolute Vina scores (especially Tier 1)
correlate strongly with MW — heavier ligand, more contacts, more negative Vina —
while the Tier 2 - Tier 1 differentials correlate much more weakly, because the
size term largely cancels between the two tiers. That is the mechanistic reason
the differential should survive MW-matching (tested in mw_matched_discrimination.py).

Output:
  - mw_score_correlation.png — 2x3 scatter grid (MW vs each metric) with fit line.
  - mw_score_correlation.csv — Spearman/Pearson r and p per metric, plus the
    within-group Spearman (known-only, decoy-only) to show it is not just a
    between-group effect.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mw_common as mw
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def main():
    known, decoy = mw.load(mw.KNOWN_CSV), mw.load(mw.DECOY_CSV)
    all_mw = np.concatenate([known["mw"].to_numpy(), decoy["mw"].to_numpy()])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    rows = []
    for ax, (label, col, better) in zip(axes.flat, mw.PANELS):
        kx, ky = known["mw"].to_numpy(), known[col].to_numpy()
        dx, dy = decoy["mw"].to_numpy(), decoy[col].to_numpy()
        all_y = np.concatenate([ky, dy])

        ax.scatter(
            dx, dy, s=8, color=mw.DECOY_COLOR, alpha=0.4, linewidths=0, label="decoys", zorder=2
        )
        ax.scatter(
            kx, ky, s=8, color=mw.BINDER_COLOR, alpha=0.5, linewidths=0, label="binders", zorder=3
        )

        # Pooled least-squares fit line across the full MW range.
        b1, b0 = np.polyfit(all_mw, all_y, 1)
        xs = np.linspace(all_mw.min(), all_mw.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color="#111827", linewidth=1.4, alpha=0.8, zorder=4)

        rho, p_rho = spearmanr(all_mw, all_y)
        r, p_r = pearsonr(all_mw, all_y)
        rho_k = spearmanr(kx, ky)[0]
        rho_d = spearmanr(dx, dy)[0]
        rows.append(
            {
                "panel": label,
                "column": col,
                "better": better,
                "spearman_pooled": float(rho),
                "spearman_p": float(p_rho),
                "pearson_pooled": float(r),
                "pearson_p": float(p_r),
                "spearman_known_only": float(rho_k),
                "spearman_decoy_only": float(rho_d),
            }
        )

        arrow = "↓ better" if better == "lower" else "↑ better"
        ax.set_title(
            f"{label}  ({arrow})\n"
            f"ρ pooled {rho:+.2f}  |  within-group {rho_k:+.2f}/{rho_d:+.2f}",
            fontsize=10,
            pad=6,
        )
        ax.set_xlabel("Molecular weight (Da)", fontsize=9)
        ax.set_ylabel(col, fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_axisbelow(True)

    axes.flat[0].legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle(
        "6TD3 / CDK12-DDB1 — oracle signal vs. molecular weight\n"
        "within each group (known / decoy), the Tier 2 − Tier 1 differentials are "
        "the least size-dependent signals",
        fontsize=12.5,
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = Path(__file__).parent / "mw_score_correlation.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")

    df = pd.DataFrame(rows)
    out_csv = Path(__file__).parent / "mw_score_correlation.csv"
    df.to_csv(out_csv, index=False)

    print("6TD3 / CDK12-DDB1 — correlation of each oracle signal with MW (pooled):\n")
    print(f"{'panel':<13}{'rho':>8}{'r':>8}{'rho|known':>11}{'rho|decoy':>11}")
    for r in rows:
        print(
            f"{r['panel']:<13}{r['spearman_pooled']:>8.2f}{r['pearson_pooled']:>8.2f}"
            f"{r['spearman_known_only']:>11.2f}{r['spearman_decoy_only']:>11.2f}"
        )
    print(f"\nSaved: {out_png}\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
