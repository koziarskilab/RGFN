"""
Step 3 (the decisive test): does each oracle signal still discriminate once the
known and decoy sets are matched on molecular weight?

This is the airtightness step Entry 006 flagged. We build a size-balanced
comparison by 1:1 nearest-neighbor matching each known glue to a unique decoy of
near-identical MW (caliper 15 Da), then recompute the unit-free discrimination
(AUROC) on the matched subset and compare it to the full-set AUROC from Entry 006.

A metric whose Entry-006 edge was really ligand size will collapse toward AUROC
0.5 after matching; a metric reading genuine DDB1 cooperativity will hold up.

Because a single greedy match depends on visiting order, we bootstrap: B
randomized matchings, reporting the mean matched AUROC and a 95% percentile
interval per metric. One seeded match is also kept for the figure and the
MW-balance table.

Output:
  - mw_matched_auroc.png  — full vs. MW-matched AUROC per metric (bar pairs, CI).
  - mw_matched_auroc.csv  — full AUROC, matched AUROC mean + 95% CI, drop.
  - mw_match_balance.csv  — MW balance before/after matching (medians, MWU p).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mw_common as mw
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

N_BOOTSTRAP = 500
CALIPER = mw.DEFAULT_CALIPER


def main():
    known, decoy = mw.load(mw.KNOWN_CSV), mw.load(mw.DECOY_CSV)
    n_known, n_decoy = len(known), len(decoy)

    # Full-set AUROC (reproduces Entry 006) for every panel.
    full_auroc = {
        label: mw.auroc(known[col].to_numpy(), decoy[col].to_numpy(), better)
        for label, col, better in mw.PANELS
    }

    # ---- bootstrap MW-matched AUROC --------------------------------------
    boot = {label: [] for label, _, _ in mw.PANELS}
    pair_counts = []
    for b in range(N_BOOTSTRAP):
        rng = np.random.default_rng(b)
        km, dm = mw.greedy_mw_match(known, decoy, caliper=CALIPER, rng=rng)
        pair_counts.append(len(km))
        for label, col, better in mw.PANELS:
            boot[label].append(mw.auroc(km[col].to_numpy(), dm[col].to_numpy(), better))

    # One canonical seeded match for the MW-balance table.
    km0, dm0 = mw.greedy_mw_match(
        known, decoy, caliper=CALIPER, rng=np.random.default_rng(mw.DEFAULT_SEED)
    )
    _, p_before = mannwhitneyu(known["mw"], decoy["mw"], alternative="two-sided")
    _, p_after = mannwhitneyu(km0["mw"], dm0["mw"], alternative="two-sided")

    balance = pd.DataFrame(
        [
            {
                "stage": "before (full)",
                "n_known": n_known,
                "n_decoy": n_decoy,
                "known_mw_median": float(known["mw"].median()),
                "decoy_mw_median": float(decoy["mw"].median()),
                "mw_gap": float(known["mw"].median() - decoy["mw"].median()),
                "mwu_p": float(p_before),
            },
            {
                "stage": "after (matched)",
                "n_known": len(km0),
                "n_decoy": len(dm0),
                "known_mw_median": float(km0["mw"].median()),
                "decoy_mw_median": float(dm0["mw"].median()),
                "mw_gap": float(km0["mw"].median() - dm0["mw"].median()),
                "mwu_p": float(p_after),
            },
        ]
    )
    balance.to_csv(Path(__file__).parent / "mw_match_balance.csv", index=False)

    # ---- assemble AUROC table --------------------------------------------
    rows = []
    for label, col, better in mw.PANELS:
        arr = np.array(boot[label])
        rows.append(
            {
                "panel": label,
                "column": col,
                "better": better,
                "auroc_full": full_auroc[label],
                "auroc_matched_mean": float(arr.mean()),
                "auroc_matched_lo95": float(np.percentile(arr, 2.5)),
                "auroc_matched_hi95": float(np.percentile(arr, 97.5)),
                "auroc_drop": float(full_auroc[label] - arr.mean()),
            }
        )
    df = pd.DataFrame(rows).sort_values("auroc_matched_mean", ascending=False)
    df = df.reset_index(drop=True)
    df.to_csv(Path(__file__).parent / "mw_matched_auroc.csv", index=False)

    # ---- figure: full vs matched AUROC -----------------------------------
    order = df["panel"].tolist()
    full_vals = df["auroc_full"].to_numpy()
    matched_vals = df["auroc_matched_mean"].to_numpy()
    lo = matched_vals - df["auroc_matched_lo95"].to_numpy()
    hi = df["auroc_matched_hi95"].to_numpy() - matched_vals

    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        x - w / 2, full_vals, w, label="full set (Entry 006)", color="#CBD5E1", edgecolor="#475569"
    )
    ax.bar(
        x + w / 2,
        matched_vals,
        w,
        yerr=[lo, hi],
        capsize=4,
        label=f"MW-matched (mean of {N_BOOTSTRAP} matchings, 95% CI)",
        color=mw.BINDER_COLOR,
        edgecolor="#1E3A8A",
        alpha=0.85,
    )
    ax.axhline(0.5, color="#991B1B", linestyle="--", linewidth=1.1, label="chance (0.5)")

    for xi, fv, mvv in zip(x, full_vals, matched_vals):
        ax.text(xi - w / 2, fv + 0.012, f"{fv:.2f}", ha="center", fontsize=8.5)
        ax.text(xi + w / 2, mvv + 0.012, f"{mvv:.2f}", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9.5)
    ax.set_ylabel("AUROC (binder vs. decoy)", fontsize=11)
    ax.set_ylim(0.3, 1.0)
    ax.set_title(
        "6TD3 / CDK12-DDB1 — discrimination before vs. after MW-matching\n"
        f"matched on MW (caliper {CALIPER:.0f} Da, ~{int(np.median(pair_counts))} pairs); "
        "the Tier 2 − Tier 1 differentials survive, the absolute scores do not",
        fontsize=12,
        pad=8,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out_png = Path(__file__).parent / "mw_matched_auroc.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")

    # ---- console summary --------------------------------------------------
    print(
        f"6TD3 / CDK12-DDB1 — MW-matched discrimination "
        f"(caliper {CALIPER:.0f} Da, {N_BOOTSTRAP} bootstraps)"
    )
    print(
        f"matched pairs: median {int(np.median(pair_counts))} "
        f"(of {n_known} known / {n_decoy} decoy)"
    )
    print(
        f"MW balance: gap {known['mw'].median() - decoy['mw'].median():+.1f} Da "
        f"(p {p_before:.1e})  ->  {km0['mw'].median() - dm0['mw'].median():+.1f} Da "
        f"(p {p_after:.2f}) after matching\n"
    )
    print(f"{'panel':<13}{'full':>7}{'matched':>9}{'95% CI':>16}{'drop':>8}")
    for _, r in df.iterrows():
        ci = f"[{r['auroc_matched_lo95']:.2f},{r['auroc_matched_hi95']:.2f}]"
        print(
            f"{r['panel']:<13}{r['auroc_full']:>7.3f}{r['auroc_matched_mean']:>9.3f}"
            f"{ci:>16}{r['auroc_drop']:>+8.3f}"
        )
    print(f"\nSaved: {out_png}")
    print(f"Saved: {Path(__file__).parent / 'mw_matched_auroc.csv'}")
    print(f"Saved: {Path(__file__).parent / 'mw_match_balance.csv'}")


if __name__ == "__main__":
    main()
