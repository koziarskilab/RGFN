"""
Violin plots: known glues vs. decoys on the 6TD3 (CDK12-DDB1) oracle.

Reuses the Experiment 002 docking results (no new docking) —
experiments/oracle_validation/docking_6td3/{known_results,decoy_cdk_results}.csv —
and renders six two-violin comparisons (decoys vs. binders):

  1. Vina  Tier 2            (vina_t2)
  2. Vina  Tier 1            (vina_t1)
  3. CNNaff Tier 2           (cnnaff_t2)
  4. CNNaff Tier 1           (cnnaff_t1)
  5. Vina  Tier 2 - Tier 1   (ddb1_dvina  == vina_t2  - vina_t1)
  6. CNNaff Tier 2 - Tier 1  (ddb1_dcnnaff == cnnaff_t2 - cnnaff_t1)

Vina: lower = stronger binding (negative is better).
CNNaff: higher = stronger predicted affinity.
Only status='ok' rows are used.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # RGFN-Fork root
T2D3 = ROOT / "experiments" / "oracle_validation" / "docking_6td3"

KNOWN_CSV = T2D3 / "known_results.csv"
DECOY_CSV = T2D3 / "decoy_cdk_results.csv"

DECOY_COLOR = "#94A3B8"  # slate — fake glues (correct warhead, random arm)
BINDER_COLOR = "#2563EB"  # blue — known CDK12-DDB1 glues


def load(path):
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    # Differentials (Tier 2 - Tier 1) on the same pose. Stored already, but
    # recompute so the plotted quantity is unambiguous.
    df["dvina"] = df["vina_t2"] - df["vina_t1"]
    df["dcnnaff"] = df["cnnaff_t2"] - df["cnnaff_t1"]
    return df


known = load(KNOWN_CSV)
decoy = load(DECOY_CSV)
n_known, n_decoy = len(known), len(decoy)

# (title, column, y-axis label, "lower"/"higher" = better)
PANELS = [
    # top row: Vina — Tier 1, Tier 2, delta
    ("Vina — Tier 1 (CDK12 only)", "vina_t1", "Vina (kcal/mol)", "lower"),
    ("Vina — Tier 2 (CDK12+DDB1)", "vina_t2", "Vina (kcal/mol)", "lower"),
    ("Vina — Tier 2 − Tier 1 (DDB1 ΔVina)", "dvina", "ΔVina (kcal/mol)", "lower"),
    # bottom row: CNN — Tier 1, Tier 2, delta
    ("CNN affinity — Tier 1 (CDK12 only)", "cnnaff_t1", "CNN affinity (pK)", "higher"),
    ("CNN affinity — Tier 2 (CDK12+DDB1)", "cnnaff_t2", "CNN affinity (pK)", "higher"),
    ("CNN affinity — Tier 2 − Tier 1 (DDB1 ΔCNN)", "dcnnaff", "ΔCNN affinity (pK)", "higher"),
]


def draw_panel(ax, title, col, ylabel, better):
    decoy_vals = decoy[col].values
    known_vals = known[col].values
    data = [decoy_vals, known_vals]
    colors = [DECOY_COLOR, BINDER_COLOR]
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

    # Jittered raw points + median annotation.
    rng = np.random.default_rng(0)
    for pos, vals, color in zip(positions, data, colors):
        x = pos + (rng.random(len(vals)) - 0.5) * 0.18
        ax.scatter(x, vals, s=6, color=color, alpha=0.35, linewidths=0, zorder=3)
        med = float(np.median(vals))
        ax.annotate(
            f"med {med:.2f}",
            (pos, med),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#111827",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"decoys\n(n={n_decoy})", f"binders\n(n={n_known})"], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    arrow = "↓ better" if better == "lower" else "↑ better"
    ax.set_title(f"{title}\n({arrow})", fontsize=10.5, pad=6)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (title, col, ylabel, better) in zip(axes.flat, PANELS):
    draw_panel(ax, title, col, ylabel, better)

fig.suptitle(
    "6TD3 / CDK12-DDB1 oracle — known glues vs. decoys "
    f"(Experiment 002 docking data, n={n_known} known / {n_decoy} decoy)",
    fontsize=13,
    y=0.995,
)
plt.tight_layout(rect=(0, 0, 1, 0.97))

out = Path(__file__).parent / "violins_known_vs_decoy.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

# Per-panel medians + known-minus-decoy gap, written alongside the figure.
print(f"\n{'panel':<40}{'decoy med':>11}{'known med':>11}{'gap (k-d)':>11}")
rows = []
for title, col, ylabel, better in PANELS:
    dm, km = float(np.median(decoy[col])), float(np.median(known[col]))
    gap = km - dm
    rows.append((title, col, dm, km, gap, better))
    print(f"{title:<40}{dm:>11.3f}{km:>11.3f}{gap:>+11.3f}")

summary = Path(__file__).parent / "violin_medians.csv"
pd.DataFrame(
    rows,
    columns=[
        "panel",
        "column",
        "decoy_median",
        "known_median",
        "gap_known_minus_decoy",
        "better_direction",
    ],
).to_csv(summary, index=False)
print(f"Saved: {summary}")
