"""
ROC and Precision-Recall curves for Tier 2 Vina as a glue discriminator.
6TD3 (CDK12-DDB1) and 5HXB (CRBN-GSPT1) — Experiment 003 data.

Lower Vina = stronger predicted binding, so scores are negated for sklearn
(higher negated score → more positive class). Only status='ok' rows included.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[4]  # RGFN-Fork root
PRE = ROOT / "experiments" / "oracle_validation"

DATA = {
    "6TD3 (CDK12–DDB1)": {
        "known": PRE / "docking_6td3" / "known_results.csv",
        "decoy": PRE / "docking_6td3" / "decoy_cdk_results.csv",
        "color": "#2563EB",
    },
    "5HXB (CRBN–GSPT1)": {
        "known": PRE / "docking_crbn" / "known_crbn_results.csv",
        "decoy": PRE / "docking_crbn" / "decoy_crbn_results.csv",
        "color": "#DC2626",
    },
}


def load(cfg):
    known = pd.read_csv(cfg["known"])
    decoy = pd.read_csv(cfg["decoy"])
    known = known[known["status"] == "ok"]
    decoy = decoy[decoy["status"] == "ok"]
    scores = pd.concat([known["vina_t2"], decoy["vina_t2"]], ignore_index=True).values
    labels = np.concatenate([np.ones(len(known)), np.zeros(len(decoy))])
    return -scores, labels, len(known), len(decoy)  # negate: lower Vina = more positive


fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for label, cfg in DATA.items():
    scores, labels, n_known, n_decoy = load(cfg)
    color = cfg["color"]

    # ROC
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(
        fpr, tpr, color=color, lw=2, label=f"{label}\nAUC = {roc_auc:.3f}  (n={n_known}/{n_decoy})"
    )

    # PR
    prec, rec, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    baseline = n_known / (n_known + n_decoy)
    axes[1].plot(
        rec, prec, color=color, lw=2, label=f"{label}\nAP = {ap:.3f}  (baseline {baseline:.2f})"
    )

# ROC panel
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.45)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC curve\nTier 2 Vina — known glues vs. decoys", fontsize=12, pad=8)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.4)

# PR panel
ax = axes[1]
# baseline references
for label, cfg in DATA.items():
    _, labels_ref, n_known, n_decoy = load(cfg)
    baseline = n_known / (n_known + n_decoy)
    ax.axhline(baseline, color=cfg["color"], lw=1, linestyle="--", alpha=0.4)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall curve\nTier 2 Vina — known glues vs. decoys", fontsize=12, pad=8)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.4)

plt.tight_layout(pad=2)

out = Path(__file__).parent / "discrimination_curves.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

print("\nSummary (Tier 2 Vina, status=ok only):")
print(
    f"{'System':<28} {'ROC-AUC':>8} {'Avg-Prec':>9} {'n_known':>8} {'n_decoy':>8} {'baseline':>9}"
)
for label, cfg in DATA.items():
    scores, labels, n_known, n_decoy = load(cfg)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(labels, scores)
    baseline = n_known / (n_known + n_decoy)
    print(f"{label:<28} {roc_auc:>8.3f} {ap:>9.3f} {n_known:>8} {n_decoy:>8} {baseline:>9.3f}")
