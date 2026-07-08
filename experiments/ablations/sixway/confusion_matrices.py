"""
Confusion matrices for the 6TD3 (CDK12-DDB1) oracle, one per score/tier,
over all docked samples (Experiment 002 data; no new docking).

A confusion matrix needs a decision threshold. We pick, per metric, the
Youden-J-optimal cut on the ROC curve (max sensitivity + specificity - 1) —
reproducible and threshold-tuning-free. Positive class = known binders,
negative class = decoys. Vina is lower-is-better so its scores are negated
before thresholding; CNN affinity is higher-is-better.

This is a descriptive, in-sample summary (the threshold is fit on the same data
it is scored on), not a held-out classifier evaluation — it shows how separable
the populations are at each metric's best single cut. Only status='ok' rows used.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

ROOT = Path(__file__).resolve().parents[3]  # RGFN-Fork root
T2D3 = ROOT / "experiments" / "oracle_validation" / "docking_6td3"
KNOWN_CSV = T2D3 / "known_results.csv"
DECOY_CSV = T2D3 / "decoy_cdk_results.csv"

PANELS = [
    # top row: Vina — Tier 1, Tier 2, delta
    ("Vina Tier 1", "vina_t1", "lower"),
    ("Vina Tier 2", "vina_t2", "lower"),
    ("Vina ΔT2−T1", "dvina", "lower"),
    # bottom row: CNN — Tier 1, Tier 2, delta
    ("CNN Tier 1", "cnnaff_t1", "higher"),
    ("CNN Tier 2", "cnnaff_t2", "higher"),
    ("CNN ΔT2−T1", "dcnnaff", "higher"),
]


def load(path):
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["dvina"] = df["vina_t2"] - df["vina_t1"]
    df["dcnnaff"] = df["cnnaff_t2"] - df["cnnaff_t1"]
    return df


def youden_threshold(y_true, oriented_score):
    """Return (oriented_thr, accuracy, sens, spec) at the max-Youden ROC point."""
    fpr, tpr, thr = roc_curve(y_true, oriented_score)
    j = np.argmax(tpr - fpr)
    return thr[j], tpr[j], fpr[j]


def main():
    known, decoy = load(KNOWN_CSV), load(DECOY_CSV)
    y_true = np.concatenate([np.ones(len(known)), np.zeros(len(decoy))])  # 1=binder

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
    rows = []
    for ax, (label, col, better) in zip(axes.flat, PANELS):
        sign = 1.0 if better == "higher" else -1.0  # orient so higher = more "binder"
        oriented = sign * np.concatenate([known[col].values, decoy[col].values])
        thr, sens, fpr = youden_threshold(y_true, oriented)
        y_pred = (oriented >= thr).astype(int)
        # rows/cols ordered [decoy(0), binder(1)]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        spec = 1.0 - fpr
        acc = (tp + tn) / cm.sum()
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        native_thr = sign * thr  # back to native units for display

        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["pred decoy", "pred binder"], fontsize=9)
        ax.set_yticks([0, 1], ["true decoy", "true binder"], fontsize=9)
        vmax = cm.max()
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                    color="white" if cm[i, j] > vmax * 0.5 else "#111827",
                )
        rule = "≤" if better == "lower" else "≥"
        ax.set_title(
            f"{label}   (cut: {col} {rule} {native_thr:.2f})\n"
            f"acc {acc:.2f}  sens {sens:.2f}  spec {spec:.2f}  prec {prec:.2f}",
            fontsize=10,
            pad=6,
        )

    fig.suptitle(
        "6TD3 / CDK12-DDB1 — confusion matrices at Youden-optimal cut "
        f"(n={len(known)} binders / {len(decoy)} decoys, Exp. 002 data)",
        fontsize=12.5,
        y=0.99,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = Path(__file__).parent / "confusion_matrices.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")

    # numeric dump
    for ax, (label, col, better) in zip(axes.flat, PANELS):
        sign = 1.0 if better == "higher" else -1.0
        oriented = sign * np.concatenate([known[col].values, decoy[col].values])
        thr, sens, fpr = youden_threshold(y_true, oriented)
        y_pred = (oriented >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        acc = (tp + tn) / cm.sum()
        rows.append(
            {
                "panel": label,
                "column": col,
                "rule": ("<=" if better == "lower" else ">="),
                "threshold_native": float(sign * thr),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "accuracy": float(acc),
                "sensitivity": float(sens),
                "specificity": float(1 - fpr),
                "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
            }
        )
    out_csv = Path(__file__).parent / "confusion_matrices.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(
        f"{'panel':<13}{'thr':>8}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}{'acc':>7}{'sens':>7}{'spec':>7}"
    )
    for r in rows:
        print(
            f"{r['panel']:<13}{r['threshold_native']:>8.2f}{r['TP']:>5}{r['FP']:>5}"
            f"{r['FN']:>5}{r['TN']:>5}{r['accuracy']:>7.2f}{r['sensitivity']:>7.2f}{r['specificity']:>7.2f}"
        )
    print(f"\nSaved: {out_png}\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
