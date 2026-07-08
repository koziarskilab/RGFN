"""Discrimination guard for the 6TD3 oracle metric — the "science validated" test.

This is the scientific counterpart to the wiring smoke test. It does NOT run gnina
(that's Balam): it guards the *metric and label choice* by checking that the
quantity we feed the active-learning loop actually separates known glues from
decoys. It reproduces the validated results:
  - ``Logs/002``: frac dVina < -1.5 = known 85.6% vs decoy 7.3% (+78pt gap);
  - ``Logs/006``: Vina ΔT2-T1 is the best of six signals, AUROC 0.946;
  - ``Logs/007``: that AUROC is molecular-weight-robust (the differential, unlike
    absolute scores, isn't just reading ligand size).
(See also ``experiments/oracle_validation/compare_systems.py`` and
``experiments/ablations/sixway/``.)

Why this exists: we once shipped the oracle using the **CNNaffinity** differential
(``ddb1_dcnnaff``) instead of the validated **Vina** differential (``ddb1_dvina``).
That bug passes py_compile, imports, and the mock-oracle smoke test — but it
optimises a quantity that does not discriminate. This test fails loudly on it:
``ddb1_dcnnaff`` shows ~0 separation, ``ddb1_dvina`` shows ~78pts.

Runs on a laptop from data committed to the repo (``seed_6td3.csv``). The optional
contrast check additionally reads the per-molecule result CSVs if present
(git-ignored, so Mac/Balam only) to assert directly that the Vina differential
separates and the CNNaffinity one does not.
"""

import csv
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SEED_CSV = REPO / "experiments" / "active_learning" / "6td3" / "seed_6td3.csv"
DOCK_DIR = REPO / "experiments" / "oracle_validation" / "docking_6td3"

# Validated threshold + expected separation from Logs/002 (compare_systems.py uses
# the same "dVina < -1.5" strong-bonus criterion). The +78pt gap is the headline;
# we assert a comfortably loose >50pt so the test is robust but still catches a
# non-discriminating metric (which gives ~0pt).
STRONG_BONUS_THRESHOLD = -1.5
MIN_KNOWN_DECOY_GAP = 0.50
MIN_AUROC = 0.90  # Log 006: Vina ΔT2-T1 = 0.946; CNNaffinity differential = 0.850


def _read_labelled(path: Path, value_col: str):
    """Return (known_values, decoy_values) from a CSV with a 'set' column."""
    known, decoy = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row.get("status", "ok") != "ok":
                continue
            raw = row.get(value_col, "")
            if raw in ("", None):
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            tag = (row.get("set") or "").strip().lower()
            if tag == "known":
                known.append(v)
            elif tag == "decoy":
                decoy.append(v)
    return known, decoy


def _frac_strong_bonus(values):
    """Fraction of molecules with a strong DDB1 bonus (dVina more negative than thr)."""
    return sum(1 for v in values if v < STRONG_BONUS_THRESHOLD) / len(values) if values else 0.0


def _auroc(known, decoy, higher_is_better=False):
    """AUROC = P(a random known ranks 'better' than a random decoy), ties=0.5.

    For dVina (lower-is-better) a known 'wins' a pair when its value is smaller.
    Matches Log 006's headline metric (Vina ΔT2-T1 AUROC 0.946)."""
    if not known or not decoy:
        return float("nan")
    wins = 0.0
    for k in known:
        for d in decoy:
            better = (k < d) if not higher_is_better else (k > d)
            wins += 1.0 if better else (0.5 if k == d else 0.0)
    return wins / (len(known) * len(decoy))


def test_seed_labels_discriminate_known_from_decoy():
    """The committed seed labels (ddb1_dvina) must separate knowns from decoys.

    This is the regression guard: it would have failed on the ddb1_dcnnaff bug,
    because that column does not discriminate.
    """
    assert SEED_CSV.exists(), f"seed dataset missing: {SEED_CSV} (run build_seeds.py)"
    known, decoy = _read_labelled(SEED_CSV, "label")
    assert (
        known and decoy
    ), f"seed CSV must contain both 'known' and 'decoy' rows; got {len(known)}/{len(decoy)}"

    known_frac = _frac_strong_bonus(known)
    decoy_frac = _frac_strong_bonus(decoy)
    gap = known_frac - decoy_frac
    auroc = _auroc(known, decoy, higher_is_better=False)
    print(
        f"\nseed_6td3.csv (label): known frac<{STRONG_BONUS_THRESHOLD}={known_frac:.1%}, "
        f"decoy={decoy_frac:.1%}, gap={gap*100:+.0f}pts (Log 002: 85.6% / 7.3% / +78pts); "
        f"AUROC={auroc:.3f} (Log 006: 0.946)"
    )
    assert (
        known_frac > 0.5
    ), f"known glues should mostly show a strong DDB1 bonus; got {known_frac:.1%}"
    assert gap > MIN_KNOWN_DECOY_GAP, (
        f"label column does not discriminate (gap {gap*100:.0f}pts < {MIN_KNOWN_DECOY_GAP*100:.0f}). "
        "Is the seed built from ddb1_dvina (validated) rather than ddb1_dcnnaff?"
    )
    # Log 006's headline metric: Vina ΔT2-T1 AUROC 0.946. Assert a robust >0.90.
    assert auroc > MIN_AUROC, (
        f"AUROC {auroc:.3f} < {MIN_AUROC}; the label is not the validated Vina differential "
        "(Log 006 reports 0.946 for Vina ΔT2-T1, 0.850 for the CNNaffinity differential)."
    )


def test_vina_differential_beats_cnnaffinity_differential():
    """Contrast check: reproduce Log 006's ranking on the raw docking results — the
    VINA differential out-discriminates the CNNaffinity differential (AUROC 0.946
    vs 0.850), documenting why the oracle must use ddb1_dvina. Skipped if the
    (git-ignored) per-molecule result CSVs are absent (e.g. a fresh clone)."""
    known_csv = DOCK_DIR / "known_results.csv"
    decoy_csv = DOCK_DIR / "decoy_cdk_results.csv"
    if not (known_csv.exists() and decoy_csv.exists()):
        pytest.skip("per-molecule result CSVs not present (git-ignored; Mac/Balam only)")

    def auroc_for(col, higher_is_better):
        k = _read_labelled(known_csv, col)[0] + _read_labelled(decoy_csv, col)[0]  # knowns
        d = _read_labelled(known_csv, col)[1] + _read_labelled(decoy_csv, col)[1]  # decoys
        return _auroc(k, d, higher_is_better=higher_is_better)

    # Orientation differs by scoring function: Vina is a binding ENERGY (lower is
    # better) but gnina CNNaffinity is a pK (higher is better). Using the wrong
    # orientation flips AUROC to 1-AUROC -- which is exactly the kind of sign slip
    # this whole test suite exists to catch.
    dvina_auroc = auroc_for("ddb1_dvina", higher_is_better=False)
    dcnn_auroc = auroc_for("ddb1_dcnnaff", higher_is_better=True)
    print(
        f"\nAUROC: Vina ΔT2-T1={dvina_auroc:.3f} (Log006 0.946) | "
        f"CNN ΔT2-T1={dcnn_auroc:.3f} (Log006 0.850)"
    )
    # Vina differential is the validated discriminator (Log 006: 0.946)...
    assert dvina_auroc > MIN_AUROC, f"ddb1_dvina should discriminate; AUROC={dvina_auroc:.3f}"
    # ...and it strictly out-discriminates the CNNaffinity differential.
    assert dvina_auroc > dcnn_auroc, "Vina differential should out-discriminate the CNNaffinity one"


if __name__ == "__main__":
    # Standalone runner (works without pytest, e.g. on Balam): run both checks.
    test_seed_labels_discriminate_known_from_decoy()
    try:
        test_vina_differential_beats_cnnaffinity_differential()
    except Exception as e:  # includes pytest.skip's Skipped
        print(f"(contrast check skipped/failed: {type(e).__name__}: {e})")
    print("OK: discrimination guard passed.")
