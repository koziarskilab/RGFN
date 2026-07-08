"""
Shared helpers for the MW-controlled 6TD3 (CDK12-DDB1) discrimination analysis.

Entry 006 (experiments/ablations/sixway/) ranked the six oracle
signals — {Vina, CNN affinity} x {Tier 1, Tier 2, Tier 2 - Tier 1} — and found
Vina ΔT2−T1 the best known-vs-decoy discriminator. The open airtightness gap it
flagged: known glues and decoys may differ in *size*, and bigger ligands dock
better simply by making more contacts, so a reviewer could attribute the Tier 2
shift to molecular weight rather than to genuine DDB1 cooperativity.

This module is the shared spine of the MW follow-up. It:
  - loads the Entry 002 docking CSVs (status='ok' only),
  - adds a molecular-weight column (RDKit average MW from SMILES) and the two
    Tier 2 - Tier 1 differentials,
  - exposes the same six-panel definition the rest of the analysis uses,
  - provides the AUROC helper and a reproducible greedy MW-matching routine.

No new docking — these are the same poses scored in Entry 002 (Balam job 69271).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.Chem.rdmolfiles import MolFromSmiles
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[3]  # RGFN-Fork root
DOCK6TD3 = ROOT / "experiments" / "oracle_validation" / "docking_6td3"
KNOWN_CSV = DOCK6TD3 / "known_results.csv"
DECOY_CSV = DOCK6TD3 / "decoy_cdk_results.csv"

# Shared palette (matches sixway/ (was full_comparison) where they overlap).
DECOY_COLOR = "#94A3B8"  # slate — fake glues (correct warhead, random arm)
BINDER_COLOR = "#2563EB"  # blue — known CDK12-DDB1 glues
MW_ACCENT = "#16A34A"  # green — MW / size annotations

# Default 1:1 nearest-neighbor matching caliper (Da) and bootstrap settings.
DEFAULT_CALIPER = 15.0
DEFAULT_SEED = 0

# (label, column, better-direction) for the six oracle signals.
# Vina: lower (more negative) = stronger binding. CNN affinity: higher = better.
PANELS = [
    ("Vina Tier 1", "vina_t1", "lower"),
    ("Vina Tier 2", "vina_t2", "lower"),
    ("Vina ΔT2−T1", "dvina", "lower"),
    ("CNN Tier 1", "cnnaff_t1", "higher"),
    ("CNN Tier 2", "cnnaff_t2", "higher"),
    ("CNN ΔT2−T1", "dcnnaff", "higher"),
]


def mol_weight(smiles):
    """RDKit average molecular weight (Da) for a SMILES, NaN if unparseable."""
    mol = MolFromSmiles(smiles)
    return float(Descriptors.MolWt(mol)) if mol is not None else float("nan")


def load(path):
    """Read a docking-result CSV, keep status='ok', add mw + differentials."""
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["mw"] = [mol_weight(s) for s in df["smiles"]]
    # Differentials (Tier 2 - Tier 1) recomputed so the plotted quantity is explicit.
    df["dvina"] = df["vina_t2"] - df["vina_t1"]
    df["dcnnaff"] = df["cnnaff_t2"] - df["cnnaff_t1"]
    df = df.dropna(subset=["mw"]).reset_index(drop=True)
    return df


def auroc(binder_vals, decoy_vals, better):
    """
    AUROC = P(a random binder scores better than a random decoy).

    Orients the metric so "better-for-binder" is larger, then uses the
    Mann-Whitney U identity AUROC = U / (n_binder * n_decoy).
    """
    sign = 1.0 if better == "higher" else -1.0
    b, d = sign * np.asarray(binder_vals), sign * np.asarray(decoy_vals)
    U, _ = mannwhitneyu(b, d, alternative="greater")
    return float(U / (len(b) * len(d)))


def greedy_mw_match(known, decoy, caliper=DEFAULT_CALIPER, rng=None):
    """
    Greedy 1:1 nearest-neighbor matching of known glues to decoys on MW.

    Each known molecule (visited in a randomized order so the matching is not an
    artifact of input order) claims the unused decoy with the closest MW, as long
    as |ΔMW| <= caliper. Returns (known_matched, decoy_matched) sub-DataFrames of
    equal length with near-identical MW distributions; unmatched rows are dropped.

    Knowns heavier than every decoy (the decoy MW ceiling is ~462 Da) simply find
    no partner within the caliper and are excluded — matching is therefore
    restricted to the MW region the two sets actually share.
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)
    decoy_mw = decoy["mw"].to_numpy()
    available = np.ones(len(decoy), dtype=bool)
    order = rng.permutation(len(known))
    k_idx, d_idx = [], []
    for i in order:
        target = known["mw"].iat[i]
        diffs = np.abs(decoy_mw - target)
        diffs[~available] = np.inf
        j = int(np.argmin(diffs))
        if diffs[j] <= caliper:
            available[j] = False
            k_idx.append(i)
            d_idx.append(j)
    return known.iloc[k_idx].copy(), decoy.iloc[d_idx].copy()
