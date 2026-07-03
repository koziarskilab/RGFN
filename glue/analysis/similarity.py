"""Fingerprint + Tanimoto helpers shared by the hub and molecule selectors.

The diversity-aware strategies (``MostModesHubSelector``, ``TopKRewardDiverseSelector``,
``ScaffoldDiverseKSelector``) all need the *same* similarity recipe the rest of the
project uses so their numbers line up with ``glue.metrics.dataset_metrics`` and
upstream ``TanimotoSimilarityModes``: **Morgan fingerprints, radius 3, 2048 bits,
Tanimoto threshold 0.7** (``[koziarski2024rgfn]`` / ``[bengio2021gflownet]``). This
module is the one place that recipe lives for the analysis package, so a change here
propagates everywhere and can't drift.

Pure functions over SMILES — no GFN, no gin, no torch.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, TypeVar

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# Matches glue.metrics.dataset_metrics + rgfn TanimotoSimilarityModes (r=3, 2048 bits).
FP_RADIUS = 3
FP_BITS = 2048
MODE_SIMILARITY_THRESHOLD = 0.7

T = TypeVar("T")


def fingerprint(smiles: str):
    """Morgan bit-vector fingerprint for a SMILES, or ``None`` if it does not parse."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_BITS)


def tanimoto(fp_a, fp_b) -> float:
    """Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def max_similarity_to_set(fp, fps: Sequence) -> float:
    """Max Tanimoto of ``fp`` against a set of fingerprints (0.0 if the set is empty)."""
    if not fps:
        return 0.0
    return max(DataStructs.BulkTanimotoSimilarity(fp, list(fps)))


def greedy_diverse_select(
    items: Sequence[T],
    k: int,
    get_smiles: Callable[[T], str],
    similarity_threshold: float = MODE_SIMILARITY_THRESHOLD,
    presorted: bool = True,
) -> List[T]:
    """Greedy MaxMin-style diversity filter.

    Walk ``items`` (assumed already ordered by preference — e.g. reward-descending —
    when ``presorted``) and keep each one whose Tanimoto to *every* already-kept item
    is ``< similarity_threshold``, stopping once ``k`` are kept. This is the standard
    "top-k but spread out" selection: it never reorders by similarity, it just skips
    near-duplicates of things already chosen, so the highest-reward representative of
    each cluster survives.

    Args:
        items: candidate objects (any type).
        k: number to keep.
        get_smiles: how to read a SMILES string off an item.
        similarity_threshold: keep an item only if it is *less* similar than this to
            all kept items (0.7 == the project's mode threshold).
        presorted: kept for signature clarity; the caller is responsible for ordering.

    Returns:
        Up to ``k`` items, in the order encountered.
    """
    kept: List[T] = []
    kept_fps: List = []
    for item in items:
        if len(kept) >= k:
            break
        fp = fingerprint(get_smiles(item))
        if fp is None:
            continue
        if max_similarity_to_set(fp, kept_fps) < similarity_threshold:
            kept.append(item)
            kept_fps.append(fp)
    return kept


def count_modes(
    smiles_list: Sequence[str], similarity_threshold: float = MODE_SIMILARITY_THRESHOLD
) -> int:
    """Number of Tanimoto modes (greedy clusters no two representatives more similar
    than ``similarity_threshold``). Same algorithm as
    ``glue.metrics.dataset_metrics.count_modes`` — duplicated here only so the analysis
    package has no import-time dependency surprises; the recipe is identical."""
    modes: List = []
    for smi in smiles_list:
        fp = fingerprint(smi)
        if fp is None:
            continue
        if max_similarity_to_set(fp, modes) <= similarity_threshold:
            modes.append(fp)
    return len(modes)


def canonical(smiles: str) -> Optional[str]:
    """RDKit canonical SMILES, or ``None`` if unparseable."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None
