"""Retroactive dataset metrics for a set of generated molecules.

These are *pure functions* over SMILES (+ optional oracle labels): no gin, no GFN
trajectories, no model state. The active-learning loop calls them live on each
round's query batch, but the same functions run on any saved CSV after the fact
(``experiments/active_learning/6td3/analyze_suggestions.py``) — so a finished run
can be re-analysed without re-docking anything.

Metric provenance (kept faithful to what we and the source papers report):

  Per-molecule descriptors mirror the exact set RGFN's own mode report emits
  (``rgfn/trainer/metrics/reaction_metrics.py`` ``TanimotoSimilarityModes._modes_to_df``):
  ExactMolWt, QED, cLogP (MolLogP), H-bond donors/acceptors, heavy atoms,
  rotatable bonds, ligand efficiency (reward / heavy atoms). We add TPSA, ring
  count, and the Lipinski rule-of-five pass flag (medchem-sanity, Objective 5 in
  ``docs/RESEARCH_CONTEXT.md``), and the synthesis length (# reactions) which is
  the RGFN-specific size proxy ``[koziarski2024rgfn]`` §5 ties to docking-score
  inflation.

  Set-level metrics:
    - uniqueness                      ``[koziarski2024rgfn]`` Table 1
    - number of modes (Tanimoto<0.7)  ``[bengio2021gflownet]`` Fig. 7 / RGFN;
                                      Morgan r=3, 2048 bits, threshold 0.7 — the
                                      exact recipe in ``TanimotoSimilarityModes``
    - internal diversity (1 - mean    standard GFlowNet diversity (mean pairwise
      pairwise Tanimoto)              Tanimoto over the same fingerprint)
    - # Murcko scaffolds              scaffold diversity (RGFN, implicit)
    - novelty vs the seed D_0         retrospective-enrichment guard (Objective 5)
    - oracle-score distribution       reward distribution ``[bengio2021gflownet]``
"""

from statistics import mean, median, pstdev
from typing import Dict, List, Optional, Sequence

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

# Suppress RDKit's per-molecule parse warnings; we already handle None mols.
RDLogger.DisableLog("rdApp.*")

# Fingerprint recipe shared by num_modes / internal_diversity. Matches
# rgfn/trainer/metrics/reaction_metrics.py:TanimotoSimilarityModes (radius 3, 2048).
_FP_RADIUS = 3
_FP_BITS = 2048
_MODE_SIMILARITY_THRESHOLD = 0.7


def _mol(smiles: str):
    return Chem.MolFromSmiles(smiles) if smiles else None


def _fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=_FP_RADIUS, nBits=_FP_BITS)


def descriptors_for_smiles(smiles: str, label: Optional[float] = None) -> Optional[Dict]:
    """Per-molecule descriptors, or ``None`` if the SMILES does not parse.

    ``label`` (the oracle score) is used only for ligand efficiency; pass ``None``
    to omit it. ``lipinski_pass`` is the classic rule-of-five (MW<=500, cLogP<=5,
    HBD<=5, HBA<=10) — a medchem-sanity flag, not a hard filter.
    """
    mol = _mol(smiles)
    if mol is None:
        return None
    mw = Descriptors.ExactMolWt(mol)
    clogp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    heavy = mol.GetNumHeavyAtoms()
    d = {
        "mol_weight": mw,
        "qed": qed(mol),
        "clogp": clogp,
        "h_donors": hbd,
        "h_acceptors": hba,
        "heavy_atoms": heavy,
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "lipinski_pass": int(mw <= 500 and clogp <= 5 and hbd <= 5 and hba <= 10),
    }
    if label is not None and label == label:  # not None / not NaN
        d["ligand_efficiency"] = float(label) / heavy if heavy else float("nan")
    return d


def murcko_scaffold(smiles: str) -> Optional[str]:
    """Canonical Murcko scaffold SMILES, or ``None`` on parse failure."""
    mol = _mol(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def _valid_fps(smiles_list: Sequence[str]):
    fps = []
    for smi in smiles_list:
        mol = _mol(smi)
        if mol is not None:
            fps.append(_fingerprint(mol))
    return fps


def count_modes(
    smiles_list: Sequence[str], similarity_threshold: float = _MODE_SIMILARITY_THRESHOLD
) -> int:
    """Number of Tanimoto modes: greedily cluster so no two representatives are
    more similar than ``similarity_threshold``. Same algorithm/recipe as upstream
    ``TanimotoSimilarityModes`` (Morgan r=3, 2048 bits, threshold 0.7)."""
    modes = []
    for fp in _valid_fps(smiles_list):
        if all(DataStructs.TanimotoSimilarity(fp, m) <= similarity_threshold for m in modes):
            modes.append(fp)
    return len(modes)


def internal_diversity(smiles_list: Sequence[str]) -> float:
    """1 - mean pairwise Tanimoto similarity over the set (higher = more diverse).

    Returns NaN for fewer than two parseable molecules (no pairs)."""
    fps = _valid_fps(smiles_list)
    n = len(fps)
    if n < 2:
        return float("nan")
    total, pairs = 0.0, 0
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        total += sum(sims)
        pairs += len(sims)
    return 1.0 - (total / pairs) if pairs else float("nan")


def count_scaffolds(smiles_list: Sequence[str]) -> int:
    """Number of distinct Murcko scaffolds in the set."""
    return len({s for s in (murcko_scaffold(x) for x in smiles_list) if s is not None})


def novelty(smiles_list: Sequence[str], reference_smiles: Sequence[str]) -> float:
    """Fraction of (canonicalised) molecules not present in ``reference_smiles``.

    Used to measure how much each round's batch departs from the seed D_0 — a
    retrospective-enrichment / not-just-memorising guard. NaN if the input set is
    empty after canonicalisation."""
    ref = {c for c in (_canon(s) for s in reference_smiles) if c is not None}
    canons = [c for c in (_canon(s) for s in smiles_list) if c is not None]
    if not canons:
        return float("nan")
    novel = sum(1 for c in canons if c not in ref)
    return novel / len(canons)


def _canon(smiles: str) -> Optional[str]:
    mol = _mol(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _summ(values: List[float], name: str) -> Dict[str, float]:
    """mean/std/median/min/max of a numeric list (NaN-safe: drops non-finite)."""
    xs = [v for v in values if v is not None and v == v]
    if not xs:
        return {f"{name}_{k}": float("nan") for k in ("mean", "std", "median", "min", "max")}
    return {
        f"{name}_mean": mean(xs),
        f"{name}_std": pstdev(xs) if len(xs) > 1 else 0.0,
        f"{name}_median": median(xs),
        f"{name}_min": min(xs),
        f"{name}_max": max(xs),
    }


def batch_metrics(
    smiles_list: Sequence[str],
    labels: Optional[Sequence[float]] = None,
    reference_smiles: Optional[Sequence[str]] = None,
    oracle_threshold: Optional[float] = None,
    oracle_higher_is_better: bool = False,
) -> Dict[str, float]:
    """Aggregate set + distribution metrics for one batch of suggested molecules.

    Args:
        smiles_list: the suggested SMILES (one round's query batch).
        labels: parallel oracle scores (optional); drives the reward distribution
            and ligand efficiency.
        reference_smiles: the seed D_0 (or any prior set) to measure novelty against.
        oracle_threshold: if given, also report the fraction of labels that beat it
            (``<=`` when lower-is-better, else ``>=``) — e.g. the -2.0 glue cutoff.
        oracle_higher_is_better: orientation of the oracle for the threshold test.

    Returns a flat ``{metric: value}`` dict suitable for CSV rows and wandb logging.
    """
    labels = list(labels) if labels is not None else [None] * len(smiles_list)
    descs = [descriptors_for_smiles(s, l) for s, l in zip(smiles_list, labels)]
    valid = [d for d in descs if d is not None]
    n_valid = len(valid)

    out: Dict[str, float] = {
        "n_suggested": len(smiles_list),
        "n_valid": n_valid,
        "n_unique": len({c for c in (_canon(s) for s in smiles_list) if c is not None}),
        "internal_diversity": internal_diversity(smiles_list),
        "num_modes": count_modes(smiles_list),
        "num_scaffolds": count_scaffolds(smiles_list),
        "frac_lipinski_pass": (
            sum(d["lipinski_pass"] for d in valid) / n_valid if n_valid else float("nan")
        ),
    }
    if reference_smiles is not None:
        out["novelty_vs_seed"] = novelty(smiles_list, reference_smiles)

    for field in (
        "mol_weight",
        "qed",
        "clogp",
        "h_donors",
        "h_acceptors",
        "heavy_atoms",
        "rotatable_bonds",
        "tpsa",
        "num_rings",
    ):
        out.update(_summ([d[field] for d in valid], field))

    valid_labels = [l for l in labels if l is not None and l == l]
    if valid_labels:
        out.update(_summ(valid_labels, "oracle"))
        if oracle_threshold is not None:
            if oracle_higher_is_better:
                hits = sum(1 for l in valid_labels if l >= oracle_threshold)
            else:
                hits = sum(1 for l in valid_labels if l <= oracle_threshold)
            out[f"oracle_frac_beating_{oracle_threshold}"] = hits / len(valid_labels)
    return out
