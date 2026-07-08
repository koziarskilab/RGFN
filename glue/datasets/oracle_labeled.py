"""The accumulating oracle-labelled dataset ``D`` of the active-learning loop.

In ``[bengio2021gflownet]`` Alg. 1 the dataset grows each round as
``D_i = D̂_i ∪ D_{i-1}`` — the proxy is refit on the *full history*, not just the
latest batch. This class holds that history: ``(SMILES, oracle_label)`` pairs,
de-duplicated by canonical SMILES (keeping the most recent label), seeded from a
CSV ``D_0``, and queryable for the final Top-K deliverable.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gin
from rdkit import Chem


def _canonical(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


@gin.configurable()
class OracleLabeledDataset:
    """Accumulating store of ``(SMILES, oracle_label)`` pairs, keyed by canonical SMILES."""

    def __init__(
        self,
        seed_csv: Optional[str] = None,
        smiles_column: str = "smiles",
        label_column: str = "label",
    ):
        """
        Args:
            seed_csv: optional path to ``D_0``; rows are read on construction.
            smiles_column / label_column: column names in the seed CSV.
        """
        self._data: Dict[str, float] = {}  # canonical smiles -> label
        self.smiles_column = smiles_column
        self.label_column = label_column
        if seed_csv:
            self.load_seed(seed_csv, smiles_column, label_column)

    def __len__(self) -> int:
        return len(self._data)

    def load_seed(self, csv_path: str, smiles_column: str, label_column: str) -> int:
        """Load a seed dataset ``D_0`` from CSV. Returns the number of rows added."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Seed CSV not found: {path}")
        added = 0
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if smiles_column not in row or label_column not in row:
                    raise KeyError(
                        f"Seed CSV {path} must have columns '{smiles_column}' and "
                        f"'{label_column}'; found {list(row.keys())}."
                    )
                try:
                    label = float(row[label_column])
                except (TypeError, ValueError):
                    continue
                if self.add_one(row[smiles_column], label):
                    added += 1
        return added

    def add_one(self, smiles: str, label: float) -> bool:
        """Add/overwrite one labelled molecule. Returns True if it was stored."""
        canon = _canonical(smiles)
        if canon is None:
            return False
        self._data[canon] = float(label)
        return True

    def add(self, smiles: List[str], labels: List[float]) -> int:
        """Accumulate a labelled batch (the ``D_i = D̂_i ∪ D_{i-1}`` update).

        Skips entries with unparseable SMILES or non-finite labels. Returns the
        number of (new-or-updated) molecules actually stored.
        """
        stored = 0
        for smi, label in zip(smiles, labels):
            if label is None or label != label:  # None or NaN
                continue
            if self.add_one(smi, label):
                stored += 1
        return stored

    def to_lists(self) -> Tuple[List[str], List[float]]:
        """Return parallel ``(smiles, labels)`` lists for proxy fitting."""
        items = list(self._data.items())
        return [s for s, _ in items], [y for _, y in items]

    def top_k(self, k: int) -> List[Tuple[str, float]]:
        """Return the Top-K ``(smiles, label)`` by label (the loop's deliverable).

        Both oracles label molecules on a *more-negative = better* scale
        (``GlueOracle.higher_is_better = False``: the 6TD3 ``ddb1_dvina``
        differential and the sEH Vina binding energy alike). The best molecules
        are therefore the *smallest* labels, so we sort ascending.
        """
        return sorted(self._data.items(), key=lambda kv: kv[1])[:k]

    def save_csv(self, path: str) -> None:
        """Dump the full dataset to CSV (provenance / resume)."""
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([self.smiles_column, self.label_column])
            for smi, label in self._data.items():
                writer.writerow([smi, label])
