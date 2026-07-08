"""Datasets — loaders for input data and generators for synthetic datasets.

Two responsibilities:
    1. Load curated inputs (known glues, decoys, building blocks) for benchmarks
       and oracle validation.
    2. Generate synthetic datasets from trained RGFN runs (sample molecules,
       score them, emit a dataset artifact).

Conventions:
    - Raw / curated inputs live under `data/` and `data/validation-molecules/`.
    - Generated synthetic datasets are written under `data/synthetic/` (gitignored).
    - Keep file formats simple (csv / smiles / parquet) and documented.
    - Generated *candidate* datasets (a generator's proposed molecules + scores +
      optional synthesis routes) use the ONE standard format in `candidates.py`
      (`CandidateDataset`), so our pipeline and every baseline generator under
      `validation/generators/` emit the same thing and the harness reads them
      uniformly. Spec: `docs/CANDIDATE_DATASET_FORMAT.md`.

Import new dataset modules below so `glue.registry` registers any gin-configurable
loaders or generators.
"""

from glue.datasets.candidates import (  # noqa: F401
    CANDIDATE_SCHEMA_VERSION,
    CandidateDataset,
    from_smiles_table,
    read_candidate_dataset,
    validate_candidate_dataset,
)
from glue.datasets.oracle_labeled import OracleLabeledDataset  # noqa: F401
from glue.datasets.suggestion_log import SuggestionLog  # noqa: F401

# from glue.datasets.synthetic import SyntheticDatasetGenerator  # noqa: F401

__all__ = [
    "OracleLabeledDataset",
    "SuggestionLog",
    "CandidateDataset",
    "from_smiles_table",
    "read_candidate_dataset",
    "validate_candidate_dataset",
    "CANDIDATE_SCHEMA_VERSION",
]
