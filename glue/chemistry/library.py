"""``ChemLibrary`` — the single source of truth for a standard fragment + reaction set.

See ``docs/CHEM_LIBRARY_FORMAT.md`` for the design. A *library* is the two inputs a
reaction-GFlowNet builds molecules from — a building-block set and a
reaction-template set — plus an optional cost annotation (per-block price,
per-reaction yield). ``ChemLibrary`` normalizes the on-disk formats already in the
repo (upstream ``chemistry.xlsx``, SCENT's ``data/small/``, and our canonical
``data/libraries/<name>/``) into one in-memory object, and exports it back into each
generator's native format so every entrant can consume the *same* chemistry.

Kept free of RGFN plumbing on purpose: turning a ``ChemLibrary`` into the action
space RGFN's ``ReactionEnv`` expects is the job of
``glue.chemistry.reaction_data_factory.GlueReactionDataFactory``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

FRAGMENTS_FILE = "fragments.csv"
REACTIONS_FILE = "reactions.csv"
MANIFEST_FILE = "manifest.json"

DEFAULT_COST = 1.0
DEFAULT_YIELD = 0.5


def _canonical_smiles(smiles: str) -> Optional[str]:
    """RDKit canonical SMILES, or None if unparseable (RDKit imported lazily)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _canonical_reaction(smarts: str) -> str:
    """Stable key for a reaction template.

    Reaction SMARTS canonicalization is not well-defined in RDKit, and the reaction
    string is used verbatim by upstream ``Reaction`` (and matches across SCENT's
    ``templates.txt`` / ``templates_yields.csv``), so we key on the whitespace-
    normalized string: strip, and collapse the spaces some sources put around ``>>``.
    """
    return " >> ".join(part.strip() for part in smarts.split(">>"))


@dataclass(frozen=True)
class FragmentSpec:
    """One building block. ``cost`` is None when the library carries no prices."""

    smiles: str
    cost: Optional[float] = None
    group: Optional[str] = None
    catalog_id: Optional[str] = None


@dataclass(frozen=True)
class ReactionSpec:
    """One reaction template (``reactants >> products``). ``yield_`` may be None."""

    smarts: str
    yield_: Optional[float] = None
    family: Optional[str] = None
    comments: Optional[str] = None


@dataclass
class ChemLibrary:
    """A standard, swappable fragment + reaction set with optional cost annotation."""

    fragments: List[FragmentSpec] = field(default_factory=list)
    reactions: List[ReactionSpec] = field(default_factory=list)
    manifest: Dict[str, object] = field(default_factory=dict)

    # --- loaders: many formats in, one object out --------------------------------

    @classmethod
    def from_canonical_dir(cls, path: str | Path) -> "ChemLibrary":
        """Read ``data/libraries/<name>/`` (fragments.csv + reactions.csv + manifest.json)."""
        path = Path(path)
        manifest: Dict[str, object] = {}
        if (path / MANIFEST_FILE).exists():
            with open(path / MANIFEST_FILE) as fh:
                manifest = json.load(fh)

        fragments: List[FragmentSpec] = []
        with open(path / FRAGMENTS_FILE, newline="") as fh:
            for row in csv.DictReader(fh):
                cost = row.get("cost", "")
                fragments.append(
                    FragmentSpec(
                        smiles=row["smiles"].strip(),
                        cost=float(cost) if cost not in ("", None) else None,
                        group=(row.get("group") or None),
                        catalog_id=(row.get("catalog_id") or None),
                    )
                )

        reactions: List[ReactionSpec] = []
        with open(path / REACTIONS_FILE, newline="") as fh:
            for row in csv.DictReader(fh):
                y = row.get("yield", "")
                reactions.append(
                    ReactionSpec(
                        smarts=row["reaction"].strip(),
                        yield_=float(y) if y not in ("", None) else None,
                        family=(row.get("family") or None),
                        comments=(row.get("comments") or None),
                    )
                )
        return cls(fragments=fragments, reactions=reactions, manifest=manifest)

    @classmethod
    def from_chemistry_xlsx(cls, path: str | Path, docking: bool = False) -> "ChemLibrary":
        """Read the upstream workbook (``Reactions_*`` / ``Fragments_*`` sheets). No costs."""
        import pandas as pd

        sheet_r = "Reactions_Docking" if docking else "Reactions_NoDocking"
        sheet_f = "Fragments_Docking" if docking else "Fragments_NoDocking"
        rdf = pd.read_excel(path, sheet_name=sheet_r)
        fdf = pd.read_excel(path, sheet_name=sheet_f)

        reactions = [
            ReactionSpec(
                smarts=_canonical_reaction(str(r["Reaction"])),
                family=(str(r["Family"]) if not pd.isna(r.get("Family")) else None),
                comments=(str(r["Comments"]) if not pd.isna(r.get("Comments")) else None),
            )
            for _, r in rdf.iterrows()
            if isinstance(r["Reaction"], str)
        ]
        fragments = [
            FragmentSpec(
                smiles=str(f["Fragment"]).strip(),
                group=(str(f["Group"]) if not pd.isna(f.get("Group")) else None),
            )
            for _, f in fdf.iterrows()
            if isinstance(f["Fragment"], str)
        ]
        return cls(fragments=fragments, reactions=reactions)

    @classmethod
    def from_scent_small(cls, path: str | Path) -> "ChemLibrary":
        """Read SCENT's ``data/small/`` — chemistry **with** costs + yields.

        Files (see ``external/scent/data/small/``):
          fragments.txt                one building-block SMILES per line
          fragment_to_real_cost.json   {fragment_SMILES: price}
          templates.txt                one reaction SMARTS per line
          templates_yields.csv         Family,Reaction,Comments,yield (keyed by Reaction)

        Fragment SMILES are canonicalized; the cost is joined from the JSON by the
        *raw* key (the JSON is keyed by the same raw strings as fragments.txt).
        """
        path = Path(path)
        with open(path / "fragment_to_real_cost.json") as fh:
            cost_by_raw: Dict[str, float] = json.load(fh)

        fragments: List[FragmentSpec] = []
        seen = set()
        with open(path / "fragments.txt") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                canon = _canonical_smiles(raw)
                if canon is None or canon in seen:
                    continue
                seen.add(canon)
                fragments.append(FragmentSpec(smiles=canon, cost=cost_by_raw.get(raw)))

        yield_by_reaction: Dict[str, float] = {}
        family_by_reaction: Dict[str, str] = {}
        comments_by_reaction: Dict[str, str] = {}
        with open(path / "templates_yields.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                key = _canonical_reaction(row["Reaction"])
                if row.get("yield") not in ("", None):
                    yield_by_reaction[key] = float(row["yield"])
                if row.get("Family"):
                    family_by_reaction[key] = row["Family"]
                if row.get("Comments"):
                    comments_by_reaction[key] = row["Comments"]

        reactions: List[ReactionSpec] = []
        seen_r = set()
        with open(path / "templates.txt") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                key = _canonical_reaction(raw)
                if key in seen_r:
                    continue
                seen_r.add(key)
                reactions.append(
                    ReactionSpec(
                        smarts=key,
                        yield_=yield_by_reaction.get(key),
                        family=family_by_reaction.get(key),
                        comments=comments_by_reaction.get(key),
                    )
                )
        return cls(fragments=fragments, reactions=reactions)

    # --- exporters ----------------------------------------------------------------

    def to_canonical_dir(self, path: str | Path) -> None:
        """Write this library in the canonical format (+ a provenance manifest)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / FRAGMENTS_FILE, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["smiles", "group", "cost", "catalog_id"])
            for f in self.fragments:
                w.writerow(
                    [
                        f.smiles,
                        f.group or "",
                        "" if f.cost is None else f.cost,
                        f.catalog_id or "",
                    ]
                )
        with open(path / REACTIONS_FILE, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["reaction", "family", "comments", "yield"])
            for r in self.reactions:
                w.writerow(
                    [
                        r.smarts,
                        r.family or "",
                        r.comments or "",
                        "" if r.yield_ is None else r.yield_,
                    ]
                )

        n_priced = sum(1 for f in self.fragments if f.cost is not None)
        n_yield = sum(1 for r in self.reactions if r.yield_ is not None)
        manifest = dict(self.manifest)
        manifest.setdefault("default_cost", DEFAULT_COST)
        manifest.setdefault("default_yield", DEFAULT_YIELD)
        manifest.update(
            {
                "n_fragments": len(self.fragments),
                "n_reactions": len(self.reactions),
                "n_fragments_priced": n_priced,
                "n_reactions_with_yield": n_yield,
                "fragments_sha256": _sha256(path / FRAGMENTS_FILE),
                "reactions_sha256": _sha256(path / REACTIONS_FILE),
            }
        )
        with open(path / MANIFEST_FILE, "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)

    def to_chemistry_xlsx(self, path: str | Path) -> None:
        """Write an upstream-format workbook so RGFN's stock ``ReactionDataFactory``
        (pristine loader) can read this library. Writes both the ``*_NoDocking`` and
        ``*_Docking`` sheet pairs with identical content (the factory picks by flag)."""
        import pandas as pd

        rxn_df = pd.DataFrame(
            {
                "Family": [r.family or "" for r in self.reactions],
                "Reaction": [r.smarts for r in self.reactions],
                "Comments": [r.comments or "" for r in self.reactions],
            }
        )
        frag_df = pd.DataFrame(
            {
                "Group": [f.group or "" for f in self.fragments],
                "Fragment": [f.smiles for f in self.fragments],
            }
        )
        with pd.ExcelWriter(path) as xl:
            rxn_df.to_excel(xl, sheet_name="Reactions_NoDocking", index=False)
            frag_df.to_excel(xl, sheet_name="Fragments_NoDocking", index=False)
            rxn_df.to_excel(xl, sheet_name="Reactions_Docking", index=False)
            frag_df.to_excel(xl, sheet_name="Fragments_Docking", index=False)

    def to_rxnflow_inputs(self, outdir: str | Path) -> "tuple[Path, Path]":
        """Write RxnFlow's raw inputs: ``building_block.smi`` (``SMILES<TAB>ID``) and
        ``template.txt`` (one SMARTS per line). These feed RxnFlow's env-baking script
        (``external/RxnFlow/data/scripts/b_create_env.py``) which computes the
        ``bb_*.npy`` caches. Returns ``(blocks_path, templates_path)``."""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        blocks = outdir / "building_block.smi"
        templates = outdir / "template.txt"
        with open(blocks, "w") as fh:
            for i, f in enumerate(self.fragments):
                fh.write(f"{f.smiles}\tBB{i:06d}\n")
        with open(templates, "w") as fh:
            for r in self.reactions:
                fh.write(f"{r.smarts}\n")
        return blocks, templates

    def to_fragment_csv(self, path: str | Path) -> None:
        """Write a one-column ``SMILES`` CSV for ``ReactionDataFactory.fragment_path``."""
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["SMILES"])
            for f in self.fragments:
                w.writerow([f.smiles])

    # --- cost tables: shared by native (SCENT) and retroactive (harness) pricing --

    def cost_of(self, smiles: str) -> float:
        """Canonicalized block-price lookup; falls back to ``default_cost``."""
        default = float(self.manifest.get("default_cost", DEFAULT_COST))
        canon = _canonical_smiles(smiles)
        return self.fragment_to_cost().get(canon, default)

    def yield_of(self, smarts: str) -> float:
        """Canonicalized reaction-yield lookup; falls back to ``default_yield``."""
        default = float(self.manifest.get("default_yield", DEFAULT_YIELD))
        return self.reaction_to_yield().get(_canonical_reaction(smarts), default)

    def fragment_to_cost(self) -> Dict[str, float]:
        """``{canonical_smiles: cost}`` for all priced fragments."""
        return {f.smiles: f.cost for f in self.fragments if f.cost is not None}

    def reaction_to_yield(self) -> Dict[str, float]:
        """``{canonical_smarts: yield}`` for all reactions with a yield."""
        return {r.smarts: r.yield_ for r in self.reactions if r.yield_ is not None}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
