"""6TD3 / CR8 two-tier docking oracle ``O`` for the active-learning loop.

Scores a molecule by the **neosubstrate differential** (the DDB1 cooperativity
bonus) on the 6TD3 system — our validated discrimination metric, and the
best-ranked of six candidate signals (see ``docs/RESEARCH_CONTEXT.md``,
``Logs/002`` validation, ``Logs/006`` six-way ablation, ``Logs/007`` MW control).
For one molecule:

    1. embed + MMFF-optimise a 3D conformer (RDKit);
    2. dock into Tier 2 (CDK12 + DDB1) with gnina, autoboxed on the crystal CR8;
    3. keep the most native-like pose (highest CNNscore — pose selection only);
    4. ``--score_only`` that same pose against Tier 1 (CDK12 alone);
    5. differential = ``Vina(Tier2) - Vina(Tier1)`` (= ``ddb1_dvina``) — *more
       negative* means the arm gains binding once the recruited partner (DDB1) is
       present. Vina is a binding energy, so **lower is better** ->
       ``higher_is_better = False``.

    Why this exact signal (Vina Tier2-Tier1), with evidence:
        - Log 002: known median -2.20 vs decoy -0.60; 85.6% vs 7.3% below -1.5
          (+78pt gap).
        - Log 006 ranked all SIX candidate signals (Vina/CNN x Tier1/Tier2/diff)
          on the same poses: **Vina ΔT2-T1 wins, AUROC 0.946** (Cohen's d 2.38);
          its Youden-optimal cut (-1.58) independently lands on the -1.5 above.
          The CNNaffinity differential ``ddb1_dcnnaff`` is far weaker (AUROC 0.850
          here, ~0 separation at the -1.5 threshold) — so we do NOT use it;
          CNNscore is used only to pick the pose (step 3).
        - Log 007 (MW control): after matching glues/decoys on molecular weight,
          Vina ΔT2-T1 stays the top discriminator (0.946 -> 0.866, -0.08), while
          absolute Vina Tier 1 collapses below chance (0.69 -> 0.38). The
          differential is the right signal precisely because it cancels ligand
          size and isolates the DDB1-recruitment contribution.

Provenance / faithfulness:
    This mirrors the **validated** batch pipeline in
    ``experiments/oracle_validation/docking_6td3/dock_cluster.py`` — identical gnina
    flags (``--autobox_ligand`` crystal, ``--autobox_add 4``, exhaustiveness,
    num_modes), identical "best pose = max CNNscore", identical Tier-1
    ``--score_only`` rescoring, and the same ``ddb1_dvina`` differential. The
    difference is orchestration only: this runs one in-process batch (the AL loop
    queries ~hundreds of molecules per round) instead of ``dock_cluster.py``'s
    multi-GPU sharded job. **The two implementations duplicate the docking logic;
    they must be reconciled and cross-validated against each other on Balam**
    (see ``docs/REFACTOR_LOG.md``). ``dock_cluster.py`` is the source of truth.

Environment: gnina, a GPU, and the prepared receptors
(``6TD3_tier{1,2}.pdbqt``, ``crystal_RC8.pdb``) are **Balam-only** — the
``.pdbqt`` files are git-ignored and absent on a laptop. Import is therefore
side-effect-free; all docking I/O is deferred to :meth:`score`, which validates
the toolchain and inputs at call time.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import gin
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from glue.oracles.base import GlueOracle
from glue.oracles.step_timing import OracleStepTimer

RDLogger.DisableLog("rdApp.*")

# Default inputs live alongside dock_cluster.py (git-ignored .pdbqt; present on Balam).
_DOCK_DIR = (
    Path(__file__).resolve().parents[2] / "experiments" / "oracle_validation" / "docking_6td3"
)


def _largest_frag(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return max(frags, key=lambda f: f.GetNumHeavyAtoms()) if len(frags) > 1 else mol


@gin.configurable()
class Docking6TD3Oracle(GlueOracle):
    """Expensive 6TD3 docking oracle returning the DDB1 neosubstrate differential."""

    name = "docking_6td3"
    # ddb1_dvina is a Vina binding-energy differential: MORE NEGATIVE = stronger
    # DDB1 cooperativity = better glue. So lower is better.
    higher_is_better = False

    def __init__(
        self,
        tier2_path: Optional[str] = None,
        tier1_path: Optional[str] = None,
        crystal_path: Optional[str] = None,
        gnina: Optional[str] = None,
        exhaustiveness: int = 16,
        num_modes: int = 9,
        autobox_add: float = 4.0,
        n_cpu: int = 8,
        seed: int = 42,
        work_dir: Optional[str] = None,
    ):
        """
        Args mirror ``dock_cluster.py`` defaults so the two stay comparable.

        Args:
            tier2_path / tier1_path / crystal_path: receptor and reference paths;
                default to ``experiments/oracle_validation/docking_6td3/`` (Balam).
            gnina: gnina launcher; defaults to ``$GNINA`` or dock_cluster.py's path.
            exhaustiveness / num_modes / autobox_add / seed: gnina docking params
                (identical defaults to dock_cluster.py).
            n_cpu: CPUs passed to gnina per call.
            work_dir: scratch dir for intermediate SDFs; a temp dir if None.
        """
        self.tier2_path = Path(tier2_path) if tier2_path else _DOCK_DIR / "6TD3_tier2.pdbqt"
        self.tier1_path = Path(tier1_path) if tier1_path else _DOCK_DIR / "6TD3_tier1.pdbqt"
        self.crystal_path = Path(crystal_path) if crystal_path else _DOCK_DIR / "crystal_RC8.pdb"
        self.gnina = gnina or os.environ.get("GNINA", "/scratch/markymoo/gnina/run_gnina.sh")
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        self.autobox_add = autobox_add
        self.n_cpu = n_cpu
        self.seed = seed
        self.work_dir = work_dir
        # Sub-step timing is opt-in (disabled = silent no-op); the AL loop turns it
        # on per run via enable_step_timing(). See glue/oracles/step_timing.py.
        self._timer = OracleStepTimer(None)

    def enable_step_timing(self, csv_path) -> None:
        """Record per-substep wall-clock (embed / tier2_dock / pose_select /
        tier1_rescore) to ``csv_path``. Called optionally by ``ActiveLearningLoop``;
        the oracle works identically (just untimed) if it is never called."""
        self._timer = OracleStepTimer(csv_path)

    # ----------------------------------------------------------------- public API
    def score(self, smiles: List[str]) -> List[float]:
        """Dock a batch and return the DDB1 differential per molecule (nan on failure)."""
        self._check_inputs()
        results = {i: float("nan") for i in range(len(smiles))}
        work = Path(self.work_dir) if self.work_dir else Path(tempfile.mkdtemp(prefix="dock6td3_"))
        work.mkdir(parents=True, exist_ok=True)
        self._timer.new_round()

        # phase 1: embed (idx -> molblock); idx is the SMILES position in the batch.
        blocks = {}
        with self._timer.step("embed", len(smiles)):
            for i, smi in enumerate(smiles):
                blk = self._embed(i, smi)
                if blk is not None:
                    blocks[i] = blk
        if not blocks:
            return [results[i] for i in range(len(smiles))]

        # write one SDF for the whole batch (titles = idx), matching dock_cluster.py
        batch_sdf = work / "batch.sdf"
        with open(batch_sdf, "w") as fh:
            fh.write("".join(blocks[i] + "$$$$\n" for i in sorted(blocks)))

        # phase 2: dock into Tier 2 (the exhaustive conformational search -- expected
        # to dominate; this is the step a GPU docker would replace, see Logs/012).
        docked = work / "batch_docked.sdf"
        with self._timer.step("tier2_dock", len(blocks)):
            self._run_gnina(
                [
                    "-r",
                    str(self.tier2_path),
                    "-l",
                    str(batch_sdf),
                    "--autobox_ligand",
                    str(self.crystal_path),
                    "--autobox_add",
                    str(self.autobox_add),
                    "--exhaustiveness",
                    str(self.exhaustiveness),
                    "--num_modes",
                    str(self.num_modes),
                    "--cpu",
                    str(self.n_cpu),
                    "--seed",
                    str(self.seed),
                    "-o",
                    str(docked),
                ]
            )

        # phase 3: parse poses + pick the most native-like (max CNNscore), exactly as
        # dock_cluster.py; the *scored differential* is Vina, not CNNaffinity.
        order, best_mols, vina_t2 = [], [], {}
        with self._timer.step("pose_select", len(blocks)):
            poses = self._poses(docked)  # idx(str) -> [pose dicts]
            for i in sorted(blocks):
                ps = poses.get(str(i))
                if not ps:
                    continue
                best = max(ps, key=lambda p: p["cnnsc"])  # most native-like pose
                order.append(i)
                best_mols.append(best["mol"])
                vina_t2[i] = best["vina"]  # Tier-2 Vina affinity of that pose

        if best_mols:
            best_sdf = work / "batch_best.sdf"
            writer = Chem.SDWriter(str(best_sdf))
            for m in best_mols:
                writer.write(m)
            writer.close()
            # phase 4: Tier-1 rescore of the chosen pose (--score_only, no search).
            with self._timer.step("tier1_rescore", len(best_mols)):
                t1 = self._score_only(self.tier1_path, best_sdf)  # [(Affinity, CNNaffinity)]
            for k, i in enumerate(order):
                if k < len(t1):
                    vina_t1 = t1[k][0]  # Tier-1 Vina affinity of the same pose
                    # ddb1_dvina = Vina(Tier2) - Vina(Tier1); more negative = better
                    # glue (validated discrimination metric, Log 002).
                    results[i] = float(vina_t2[i] - vina_t1)

        return [results[i] for i in range(len(smiles))]

    # ------------------------------------------------------------- gnina plumbing
    def _check_inputs(self) -> None:
        missing = [
            str(p) for p in (self.tier2_path, self.tier1_path, self.crystal_path) if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Docking6TD3Oracle is missing receptor/crystal inputs (expected on Balam): "
                + ", ".join(missing)
            )

    def _embed(self, idx: int, smi: str) -> Optional[str]:
        """3D embed + MMFF optimise; mirrors dock_cluster.embed (title = idx)."""
        try:
            mol = Chem.MolFromSmiles(smi) if smi is not None else None
            if mol is None:
                return None
            mol = Chem.AddHs(_largest_frag(mol))
            if AllChem.EmbedMolecule(mol, randomSeed=self.seed) != 0:
                AllChem.EmbedMolecule(mol, randomSeed=self.seed, useRandomCoords=True)
            AllChem.MMFFOptimizeMolecule(mol)
            mol.SetProp("_Name", str(idx))
            return Chem.MolToMolBlock(mol)
        except Exception:
            return None

    def _run_gnina(self, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.gnina, *args], capture_output=True, text=True, check=True, env=dict(os.environ)
        )

    @staticmethod
    def _poses(sdf: Path) -> dict:
        """idx(str) -> list of pose dicts; mirrors dock_cluster._poses."""
        groups: dict = {}
        for m in Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False):
            if m is None:
                continue
            idx = m.GetProp("_Name") if m.HasProp("_Name") else None
            g = lambda k: float(m.GetProp(k)) if m.HasProp(k) else float("nan")
            groups.setdefault(idx, []).append(
                {
                    "mol": m,
                    "vina": g("minimizedAffinity"),
                    "cnnsc": g("CNNscore"),
                    "cnnaff": g("CNNaffinity"),
                }
            )
        return groups

    def _score_only(self, receptor: Path, sdf: Path) -> List[tuple]:
        """Ordered [(Affinity, CNNaffinity)]; mirrors dock_cluster._score_only_stream."""
        r = self._run_gnina(["-r", str(receptor), "-l", str(sdf), "--score_only"])
        affs, caffs = [], []
        for line in r.stdout.splitlines():
            if line.startswith("Affinity:"):
                affs.append(float(line.split()[1]))
            elif line.startswith("CNNaffinity:"):
                caffs.append(float(line.split()[1]))
        return list(zip(affs, caffs))
