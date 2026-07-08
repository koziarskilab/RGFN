"""GPU-docked two-tier differential oracle (QuickVina2-GPU pose generation).

This is the **GPU-accelerated sibling** of :class:`~glue.oracles.docking_6td3_oracle.Docking6TD3Oracle`.
It computes the *same* neosubstrate differential — ``Vina(Tier2) - Vina(Tier1)``,
*more negative = stronger recruited-partner cooperativity = better glue*, so
``higher_is_better = False`` — but swaps **only** the expensive step.

Why this exists (see ``Logs/006`` six-way ablation, ``Logs/008`` pose-selection
ablation, ``Logs/012`` cost breakdown):
    The gnina oracle spends ~99% of its docking time in a **CPU** conformational
    *search* (gnina/AutoDock-Vina MCMC, exhaustiveness 16, 9 modes). Everything
    else — CNN pose ranking and the two ``--score_only`` passes — is cheap. So we
    keep the validated *structure* exactly and move *only the search* to the GPU:

        006 (gnina)                      this oracle (QV2-GPU)
        ------------------------------   ------------------------------------
        gnina SEARCH -> 9 poses (CPU)    QuickVina2-GPU SEARCH -> 9 poses (GPU)
        gnina CNN -> pick max CNNscore   gnina --score_only CNN -> pick max CNNscore
        gnina --score_only Tier2 (impl.) gnina --score_only Tier2
        gnina --score_only Tier1         gnina --score_only Tier1
        dvina = Vina(T2) - Vina(T1)      dvina = Vina(T2) - Vina(T1)

    The CNN selection and BOTH tier scores stay on gnina ``--score_only`` (the
    same scoring function as 006) so the differential is a 1:1 comparison to the
    validated AUROC 0.946. QuickVina2-GPU is used **purely to generate the
    candidate poses** — because (a) it has no ``--score_only`` mode (the option is
    commented out of its source), and (b) subtracting a QV2 score from a gnina
    score would not cancel the two engines' calibration offset.

Faithfulness to the structure-preserving swap:
    - Poses come from ONE QV2 search per molecule with ``num_modes`` output modes
      (default 9) — mirroring 006's "one search -> 9 modes", not N independent
      conformer docks.
    - The QV2 search box is derived from the crystal ligand bounding box +
      ``autobox_add`` (default 4 A), reproducing gnina's
      ``--autobox_ligand crystal --autobox_add 4``.
    - Tier 1 is **score-only on the frozen selected pose** (no re-docking): the
      differential is the same pose scored in two contexts (with/without the
      recruited partner), exactly as in 006.

Generic by construction: the base class takes arbitrary Tier1/Tier2 receptors and
a box (or a crystal ligand to autobox from), so a future system (a different glue,
a different ternary complex) is a one-line subclass — see
:class:`Docking6TD3GpuOracle` for the 6TD3/CR8 instance.

Environment (Balam-only, deferred to first :meth:`score` so imports stay
laptop-safe): a GPU, the QuickVina2-GPU-2.1 build (``quickvina_dir`` symlink),
its boost libs on ``LD_LIBRARY_PATH``, the gnina launcher (``$GNINA``), Open Babel
(pdbqt->sdf), and the prepared ``.pdbqt`` receptors.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import gin

from glue.oracles.base import GlueOracle
from glue.oracles.step_timing import OracleStepTimer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_QV_DIR = _REPO_ROOT / "quickvina_dir"  # -> $SCRATCH/vina_gpu/Vina-GPU-2.1
# 6TD3 receptors live alongside the validated batch pipeline (git-ignored .pdbqt).
_DOCK_DIR = _REPO_ROOT / "experiments" / "oracle_validation" / "docking_6td3"


def _box_from_crystal(pdb_path: Path, autobox_add: float) -> tuple:
    """Center + size of the crystal-ligand bounding box, padded ``autobox_add`` A
    on each side. Reproduces gnina's ``--autobox_ligand <pdb> --autobox_add``."""
    xs, ys, zs = [], [], []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
    if not xs:
        raise ValueError(f"No atoms parsed from crystal ligand {pdb_path}")

    def axis(v):
        lo, hi = min(v), max(v)
        return (lo + hi) / 2.0, (hi - lo) + 2 * autobox_add

    cx, sx = axis(xs)
    cy, sy = axis(ys)
    cz, sz = axis(zs)
    return [cx, cy, cz], [sx, sy, sz]


def _parse_score_only(stdout: str) -> tuple:
    """Streamed gnina ``--score_only`` output -> (affinities, cnnscores, cnnaffs),
    one entry per input ligand, in order. Mirrors dock_cluster._score_only."""
    affs, cnnsc, cnnaff = [], [], []
    for line in stdout.splitlines():
        if line.startswith("Affinity:"):
            affs.append(float(line.split()[1]))
        elif line.startswith("CNNscore:"):
            cnnsc.append(float(line.split()[1]))
        elif line.startswith("CNNaffinity:"):
            cnnaff.append(float(line.split()[1]))
    return affs, cnnsc, cnnaff


@gin.configurable()
class GpuDifferentialDockingOracle(GlueOracle):
    """Generic two-tier differential oracle with GPU (QuickVina2-GPU) pose search.

    Returns ``Vina(Tier2) - Vina(Tier1)`` per molecule (nan on failure); both tier
    scores and the pose selection come from gnina ``--score_only`` on the
    QV2-generated poses. Reusable for any Tier1/Tier2 system: supply the two
    receptor ``.pdbqt`` paths and either an explicit box (``center``/``size``) or a
    ``crystal_path`` to autobox from.
    """

    name = "gpu_differential_docking"
    # Vina binding-energy differential: MORE NEGATIVE = better glue -> lower is better.
    higher_is_better = False

    def __init__(
        self,
        tier2_path: str,
        tier1_path: str,
        center: Optional[Sequence[float]] = None,
        size: Optional[Sequence[float]] = None,
        crystal_path: Optional[str] = None,
        autobox_add: float = 4.0,
        qv_dir: Optional[str] = None,
        gnina: Optional[str] = None,
        vina_mode: str = "QuickVina2",
        num_modes: int = 9,
        exhaustiveness: int = 8000,
        docking_batch_size: int = 25,
        n_gpu: int = 1,
        n_cpu: Optional[int] = None,
        work_dir: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Args:
            tier2_path / tier1_path: receptor ``.pdbqt`` for the complex (Tier 2)
                and the anchoring protein alone (Tier 1).
            center / size: QV2 search box (3-vectors). If omitted, derived from
                ``crystal_path`` + ``autobox_add``.
            crystal_path: crystal-ligand PDB used to autobox the search region
                (mirrors gnina ``--autobox_ligand``). Required if center/size are None.
            autobox_add: padding (A) added on each side of the crystal bbox.
            qv_dir: QuickVina2-GPU build dir (default: repo ``quickvina_dir`` symlink).
            gnina: gnina launcher (default ``$GNINA`` or the project path).
            vina_mode: Vina-GPU implementation ("QuickVina2").
            num_modes: poses requested from ONE QV2 search per molecule (default 9,
                matching the 006 gnina num_modes).
            exhaustiveness: QV2 ``thread`` count (search effort; min 1000).
            docking_batch_size: SMILES per QV2 invocation.
            n_gpu / n_cpu: GPUs to shard across / CPUs for Meeko conformer prep.
            work_dir: scratch dir for intermediates (a temp dir if None).
            name: override the oracle ``name`` (dataset provenance / logs).
        """
        if center is None or size is None:
            if crystal_path is None:
                raise ValueError("Provide either center+size or crystal_path to autobox from.")
            center, size = _box_from_crystal(Path(crystal_path), autobox_add)
        self.tier2_path = Path(tier2_path)
        self.tier1_path = Path(tier1_path)
        self.crystal_path = Path(crystal_path) if crystal_path else None
        self.center = list(center)
        self.size = list(size)
        self.qv_dir = str(qv_dir) if qv_dir else str(_DEFAULT_QV_DIR)
        self.gnina = gnina or os.environ.get("GNINA", "/scratch/markymoo/gnina/run_gnina.sh")
        self.vina_mode = vina_mode
        self.num_modes = num_modes
        self.exhaustiveness = exhaustiveness
        self.docking_batch_size = docking_batch_size
        self.n_gpu = n_gpu
        self.n_cpu = n_cpu
        self.work_dir = work_dir
        if name:
            self.name = name
        self._proxy = None  # lazy QV2 proxy (defers meeko/openbabel import to Balam)
        self._timer = OracleStepTimer(None)  # opt-in substep timing (no-op until enabled)

    def enable_step_timing(self, csv_path) -> None:
        """Record per-substep wall-clock (qv2_dock / pose_convert / tier2_score /
        pose_select / tier1_score) to ``csv_path``; matches Docking6TD3Oracle so the
        GPU vs CPU cost is directly comparable. No-op if never called."""
        self._timer = OracleStepTimer(csv_path)

    # ----------------------------------------------------------------- public API
    def score(self, smiles: List[str]) -> List[float]:
        """Return the DDB1-style differential per molecule (nan on failure)."""
        return [d["dvina"] for d in self.score_detailed(smiles)]

    def score_detailed(self, smiles: List[str]) -> List[Dict]:
        """Per-molecule dict with the full breakdown (used by score() and by the
        validation harness): ``vina_t2``, ``vina_t1``, ``dvina``, ``cnnsc_t2``
        (selected pose's CNNscore), ``qv2_t2`` (QV2's own best score, diagnostic),
        ``n_poses``, ``status``. Order matches ``smiles``."""
        self._check_inputs()
        n = len(smiles)
        out = [
            {
                "vina_t2": float("nan"),
                "vina_t1": float("nan"),
                "dvina": float("nan"),
                "cnnsc_t2": float("nan"),
                "qv2_t2": float("nan"),
                "n_poses": 0,
                "status": "no_pose",
            }
            for _ in range(n)
        ]
        proxy = self._build_proxy()
        work = (
            Path(self.work_dir) if self.work_dir else Path(tempfile.mkdtemp(prefix="dock6td3gpu_"))
        )
        work.mkdir(parents=True, exist_ok=True)
        self._timer.new_round()

        # 1. QV2-GPU search: one search per molecule, `num_modes` poses each.
        with self._timer.step("qv2_dock", n):
            try:
                result = proxy.docking_module_gpu(list(smiles))
            except Exception as e:  # whole-batch GPU failure -> all nan
                for d in out:
                    d["status"] = f"qv2_fail:{type(e).__name__}"
                return out
        if not isinstance(result, tuple):
            return out
        qv_scores, qv_poses = result

        # 2. Convert each molecule's multi-MODEL pdbqt -> per-pose SDF entries and
        #    accumulate ALL poses into one Tier-2 SDF (one batched gnina call).
        index: List[int] = []  # entry position -> molecule index
        blocks: List[str] = []  # entry position -> single-mol SDF text
        with self._timer.step("pose_convert", n):
            for i in range(n):
                pose = qv_poses[i] if i < len(qv_poses) else None
                if not pose:
                    continue
                if i < len(qv_scores) and qv_scores[i] is not None:
                    out[i]["qv2_t2"] = float(qv_scores[i])
                sdf_blocks = self._pdbqt_to_sdf_blocks(pose, work, i)
                out[i]["n_poses"] = len(sdf_blocks)
                for blk in sdf_blocks:
                    index.append(i)
                    blocks.append(blk)
        if not blocks:
            return out
        tier2_sdf = work / "tier2_all.sdf"
        tier2_sdf.write_text("".join(blocks))

        # 3. gnina --score_only against Tier 2 -> per-pose (Affinity, CNNscore).
        with self._timer.step("tier2_score", len(index)):
            affs, cnnsc, _ = self._score_only(self.tier2_path, tier2_sdf)
        if not (len(affs) == len(cnnsc) == len(index)):
            # Count mismatch would misalign poses -> refuse to guess; nan the batch.
            for d in out:
                if d["status"] == "no_pose":
                    d["status"] = "tier2_count_mismatch"
            return out

        # 4. Select the most native-like pose (max CNNscore) per molecule.
        best: Dict[int, tuple] = {}  # mol idx -> (entry_pos, affinity_t2, cnnscore)
        with self._timer.step("pose_select", n):
            for pos, i in enumerate(index):
                if i not in best or cnnsc[pos] > best[i][2]:
                    best[i] = (pos, affs[pos], cnnsc[pos])

        # 5. Tier 1 score-only on the frozen selected poses (one batched call).
        order = sorted(best)
        sel_sdf = work / "tier1_selected.sdf"
        sel_sdf.write_text("".join(blocks[best[i][0]] for i in order))
        with self._timer.step("tier1_score", len(order)):
            t1affs, _, _ = self._score_only(self.tier1_path, sel_sdf)
        for k, i in enumerate(order):
            out[i]["vina_t2"] = float(best[i][1])
            out[i]["cnnsc_t2"] = float(best[i][2])
            out[i]["status"] = "ok"
            if k < len(t1affs):
                out[i]["vina_t1"] = float(t1affs[k])
                out[i]["dvina"] = float(best[i][1] - t1affs[k])
            else:
                out[i]["status"] = "tier1_missing"
        return out

    # ------------------------------------------------------------------ internals
    def _check_inputs(self) -> None:
        missing = [str(p) for p in (self.tier2_path, self.tier1_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "GpuDifferentialDockingOracle missing receptor inputs (expected on Balam): "
                + ", ".join(missing)
            )

    def _build_proxy(self):
        """Lazily build the upstream QV2-GPU proxy and force multi-mode output.

        Composes ``DockingMoleculeProxy`` (Meeko prep + QuickVina2-GPU) rather than
        re-implementing docking. The upstream proxy hardcodes ``num_modes=1``; we
        set it on the underlying ``VinaDocking`` so ONE search returns ``num_modes``
        poses (read at call time in its config writer)."""
        if self._proxy is None:
            from rgfn.gfns.reaction_gfn.proxies.docking_proxy.docking_proxy import (
                DockingMoleculeProxy,
            )

            proxy = DockingMoleculeProxy(
                qv_dir=self.qv_dir,
                vina_mode=self.vina_mode,
                gnina=False,  # we run our own gnina --score_only with CNN selection
                receptor_path=str(self.tier2_path),
                center=self.center,
                size=self.size,
                n_conformers=1,  # ONE search per molecule...
                exhaustiveness=self.exhaustiveness,
                docking_batch_size=self.docking_batch_size,
                n_gpu=self.n_gpu,
                n_cpu=self.n_cpu,
                print_msgs=False,
            )
            proxy.docking_module_gpu.additional_vina_args["num_modes"] = self.num_modes
            self._proxy = proxy
        return self._proxy

    def _run_gnina(self, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.gnina, *args], capture_output=True, text=True, env=dict(os.environ)
        )

    def _score_only(self, receptor: Path, sdf: Path) -> tuple:
        r = self._run_gnina(["-r", str(receptor), "-l", str(sdf), "--score_only"])
        return _parse_score_only(r.stdout)

    @staticmethod
    def _pdbqt_to_sdf_blocks(pose_pdbqt: str, work: Path, idx: int) -> List[str]:
        """Multi-MODEL docked pdbqt -> list of single-molecule SDF blocks (one per
        pose), via Open Babel. Each block ends with the ``$$$$`` record separator."""
        pq = work / f"m{idx}.pdbqt"
        sdf = work / f"m{idx}.sdf"
        pq.write_text(pose_pdbqt)
        subprocess.run(
            ["obabel", "-ipdbqt", str(pq), "-osdf", "-O", str(sdf)],
            capture_output=True,
            text=True,
        )
        if not sdf.exists() or sdf.stat().st_size == 0:
            return []
        text = sdf.read_text()
        parts = [p for p in text.split("$$$$\n") if p.strip()]
        return [p + "$$$$\n" for p in parts]


@gin.configurable()
class Docking6TD3GpuOracle(GpuDifferentialDockingOracle):
    """6TD3 / CR8 instance of the GPU differential oracle (CDK12+DDB1 vs CDK12).

    Same DDB1 neosubstrate differential and receptors as
    :class:`~glue.oracles.docking_6td3_oracle.Docking6TD3Oracle`, but with the
    Tier-2 conformational search run on the GPU (QuickVina2-GPU) instead of CPU
    gnina. Drop-in: ``higher_is_better = False`` and the same ``dvina`` metric.
    """

    name = "docking_6td3_gpu"

    def __init__(
        self,
        tier2_path: Optional[str] = None,
        tier1_path: Optional[str] = None,
        crystal_path: Optional[str] = None,
        autobox_add: float = 4.0,
        num_modes: int = 9,
        exhaustiveness: int = 8000,
        qv_dir: Optional[str] = None,
        gnina: Optional[str] = None,
        n_gpu: int = 1,
        n_cpu: Optional[int] = None,
        docking_batch_size: int = 25,
        work_dir: Optional[str] = None,
    ):
        super().__init__(
            tier2_path=tier2_path or str(_DOCK_DIR / "6TD3_tier2.pdbqt"),
            tier1_path=tier1_path or str(_DOCK_DIR / "6TD3_tier1.pdbqt"),
            crystal_path=crystal_path or str(_DOCK_DIR / "crystal_RC8.pdb"),
            autobox_add=autobox_add,
            num_modes=num_modes,
            exhaustiveness=exhaustiveness,
            qv_dir=qv_dir,
            gnina=gnina,
            n_gpu=n_gpu,
            n_cpu=n_cpu,
            docking_batch_size=docking_batch_size,
            work_dir=work_dir,
            name="docking_6td3_gpu",
        )
