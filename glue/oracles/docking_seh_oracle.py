"""sEH GPU-docking oracle ``O`` for the active-learning loop.

Scores a molecule by its **AutoDock-Vina binding affinity against sEH**, computed
with the GPU-accelerated QuickVina2-GPU-2.1 docking engine — *exactly the docking
that upstream RGFN performs* in ``configs/rgfn_seh_docking.gin``. The difference
from upstream is only the *role*: there, docking is the in-loop reward and RGFN
queries it millions of times; here it is the expensive oracle ``O`` of
``[bengio2021gflownet]`` Alg. 1, queried only on the per-round query batch, while
the cheap learned MPNN proxy ``M`` (``glue.proxies.learned_proxy``) carries the
inner reward. This is the active-learning restructuring the project is building
toward (see ``docs/RESEARCH_CONTEXT.md``): a regular MPNN proxy + a real
GPU-docking oracle, instead of docking inside the inner loop.

Faithfulness / no duplication:
    This oracle does **not** re-implement docking. It *composes* upstream
    ``DockingMoleculeProxy`` and calls its ``dock_batch_qv2gpu`` directly on
    SMILES, so the conformer prep (Meeko), the QuickVina2-GPU invocation, the box
    centre/size for sEH, exhaustiveness/threads, and the multi-conformer ranking
    are byte-for-byte the upstream code path. Only the ReactionState <-> SMILES
    glue and the oracle sign convention live here. (Contrast the 6TD3 oracle,
    whose docstring flags that it *duplicates* gnina logic and must be reconciled
    — this oracle avoids that debt by reusing the upstream proxy outright.)

Sign convention:
    Vina returns a binding **energy** in kcal/mol; ``dock_batch_qv2gpu`` returns
    it raw (negative for a real pose). More negative = stronger binding = better,
    so ``higher_is_better = False`` — matching the 6TD3 oracle's convention and
    what ``LearnedGlueProxy.higher_is_better`` must be set to. Per the
    :class:`~glue.oracles.base.GlueOracle` contract we return ``float('nan')`` for
    molecules that fail to dock (upstream's ``dock_batch_qv2gpu`` yields ``None``
    for those), so the loop drops them when refitting ``M`` rather than feeding the
    proxy a fake ``failed_score``.

Environment: a GPU, the QuickVina2-GPU-2.1 build (``quickvina_dir`` symlink ->
``$SCRATCH/vina_gpu/Vina-GPU-2.1``), its boost runtime libs on
``LD_LIBRARY_PATH``, and ``data/targets/sEH.pdbqt`` — see the
``gpu-docking-oracle-setup`` project note. Import is side-effect-free: the heavy
``DockingMoleculeProxy`` (which imports meeko/openbabel) is built lazily on the
first :meth:`score` call, so this module imports and gin-validates on a laptop.
"""

from pathlib import Path
from typing import List, Optional, Union

import gin

from glue.oracles.base import GlueOracle

# The repo-root ``quickvina_dir`` symlink resolves to the QuickVina2-GPU install
# on $SCRATCH (git-ignored; Balam-only). Same default DockingMoleculeProxy expects.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_QV_DIR = _REPO_ROOT / "quickvina_dir"


def _chunks(lst, n):
    """Yield successive ``n``-sized chunks (mirrors docking_proxy._chunks)."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@gin.configurable()
class DockingSEHOracle(GlueOracle):
    """Expensive sEH GPU-docking oracle returning the Vina binding affinity.

    Thin SMILES-facing wrapper over upstream ``DockingMoleculeProxy`` (QuickVina2-
    GPU-2.1). The active-learning loop calls :meth:`score` on each round's query
    batch; the returned Vina energies grow ``D`` and the MPNN proxy is refit on
    them.
    """

    name = "docking_seh"
    # Vina binding energy (kcal/mol): more negative = stronger binding = better.
    higher_is_better = False

    def __init__(
        self,
        qv_dir: Optional[Union[Path, str]] = None,
        receptor_name: str = "sEH",
        vina_mode: str = "QuickVina2",
        exhaustiveness: int = 8000,
        docking_batch_size: int = 25,
        n_conformers: int = 1,
        gnina: bool = False,
        conformer_attempts: int = 20,
        docking_attempts: int = 10,
        n_gpu: int = 1,
        n_cpu: Optional[int] = None,
        print_msgs: bool = False,
    ):
        """
        Args mirror upstream ``DockingMoleculeProxy`` so the two dock identically.

        Args:
            qv_dir: directory holding the QuickVina2-GPU build; defaults to the
                repo ``quickvina_dir`` symlink (-> ``$SCRATCH/vina_gpu/Vina-GPU-2.1``).
            receptor_name: a key in ``DockingMoleculeProxy``'s receptor tables
                (``sEH`` here; box centre/size are predefined there).
            vina_mode: Vina-GPU implementation (``QuickVina2`` default).
            exhaustiveness: QuickVina ``thread`` count (upstream default 8000; min
                1000). Higher = more thorough but slower.
            docking_batch_size: SMILES per QuickVina2-GPU invocation.
            n_conformers: conformers generated/docked per molecule; the best score
                is kept (upstream behaviour). 1 = fastest.
            gnina: rescore poses with gnina (upstream option; off by default).
            conformer_attempts / docking_attempts: RDKit/Vina retry budgets.
            n_gpu: number of GPUs to shard the batch across.
            n_cpu: CPUs for conformer prep (None = all available).
            print_msgs: verbose docking logs.
        """
        self.qv_dir = str(qv_dir) if qv_dir is not None else str(_DEFAULT_QV_DIR)
        self.receptor_name = receptor_name
        self.vina_mode = vina_mode
        self.exhaustiveness = exhaustiveness
        self.docking_batch_size = docking_batch_size
        self.n_conformers = n_conformers
        self.gnina = gnina
        self.conformer_attempts = conformer_attempts
        self.docking_attempts = docking_attempts
        self.n_gpu = n_gpu
        self.n_cpu = n_cpu
        self.print_msgs = print_msgs
        self._proxy = None  # built lazily on first score() (keeps import laptop-safe)

    # ----------------------------------------------------------------- public API
    def score(self, smiles: List[str]) -> List[float]:
        """Dock a batch against sEH; return Vina affinity per SMILES (nan on failure)."""
        proxy = self._build_proxy()
        out: List[float] = []
        for chunk in _chunks(list(smiles), self.docking_batch_size):
            chunk = list(chunk)
            # Upstream ``dock_batch_qv2gpu`` unpacks ``docking_module_gpu(smiles)``
            # directly, and that call returns a BARE ``None`` (not a 2-tuple) when a
            # whole chunk fails prep/docking -- e.g. every RGFN-generated molecule in
            # the chunk fails RDKit/Meeko conformer embedding. That raises
            # ``TypeError: cannot unpack non-iterable NoneType`` *inside* the upstream
            # call, before our ``scores is None`` guard runs, and (unguarded) it kills
            # a multi-hour AL run over one unlucky chunk. Treat any such whole-chunk
            # failure as nan-for-the-chunk so the loop drops those molecules rather
            # than crashing; the loop's all-NaN abort guard still catches a wholesale
            # backend failure (every chunk nan). Mirrors the GPU differential oracle,
            # which wraps its own dock call the same way.
            try:
                result = proxy.dock_batch_qv2gpu(chunk)
            except Exception as exc:  # noqa: BLE001 - a bad chunk must not kill the run
                print(
                    f"[docking_seh] WARNING chunk of {len(chunk)} failed to dock "
                    f"({type(exc).__name__}: {exc}); scoring it nan.",
                    flush=True,
                )
                result = None
            if not (isinstance(result, tuple) and len(result) == 2):
                # Bare None / malformed return -> whole-chunk failure.
                out.extend([float("nan")] * len(chunk))
                continue
            scores, _poses = result
            if scores is None:
                # Whole-batch docking failure -> nan for every molecule in it.
                out.extend([float("nan")] * len(chunk))
            else:
                # Per-molecule failures come back as None; raw scores are negative.
                out.extend(float("nan") if s is None else float(s) for s in scores)
        return out

    # ----------------------------------------------------------------- internals
    def _build_proxy(self):
        """Lazily construct the upstream GPU-docking proxy (defers meeko/openbabel)."""
        if self._proxy is None:
            # Imported here, not at module top, so this oracle imports on a laptop
            # where meeko/openbabel/Vina-GPU are absent (mirrors Docking6TD3Oracle).
            from rgfn.gfns.reaction_gfn.proxies.docking_proxy.docking_proxy import (
                DockingMoleculeProxy,
            )

            self._proxy = DockingMoleculeProxy(
                qv_dir=self.qv_dir,
                vina_mode=self.vina_mode,
                gnina=self.gnina,
                receptor_name=self.receptor_name,
                print_msgs=self.print_msgs,
                conformer_attempts=self.conformer_attempts,
                n_conformers=self.n_conformers,
                docking_attempts=self.docking_attempts,
                docking_batch_size=self.docking_batch_size,
                exhaustiveness=self.exhaustiveness,
                n_gpu=self.n_gpu,
                n_cpu=self.n_cpu,
            )
        return self._proxy


@gin.configurable()
class DockingClpPOracle(DockingSEHOracle):
    """Single-target GPU-docking oracle for **ClpP** (caseinolytic protease P).

    Identical machinery to :class:`DockingSEHOracle` — a thin wrapper over upstream
    ``DockingMoleculeProxy`` (QuickVina2-GPU) returning the raw Vina binding energy
    (more negative = better, so ``higher_is_better = False``) — but against the ClpP
    receptor. The box centre/size + prepared ``data/targets/ClpP.pdbqt`` come from the
    upstream receptor tables (``docking_proxy.RECEPTOR_CENTERS['ClpP']``), so this is a
    one-line target swap. It is one of the RGFN paper's benchmark docking targets
    (``configs/rgfn_clpp_docking.gin``).
    """

    name = "docking_clpp"

    def __init__(self, receptor_name: str = "ClpP", **kwargs):
        super().__init__(receptor_name=receptor_name, **kwargs)
