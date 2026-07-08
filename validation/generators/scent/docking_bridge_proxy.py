"""``DockingBridgeProxy`` — per-step GPU **docking** as a fixed reward for SCENT.

SCENT's fixed reward is a ``CachedProxyBase`` bound to ``%train_proxy`` (see
``scent_seh_fixed.gin`` → ``@SehMoleculeProxy``). This is the docking counterpart: a
proxy whose ``compute_proxy_output`` docks the batch by shelling out to
``scripts/score_batch.py`` under the ``rgfn`` env (SCENT's ``scent`` env cannot import
``glue``), every training step — the RGFN paper's "docking directly in the training
loop" for the SCENT baseline. The cross-env per-step bridge was validated feasible by
benchmark 69691 (startup amortizes over the step's batch).

It is the SCENT-clone twin of ``glue.proxies.OracleRewardProxy`` (which the RGFN entrant
uses in-process) and of ``DockingBridgeReward`` in the FragGFN/RxnFlow adapters (which
shell out the same way): the raw docking score is *lower-is-better* (Vina energy / dvina),
so the proxy **value** is ``clip(-raw/norm, 0, inf)`` (higher-is-better) and SCENT's
``Reward`` (exponential boosting) applies ``exp(value·β)``. The raw score is preserved as
a ``raw_score`` component so ``ScentFixedRewardRun`` can record it as a candidate column.
``torch.cuda.empty_cache()`` runs before each dock so QuickVina2-GPU is not starved by
SCENT's torch on the same GPU (Logs/014).

Kept a per-clone copy (like this generator's ``proxy.py`` / ``route.py``) so the SCENT env
stays self-contained — it imports SCENT's ``rgfn`` fork, never ``glue``/our ``rgfn``.
"""

import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import gin

# Resolves to SCENT's installed `rgfn` fork (this module is only imported in the scent env).
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase


@gin.configurable()
class DockingBridgeProxy(CachedProxyBase[ReactionState]):
    """Dock each batch via the rgfn-env ``score_batch.py`` bridge; expose a GFN reward."""

    def __init__(
        self,
        oracle: str,
        repo_root: str,
        norm: float = 1.0,
        failed_score: float = 0.0,
        conda_env: str = "rgfn",
        oracle_args: Optional[Dict] = None,
        workdir: Optional[str] = None,
    ):
        super().__init__()
        self.oracle = oracle
        self.repo_root = Path(repo_root)
        self.norm = float(norm)
        self.failed_score = float(failed_score)
        self.conda_env = conda_env
        self.oracle_args = dict(oracle_args or {})
        self.workdir = Path(workdir) if workdir else (self.repo_root / "reward_bridge_scent")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._step = 0
        # early-terminal (invalid) -> worst reward; dict form so all cache entries match.
        self.cache = {
            ReactionStateEarlyTerminal(None): {
                "value": self.failed_score,
                "raw_score": float("nan"),
            }
        }

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True  # the recorded VALUE (clip(-raw/norm)) is higher-is-better

    # ----------------------------------------------------------------- inference
    def _compute_proxy_output(self, states: List[ReactionState]) -> List[Dict[str, float]]:
        smiles: List[Optional[str]] = []
        for state in states:
            mol = getattr(state, "molecule", None)
            smiles.append(mol.smiles if mol is not None else None)
        raws = self._dock([s for s in smiles if s is not None])
        raw_by_smi = {}
        j = 0
        for s in smiles:
            if s is not None:
                raw_by_smi[s] = raws[j]
                j += 1
        out: List[Dict[str, float]] = []
        for s in smiles:
            raw = raw_by_smi.get(s, float("nan")) if s is not None else float("nan")
            if raw != raw:  # nan
                out.append({"value": self.failed_score, "raw_score": float("nan")})
            else:
                out.append({"value": max(-float(raw) / self.norm, 0.0), "raw_score": float(raw)})
        return out

    def _dock(self, smiles: List[str]) -> List[float]:
        """Raw docking score per SMILES via one score_batch.py subprocess (nan on failure)."""
        if not smiles:
            return []
        self._step += 1
        smi_path = self.workdir / f"step_{self._step:05d}.smi"
        lbl_path = self.workdir / f"step_{self._step:05d}_labels.csv"
        smi_path.write_text("\n".join(smiles) + "\n")
        cmd = [
            "conda", "run", "--no-capture-output", "-n", self.conda_env,
            "python", "scripts/score_batch.py",
            "--oracle", self.oracle, "--in", str(smi_path), "--out", str(lbl_path),
        ]  # fmt: skip
        for k, v in self.oracle_args.items():
            cmd += ["--oracle-arg", f"{k}={v}"]
        self._free_gpu_cache()  # let the bridge's QuickVina2-GPU allocate (Logs/014)
        subprocess.run(cmd, check=True, cwd=str(self.repo_root))
        labels = self._read_labels(lbl_path, smiles)
        if smiles and all(lab != lab for lab in labels):
            print(
                f"[dock-bridge-scent] WARNING step {self._step}: all {len(smiles)} docks "
                "failed (nan) -- flat reward this step (check GPU/OpenCL).",
                flush=True,
            )
        return labels

    def set_device(self, device: str, recursive: bool = True):
        self.device = device  # docking runs out-of-process; nothing to move

    @staticmethod
    def _read_labels(path: Path, batch: List[str]) -> List[float]:
        by_smi: Dict[str, float] = {}
        if path.exists():
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        by_smi[row["smiles"]] = float(row["label"])
                    except (KeyError, TypeError, ValueError):
                        by_smi[row.get("smiles", "")] = float("nan")
        return [by_smi.get(s, float("nan")) for s in batch]

    @staticmethod
    def _free_gpu_cache() -> None:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
