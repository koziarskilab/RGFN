"""Frozen pretrained sEH proxy as a **fixed reward generator** for RxnFlow.

The fixed-reward analogue of RxnFlow's refit-able ``AtomMPNNProxy`` ``M``: the pretrained
Bengio-2021 sEH MPNN, frozen, called directly as the reward every step (the RGFN paper's
sEH-proxy benchmark). No oracle, no refit. Wraps
``gflownet.models.bengio2021flow.load_original_model()`` — the *same* checkpoint +
featurization RGFN's ``SehMoleculeProxy`` uses — so the matched four-way sEH comparison
shares one reward generator. Reward math is bit-for-bit RGFN's (see the FragGFN twin,
``validation/generators/fraggfn/fixed_reward.py``): ``reward = exp(clip(value))``, β applied
later by the task's constant-temperature conditional → ``exp(value·β)``.

(Kept as a per-adapter copy — like this generator's own ``proxy.py`` / ``route.py`` — so
RxnFlow's env stays self-contained and independent of the FragGFN adapter.)
"""

import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from gflownet.models import bengio2021flow
from rdkit import Chem


class SEHFrozenReward:
    """Frozen sEH MPNN reward generator (drop-in for ``AtomMPNNProxy`` as the task reward)."""

    higher_is_better = True

    def __init__(self, device: str = "cpu", clip: float = 10.0, batch_size: int = 128):
        self.device = device
        self.clip = float(clip)
        self.batch_size = int(batch_size)
        self.model = bengio2021flow.load_original_model()
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, smiles: List[str]) -> List[float]:
        """Raw sEH proxy value per SMILES (higher = better); ``nan`` for invalid molecules."""
        out: List[float] = []
        for start in range(0, len(smiles), self.batch_size):
            chunk = smiles[start : start + self.batch_size]
            graphs, valid = [], []
            for s in chunk:
                mol = Chem.MolFromSmiles(s) if s else None
                g = bengio2021flow.mol2graph(mol) if mol is not None else None
                valid.append(g is not None)
                if g is not None:
                    graphs.append(g)
            preds: List[float] = []
            if graphs:
                batch = bengio2021flow.mols2batch(graphs).to(self.device)
                preds = self.model(batch).view(-1).cpu().numpy().tolist()
            it = iter(preds)
            for ok in valid:
                out.append(float(next(it)) if ok else float("nan"))
        return out

    def reward(self, smiles: List[str]) -> List[float]:
        """Positive GFN reward ``exp(clip(value))`` per SMILES (β applied later → ``exp(value·β)``)."""
        rewards: List[float] = []
        for v in self.predict(smiles):
            if v != v:  # NaN (invalid)
                rewards.append(float(np.exp(-self.clip)))
            else:
                rewards.append(float(np.exp(np.clip(v, -self.clip, self.clip))))
        return rewards

    def fit(self, *args, **kwargs) -> dict:  # interface parity; never called (fixed reward)
        return {}

    def set_device(self, device: str) -> None:
        self.device = device
        self.model.to(device)


class DRD2FrozenReward:
    """Frozen DRD2 activity oracle as a **fixed reward generator** — the RGFN paper's third
    proxy. Reproduces ``tdc.Oracle("DRD2")`` bit-for-bit (verified) without the heavy PyTDC
    install: loads the cached sklearn model (``oracle/drd2_current.pkl``) + TDC's exact
    featurization (count Morgan, ``useFeatures=True`` = FCFP6, folded to 2048). Matches RGFN's
    ``DRD2Proxy`` (same tdc oracle). Per-adapter copy (like this generator's ``proxy.py``)."""

    higher_is_better = True

    def __init__(self, model_path: str, clip: float = 10.0, **_ignored):
        import pickle

        with open(model_path, "rb") as fh:
            self.model = pickle.load(fh)  # nosec - trusted local TDC oracle
        self.clip = float(clip)

    @staticmethod
    def _fp(mol):
        from rdkit.Chem import AllChem

        f = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
        nfp = np.zeros((1, 2048), np.int32)
        for idx, v in f.GetNonzeroElements().items():
            nfp[0, idx % 2048] += int(v)
        return nfp

    def predict(self, smiles: List[str]) -> List[float]:
        mols = [Chem.MolFromSmiles(s) if s else None for s in smiles]
        valid_idx = [i for i, m in enumerate(mols) if m is not None]
        out = [float("nan")] * len(smiles)
        if valid_idx:
            X = np.concatenate([self._fp(mols[i]) for i in valid_idx], axis=0)
            probs = self.model.predict_proba(X)[:, 1]
            for j, i in enumerate(valid_idx):
                out[i] = float(probs[j])
        return out

    def reward(self, smiles: List[str]) -> List[float]:
        rewards: List[float] = []
        for v in self.predict(smiles):
            if v != v:
                rewards.append(float(np.exp(-self.clip)))
            else:
                rewards.append(float(np.exp(np.clip(v, -self.clip, self.clip))))
        return rewards

    def fit(self, *args, **kwargs) -> dict:
        return {}

    def set_device(self, *args, **kwargs) -> None:
        pass


class DockingBridgeReward:
    """Per-step GPU **docking** as a fixed reward for RxnFlow, reached across the env
    boundary via ``scripts/score_batch.py`` under the ``rgfn`` env.

    Per-adapter twin of ``validation/generators/fraggfn/fixed_reward.DockingBridgeReward``
    (kept a copy so each generator's env stays self-contained). Docks each training step
    (QV2-GPU pose-gen + CPU gnina), converts the lower-is-better raw score to the GFN
    **value** ``clip(-raw/norm, 0, inf)`` and ``reward = exp(clip(value))`` (β applied by
    the task → ``exp(value·β)``), frees torch's GPU cache before each dock (Logs/014), and
    caches results per canonical SMILES. See the FragGFN twin for the full rationale."""

    higher_is_better = True  # the recorded VALUE (clip(-raw/norm)) is higher-is-better

    def __init__(
        self,
        oracle: str,
        repo_root: str,
        norm: float = 1.0,
        failed_score: float = 0.0,
        clip: float = 10.0,
        conda_env: str = "rgfn",
        oracle_args: Optional[Dict] = None,
        workdir: Optional[str] = None,
    ):
        self.oracle = oracle
        self.repo_root = Path(repo_root)
        self.norm = float(norm)
        self.failed_score = float(failed_score)
        self.clip = float(clip)
        self.conda_env = conda_env
        self.oracle_args = dict(oracle_args or {})
        self.workdir = Path(workdir) if workdir else (self.repo_root / "reward_bridge")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, float] = {}  # canonical smiles -> raw docking score
        self._step = 0

    def predict(self, smiles: List[str]) -> List[float]:
        """The GFN **value** per SMILES = ``clip(-raw/norm, 0, inf)`` (higher = better)."""
        return [self._value(r) for r in self._dock(smiles)]

    def raw_scores(self, smiles: List[str]) -> List[float]:
        """Raw docking score per SMILES (Vina/dvina, lower = better; ``nan`` on failure)."""
        return self._dock(smiles)

    def reward(self, smiles: List[str]) -> List[float]:
        """Positive GFN reward ``exp(clip(value))`` (β applied later → ``exp(value·β)``)."""
        return [float(np.exp(min(v, self.clip))) for v in self.predict(smiles)]

    def fit(self, *args, **kwargs) -> dict:
        return {}

    def set_device(self, *args, **kwargs) -> None:
        pass

    def _value(self, raw: float) -> float:
        if raw is None or raw != raw:
            return self.failed_score
        return max(-float(raw) / self.norm, 0.0)

    def _dock(self, smiles: List[str]) -> List[float]:
        canons = [self._canonical(s) for s in smiles]
        todo = [c for c in dict.fromkeys(canons) if c and c not in self._cache]
        if todo:
            self._step += 1
            smi_path = self.workdir / f"step_{self._step:05d}.smi"
            lbl_path = self.workdir / f"step_{self._step:05d}_labels.csv"
            smi_path.write_text("\n".join(todo) + "\n")
            cmd = [
                "conda", "run", "--no-capture-output", "-n", self.conda_env,
                "python", "scripts/score_batch.py",
                "--oracle", self.oracle, "--in", str(smi_path), "--out", str(lbl_path),
            ]  # fmt: skip
            for k, v in self.oracle_args.items():
                cmd += ["--oracle-arg", f"{k}={v}"]
            self._free_gpu_cache()
            subprocess.run(cmd, check=True, cwd=str(self.repo_root))
            labels = self._read_labels(lbl_path, todo)
            for c, lab in zip(todo, labels):
                self._cache[c] = lab
            if todo and all(lab != lab for lab in labels):
                print(
                    f"[dock-bridge] WARNING step {self._step}: all {len(todo)} docks "
                    "failed (nan) -- flat reward this step (check GPU/OpenCL).",
                    flush=True,
                )
        return [self._cache.get(c, float("nan")) if c else float("nan") for c in canons]

    @staticmethod
    def _canonical(smiles: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        return Chem.MolToSmiles(mol) if mol is not None else None

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

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
