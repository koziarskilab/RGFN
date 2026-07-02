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

from typing import List

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
