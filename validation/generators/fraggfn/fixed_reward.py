"""Frozen pretrained sEH proxy as a **fixed reward generator** for FragGFN.

The active-learning FragGFN entrant (``al_loop.py``) trains against a refit-able
``AtomMPNNProxy`` ``M``. The **fixed-reward** entrant trains against this instead: the
pretrained Bengio-2021 sEH MPNN, frozen, called directly as the reward every step —
mirroring the RGFN paper's sEH-proxy benchmark. There is no oracle and no refit.

Crucially, this wraps ``gflownet.models.bengio2021flow.load_original_model()`` — the
*same* checkpoint + graph featurization RGFN's ``rgfn...seh_proxy.SehMoleculeProxy``
uses — so the matched four-way sEH comparison shares one reward generator. The reward
math is bit-for-bit RGFN's: this exposes ``reward(smiles) = exp(clip(value))`` (β applied
later by the task's constant-temperature conditional → ``exp(value·β)``), identical to
``AtomMPNNProxy.reward`` but with a frozen model instead of the in-loop proxy.
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
    """Frozen sEH MPNN reward generator (drop-in for ``AtomMPNNProxy`` as the task reward).

    Only the read interface the task/loop touch is implemented: ``reward`` (used by
    ``FragGFNTask.compute_obj_properties``), ``predict`` (raw values for the score column),
    ``higher_is_better``, and a no-op ``fit`` for interface parity (never called in a
    fixed-reward run).
    """

    higher_is_better = True  # sEH proxy: higher value = better binder

    def __init__(self, device: str = "cpu", clip: float = 10.0, batch_size: int = 128):
        self.device = device
        self.clip = float(clip)
        self.batch_size = int(batch_size)
        self.model = bengio2021flow.load_original_model()
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, smiles: List[str]) -> List[float]:
        """Raw sEH proxy value per SMILES (higher = better); ``nan`` for invalid molecules.

        Same featurization as RGFN's ``SehMoleculeProxy`` (bengio2021flow ``mol2graph`` /
        ``mols2batch``), so the value is identical to what RGFN trains against.
        """
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
        """Positive GFN reward ``exp(clip(value))`` per SMILES (β applied later by the
        task's temperature conditional → ``exp(value·β)`` — identical to RGFN's
        exponential-boosted reward and to ``AtomMPNNProxy.reward``)."""
        rewards: List[float] = []
        for v in self.predict(smiles):
            if v != v:  # NaN (invalid) — a tiny reward so the trajectory is discouraged
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
    proxy. Reproduces ``tdc.Oracle("DRD2")`` *bit-for-bit* (verified) without installing the
    heavy PyTDC stack into this pinned env: it loads the same cached sklearn model
    (``oracle/drd2_current.pkl``) and applies TDC's exact featurization — count Morgan
    fingerprints with ``useFeatures=True`` (FCFP6), folded to 2048. So RGFN (`DRD2Proxy` =
    tdc oracle) and this baseline entrant share the identical reward generator.
    """

    higher_is_better = True  # DRD2 activity probability: higher = better

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
        """DRD2 activity probability per SMILES (higher = better); ``nan`` for invalid."""
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
        """Positive GFN reward ``exp(clip(value))`` (β applied later → ``exp(value·β)``)."""
        rewards: List[float] = []
        for v in self.predict(smiles):
            if v != v:
                rewards.append(float(np.exp(-self.clip)))
            else:
                rewards.append(float(np.exp(np.clip(v, -self.clip, self.clip))))
        return rewards

    def fit(self, *args, **kwargs) -> dict:  # interface parity; never called
        return {}

    def set_device(self, *args, **kwargs) -> None:  # sklearn model — CPU only
        pass


class DockingBridgeReward:
    """Per-step GPU **docking** as a fixed reward — reached across the env boundary.

    FragGFN runs in its own env and cannot import ``glue``, so (like the AL oracle
    bridge and ``ingest_candidates``) it shells out to ``scripts/score_batch.py`` under
    the ``rgfn`` env. Here that happens **every training step**: the RGFN paper's
    "docking directly in the training loop" for a *baseline*. The benchmark (job 69691)
    showed this is feasible for a single run — the ~35-44 s subprocess startup amortizes
    to ~10-13 % over a ~64-molecule step (QV2-GPU pose-gen + CPU gnina).

    Reward math is RGFN's docking convention (``docking_proxy.py``: ``clip(-raw/norm,
    0, inf)``; same as ``glue.proxies.OracleRewardProxy``): the raw docking score is
    *lower-is-better* (Vina energy / dvina), so the GFN **value** is ``clip(-raw/norm,
    0, inf)`` (higher-is-better), and ``reward = exp(clip(value))`` (β applied later by
    the task → ``exp(value·β)``, matching the frozen-proxy rewards above).

    Interface parity with ``SEHFrozenReward``: ``reward`` (task training),
    ``predict`` (the VALUE = score column), ``higher_is_better``, no-op ``fit`` /
    ``set_device``; plus ``raw_scores`` (the raw docking score for a provenance column).
    Docking results are cached per canonical SMILES, so a molecule re-sampled across
    steps is docked once.
    """

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

    # ------------------------------------------------------------------ interface
    def predict(self, smiles: List[str]) -> List[float]:
        """The GFN **value** per SMILES = ``clip(-raw/norm, 0, inf)`` (higher = better)."""
        return [self._value(r) for r in self._dock(smiles)]

    def raw_scores(self, smiles: List[str]) -> List[float]:
        """Raw docking score per SMILES (Vina/dvina, lower = better; ``nan`` on failure)."""
        return self._dock(smiles)

    def reward(self, smiles: List[str]) -> List[float]:
        """Positive GFN reward ``exp(clip(value))`` (β applied later → ``exp(value·β)``)."""
        return [float(np.exp(min(v, self.clip))) for v in self.predict(smiles)]

    def fit(self, *args, **kwargs) -> dict:  # interface parity; never called (fixed reward)
        return {}

    def set_device(self, *args, **kwargs) -> None:  # docking runs out-of-process
        pass

    # ------------------------------------------------------------------- internals
    def _value(self, raw: float) -> float:
        """lower-is-better raw -> non-negative higher-is-better GFN value."""
        if raw is None or raw != raw:  # nan / None -> dock failure
            return self.failed_score
        return max(-float(raw) / self.norm, 0.0)

    def _dock(self, smiles: List[str]) -> List[float]:
        """Raw docking score per SMILES (cached by canonical SMILES). One
        ``score_batch.py`` subprocess per step for the uncached molecules."""
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
            self._free_gpu_cache()  # let the bridge's QuickVina2-GPU allocate (Logs/014)
            subprocess.run(cmd, check=True, cwd=str(self.repo_root))
            labels = self._read_labels(lbl_path, todo)
            for c, lab in zip(todo, labels):
                self._cache[c] = lab
            if todo and all(lab != lab for lab in labels):  # every dock returned nan
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
        """Read the bridge's labels CSV (``smiles,label,...``) keyed by SMILES, in order."""
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
        """Return torch's reserved-but-unused GPU memory so the docking subprocess can
        allocate on the same card (Logs/014). Best-effort/no-op without torch/CUDA."""
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
