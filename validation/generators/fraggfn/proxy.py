"""Trainable atom-graph MPNN surrogate (the proxy ``M``) for the FragGFN
active-learning loop — the FragGFN counterpart of ``glue.proxies.LearnedGlueProxy``.

This is the fast in-loop reward of the active-learning loop (``[bengio2021gflownet]``
§4.3 / Alg. 1): the fragment-GFlowNet trains against ``M(x)^β``, the expensive
docking oracle ``O`` is queried only on per-round query batches (via the oracle
bridge), and ``M`` is refit on the accumulated labels each round.

Faithfulness — the SAME proxy as the RGFN entrant, by construction
-----------------------------------------------------------------
``[bengio2021gflownet]`` A.4 specifies an MPNN over the RDKit **atom graph**
(NNConv + GRU ×12 → Set2Set → MLP, dim 64). RGFN's ``LearnedGlueProxy`` reuses
RGFN's copy of that network (``rgfn ... MPNNet`` + ``mol2graph``). FragGFN runs in
its own env where ``rgfn`` is unavailable, so here we import the **same network
from its original source** — Recursion's ``gflownet.models.bengio2021flow``
(``MPNNet`` / ``mol2graph`` / ``mols2batch``), which is the very code RGFN copied.
So both entrants fit the identical proxy architecture/featurization; only the
import path differs. We do **not** load the pretrained sEH weights — fresh random
init, fit on ``D_0``, refit each round (Alg. 1), matching ``LearnedGlueProxy``.

Label scaling & sign mirror ``LearnedGlueProxy``: labels are standardised at
``fit`` time; predictions live in standardised space; the *reward* mapping
(:meth:`reward`) handles sign + positivity so the GFN target ``exp(signed·β)``
matches RGFN's exactly (``higher_is_better=False`` for the 6TD3 ``dvina``
differential: more negative = better, so ``signed = -prediction``).
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from gflownet.models.bengio2021flow import MPNNet, mol2graph, mols2batch
from rdkit import Chem


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


class AtomMPNNProxy:
    """A refit-able atom-graph MPNN regressor (Bengio-2021 / gflownet ``MPNNet``).

    Pure science: maps SMILES → standardised scalar prediction (``predict``), and
    standardised prediction → positive GFN reward (``reward``). No gflownet trainer
    or rgfn dependency, so it is unit-testable on its own and reused by the task's
    reward hook.
    """

    def __init__(
        self,
        higher_is_better: bool = False,
        dim: int = 64,
        num_conv_steps: int = 12,
        lr: float = 5e-4,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        max_epochs: int = 100,
        patience: int = 10,
        val_fraction: float = 0.1,
        clip: float = 10.0,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.higher_is_better = higher_is_better
        self.dim = dim
        self.num_conv_steps = num_conv_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.clip = clip
        self.seed = seed
        self.device = device

        self._num_feat: Optional[int] = None
        self.model: Optional[MPNNet] = None
        self._label_mean: float = 0.0
        self._label_std: float = 1.0
        self.is_fitted: bool = False

    # ------------------------------------------------------------------ featurize
    @staticmethod
    def _graph(smiles: str):
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return None
        try:
            return mol2graph(mol)
        except Exception:
            return None

    def _build_model(self, num_feat: int) -> MPNNet:
        torch.manual_seed(self.seed)
        # Fresh random init — NO pretrained sEH weights (load_original_model is
        # never called). num_vec=0 / num_out_per_mol=1 = a scalar regressor whose
        # input width matches mol2graph's node features exactly.
        return MPNNet(
            num_feat=num_feat,
            num_vec=0,
            dim=self.dim,
            num_out_per_mol=1,
            num_conv_steps=self.num_conv_steps,
        ).to(self.device)

    @property
    def _failed_value(self) -> float:
        """Worst-case standardised prediction for invalid molecules (sign-aware)."""
        return -self.clip if self.higher_is_better else self.clip

    # ------------------------------------------------------------------- training
    def fit(self, smiles: List[str], labels: List[float]) -> dict:
        """Refit the MPNN on accumulated ``(SMILES, oracle-label)`` pairs.

        Implements "Fit M on D_{i-1}" of ``[bengio2021gflownet]`` Alg. 1. Labels are
        standardised; sign/positivity of the reward is applied later in :meth:`reward`.
        Returns a small fit-metrics dict for logging.
        """
        graphs, ys = [], []
        for smi, y in zip(smiles, labels):
            g = self._graph(smi)
            if g is None or y is None or not np.isfinite(y):
                continue
            graphs.append(g)
            ys.append(float(y))
        if len(graphs) < 2:
            raise ValueError(
                f"AtomMPNNProxy.fit needs >=2 valid (smiles,label) pairs, got {len(graphs)}."
            )

        if self._num_feat is None:
            self._num_feat = int(graphs[0].x.shape[1])

        y_arr = np.asarray(ys, dtype=np.float64)
        self._label_mean = float(y_arr.mean())
        self._label_std = float(y_arr.std()) or 1.0
        y_std = (y_arr - self._label_mean) / self._label_std
        targets = torch.tensor(y_std, dtype=torch.float, device=self.device).view(-1, 1)

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(graphs))
        n_val = max(1, int(round(self.val_fraction * len(graphs))))
        val_idx, train_idx = set(perm[:n_val].tolist()), perm[n_val:].tolist()
        if not train_idx:
            train_idx = list(range(len(graphs)))
            val_idx = set(train_idx)

        self.model = self._build_model(self._num_feat)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        no_improve, epochs_run = 0, 0
        for epoch in range(self.max_epochs):
            epochs_run = epoch + 1
            self.model.train()
            rng.shuffle(train_idx)
            for chunk in _chunks(train_idx, self.batch_size):
                batch = mols2batch([graphs[i] for i in chunk]).to(self.device)
                pred = self.model(batch)
                loss = loss_fn(pred, targets[chunk])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_mse = self._eval_mse(graphs, targets, sorted(val_idx), loss_fn)
            if val_mse < best_val - 1e-6:
                best_val = val_mse
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()
        self.is_fitted = True
        return {
            "proxy_fit_n": len(graphs),
            "proxy_fit_best_val_mse": best_val,
            "proxy_fit_epochs": epochs_run,
            "proxy_label_mean": self._label_mean,
            "proxy_label_std": self._label_std,
        }

    @torch.no_grad()
    def _eval_mse(self, graphs, targets, idx, loss_fn) -> float:
        if not idx:
            return float("inf")
        self.model.eval()
        losses, weights = [], []
        for chunk in _chunks(idx, self.batch_size):
            batch = mols2batch([graphs[i] for i in chunk]).to(self.device)
            pred = self.model(batch)
            losses.append(loss_fn(pred, targets[chunk]).item())
            weights.append(len(chunk))
        return float(np.average(losses, weights=weights))

    # ----------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, smiles: List[str]) -> List[float]:
        """Standardised prediction per SMILES; invalid → worst-case (sign-aware).

        Before the first ``fit`` (no model yet) everything is the failed value, so
        the very first round trains against a flat reward — but the loop always
        warm-starts ``M`` on ``D_0`` before round 1, exactly as Alg. 1 prescribes.
        """
        if self.model is None:
            return [self._failed_value] * len(smiles)
        self.model.eval()
        out: List[float] = []
        for chunk in _chunks(smiles, self.batch_size):
            graphs, valid = [], []
            for s in chunk:
                g = self._graph(s)
                valid.append(g is not None)
                if g is not None:
                    graphs.append(g)
            preds: List[float] = []
            if graphs:
                batch = mols2batch(graphs).to(self.device)
                preds = self.model(batch).view(-1).cpu().numpy().tolist()
            it = iter(preds)
            for ok in valid:
                out.append(
                    float(np.clip(next(it), -self.clip, self.clip)) if ok else self._failed_value
                )
        return out

    def reward(self, smiles: List[str]) -> List[float]:
        """Positive GFN reward ``exp(signed_value)`` per SMILES (β applied later by
        the task's temperature conditional, giving ``exp(signed·β)`` — identical to
        RGFN's exponential-boosted reward). ``signed = pred`` if higher-is-better
        else ``-pred``."""
        preds = self.predict(smiles)
        rewards: List[float] = []
        for v in preds:
            signed = v if self.higher_is_better else -v
            rewards.append(float(np.exp(np.clip(signed, -self.clip, self.clip))))
        return rewards
