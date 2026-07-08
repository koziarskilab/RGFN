"""Trainable atom-graph MPNN surrogate (the proxy ``M``) for the SCENT
active-learning loop — the SCENT counterpart of ``glue.proxies.LearnedGlueProxy``.

This is the fast in-loop reward of the active-learning loop (``[bengio2021gflownet]``
§4.3 / Alg. 1): SCENT's cost-guided reaction-GFlowNet trains against ``M(x)^β``, the
expensive docking oracle ``O`` is queried only on per-round query batches (via the
oracle bridge), and ``M`` is refit on the accumulated labels each round.

Faithfulness — the SAME proxy as the RGFN entrant, by construction
-----------------------------------------------------------------
``[bengio2021gflownet]`` A.4 specifies an MPNN over the RDKit **atom graph**
(NNConv + GRU ×12 → Set2Set → MLP, dim 64). RGFN's ``LearnedGlueProxy`` reuses
RGFN's copy of that network (``MPNNet`` + ``mol2graph``). SCENT is a *fork of RGFN*
whose package is also named ``rgfn`` and bundles the **exact same** symbols at the
**same import paths** — so here we subclass SCENT's ``CachedProxyBase`` and import
SCENT's ``MPNNet`` / ``mol2graph`` / ``mols2batch`` verbatim. Both entrants thus fit
the identical proxy architecture/featurization from scratch (fresh random init, no
pretrained sEH weights — ``load_original_model`` is never called); only the package
the symbols come from differs (our ``rgfn/`` for the RGFN entrant, SCENT's installed
``rgfn`` here, which lives in its own ``scent`` conda env). This is why we do NOT
import ``glue`` (whose heavy oracle deps need the ``rgfn`` env): the proxy is a
self-contained copy, identical line-for-line to ``LearnedGlueProxy`` except for its
class name + this docstring.

Label scaling & sign mirror ``LearnedGlueProxy``: labels are standardised at ``fit``
time; predictions live in standardised space; the *Reward* (SCENT's exponential
boosting, configs/rewards/exponential.gin) applies sign + positivity, giving the GFN
target ``exp(signed·β)`` — identical to the RGFN entrant. ``higher_is_better=False``
for the 6TD3 ``dvina`` differential (more negative = better, so ``signed = -pred``).
"""

from typing import List, Optional

import gin
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem

# All four imports resolve to SCENT's installed `rgfn` fork (NOT our repo-local
# rgfn/) because this module is only ever imported inside the `scent` conda env,
# where SCENT is `pip install -e .`. The symbols are byte-for-byte the ones
# glue.proxies.LearnedGlueProxy uses.
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.gfns.reaction_gfn.policies.graph_transformer import (
    _chunks,
    mol2graph,
    mols2batch,
)
from rgfn.gfns.reaction_gfn.proxies.seh_proxy import NUM_ATOMIC_NUMBERS, MPNNet
from rgfn.shared.proxies.cached_proxy import CachedProxyBase

# Feature width produced by mol2graph(one_hot_atom=True, donor_features=False),
# matching load_original_model() in seh_proxy.py: 14 base feats + 1 + atom one-hot.
_NUM_FEAT = 14 + 1 + NUM_ATOMIC_NUMBERS


@gin.configurable()
class LearnedDockingProxy(CachedProxyBase[ReactionState]):
    """A refit-able atom-graph MPNN proxy (Bengio-2021 / RGFN ``MPNNet``).

    The active-learning loop calls :meth:`fit` to (re)train the network on the
    accumulated oracle labels, then SCENT queries it as a normal proxy during GFN
    training. Because predictions change after every refit, the loop must call
    :meth:`clear_cache` (inherited) right after :meth:`fit`.
    """

    def __init__(
        self,
        batch_size: int = 128,
        dim: int = 64,
        num_conv_steps: int = 12,
        lr: float = 5e-4,
        weight_decay: float = 0.0,
        max_epochs: int = 100,
        patience: int = 10,
        val_fraction: float = 0.1,
        higher_is_better: bool = False,
        clip: float = 10.0,
        seed: int = 42,
    ):
        """
        Args:
            batch_size: minibatch size for both training and inference.
            dim: MPNN hidden width (paper uses 64).
            num_conv_steps: GRU graph-conv iterations (paper uses 12).
            lr: Adam learning rate for proxy fitting (paper uses 5e-4).
            weight_decay: optional L2 regularisation for the proxy optimiser.
            max_epochs: cap on epochs per :meth:`fit` call.
            patience: early-stopping patience (epochs without val-MSE improvement).
            val_fraction: fraction of the dataset held out for early stopping.
            higher_is_better: MUST match the oracle whose labels this proxy is fit
                on. For the 6TD3 ``dvina`` differential this is **False** (Vina
                binding energy: more negative = better). The loop asserts this
                equals the oracle's flag to catch sign mismatches.
            clip: symmetric sanity bound on (standardised) predictions. NOT a
                positivity assumption — SCENT's exponential reward boosting keeps
                the GFN reward ``exp(signed·β)`` positive for any prediction sign.
            seed: RNG seed for the train/val split and weight init reproducibility.
        """
        super().__init__()
        self.device = "cpu"
        self.batch_size = batch_size
        self.dim = dim
        self.num_conv_steps = num_conv_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self._higher_is_better = higher_is_better
        self.clip = clip
        self.seed = seed

        self.model = self._build_model()
        self._label_mean: float = 0.0
        self._label_std: float = 1.0
        self._is_fitted: bool = False

        # Invalid (early-terminal) states get the worst-case standardised value so
        # they always receive the lowest reward (sign-aware).
        self.cache = {ReactionStateEarlyTerminal(None): self._failed_value}

    @property
    def _failed_value(self) -> float:
        """Worst-case standardised prediction for invalid molecules (sign-aware)."""
        return -self.clip if self._higher_is_better else self.clip

    def _build_model(self) -> MPNNet:
        torch.manual_seed(self.seed)
        # Fresh random-init MPNNet — NO pretrained weights (load_original_model is
        # never called). num_vec=0 / num_feat=_NUM_FEAT mirror that function's
        # *shape* only, so the input matches mol2graph's node-feature width exactly.
        return MPNNet(
            num_feat=_NUM_FEAT,
            num_vec=0,
            dim=self.dim,
            num_out_per_mol=1,
            num_conv_steps=self.num_conv_steps,
        ).to(self.device)

    @property
    def is_non_negative(self) -> bool:
        # Standardised predictions (mean ~0) may be negative — fine for exponential
        # reward boosting; this proxy does not support "linear" boosting.
        return False

    @property
    def higher_is_better(self) -> bool:
        return self._higher_is_better

    # ------------------------------------------------------------------ training
    def fit(self, smiles: List[str], labels: List[float]) -> dict:
        """Refit the MPNN on accumulated ``(SMILES, oracle-label)`` pairs.

        Implements "Fit M on D_{i-1}" of ``[bengio2021gflownet]`` Alg. 1. Labels are
        standardised; sign/positivity of the reward is handled downstream by SCENT's
        Reward (via ``higher_is_better`` + exponential boosting). Returns a small
        fit-metrics dict for logging.
        """
        graphs, ys = [], []
        for smi, y in zip(smiles, labels):
            mol = Chem.MolFromSmiles(smi) if smi is not None else None
            if mol is None or y is None or not np.isfinite(y):
                continue
            graphs.append(mol2graph(mol))
            ys.append(float(y))
        if len(graphs) < 2:
            raise ValueError(
                f"LearnedDockingProxy.fit needs >=2 valid (smiles,label) pairs, got {len(graphs)}."
            )

        y_arr = np.asarray(ys, dtype=np.float64)
        self._label_mean = float(y_arr.mean())
        self._label_std = float(y_arr.std()) or 1.0
        y_std = (y_arr - self._label_mean) / self._label_std
        targets = torch.tensor(y_std, dtype=torch.float, device=self.device).view(-1, 1)

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(graphs))
        n_val = max(1, int(round(self.val_fraction * len(graphs))))
        val_idx, train_idx = set(perm[:n_val].tolist()), perm[n_val:].tolist()
        if not train_idx:  # tiny datasets: train on everything, validate on it too
            train_idx = list(range(len(graphs)))
            val_idx = set(train_idx)

        # Fresh weights each round (paper refits; we do not anneal from last round
        # to avoid compounding drift — documented choice, matches LearnedGlueProxy).
        self.model = self._build_model()
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        epochs_without_improvement = 0
        epochs_run = 0
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
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()
        self._is_fitted = True
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
    def _compute_proxy_output(self, states: List[ReactionState]) -> List[float]:
        # Robust to early-terminal (invalid) states, which have no `.molecule`.
        # The loop clears the cache after each refit, so we can't rely on the
        # pre-seeded ReactionStateEarlyTerminal entry being present here.
        out: List[float] = [self._failed_value] * len(states)
        positions, smiles = [], []
        for i, state in enumerate(states):
            mol = getattr(state, "molecule", None)
            if mol is None:
                continue
            positions.append(i)
            smiles.append(mol.smiles)
        for i, pred in zip(positions, self._predict(smiles)):
            out[i] = pred
        return out

    @torch.no_grad()
    def _predict(self, smiles: List[str]) -> List[float]:
        self.model.eval()
        out: List[float] = []
        for chunk in _chunks(smiles, self.batch_size):
            graphs, valid = [], []
            for s in chunk:
                mol = Chem.MolFromSmiles(s) if s is not None else None
                if mol is None:
                    valid.append(False)
                    continue
                graphs.append(mol2graph(mol))
                valid.append(True)
            preds: List[Optional[float]] = []
            if graphs:
                batch = mols2batch(graphs).to(self.device)
                preds = list(self.model(batch).view(-1).cpu().numpy().tolist())
            it = iter(preds)
            for ok in valid:
                out.append(
                    float(np.clip(next(it), -self.clip, self.clip)) if ok else self._failed_value
                )
        return out

    def set_device(self, device: str, recursive: bool = True):
        self.device = device
        self.model.to(device)
