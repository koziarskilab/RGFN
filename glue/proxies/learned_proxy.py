"""Trainable atom-graph MPNN surrogate (the proxy ``M``) for the active-learning loop.

This is the fast in-loop reward of the active-learning loop described in
``docs/RESEARCH_CONTEXT.md`` (``[bengio2021gflownet]`` §4.3 / Alg. 1). RGFN trains
against ``M(x)^beta``; the expensive oracle ``O`` is queried only on per-round
query batches and ``M`` is refit on the accumulated labels each round.

Faithfulness — we reuse the *published* proxy architecture rather than inventing one:
    ``[bengio2021gflownet]`` A.4 specifies an MPNN over the RDKit **atom graph**
    (NNConv + GRU applied 12 times -> Set2Set -> 3-layer MLP, 64 hidden units,
    LeakyReLU everywhere). RGFN already ships exactly that network as ``MPNNet``
    (the class behind ``SehMoleculeProxy``, which loads Bengio's pretrained sEH
    weights). We import ``MPNNet`` and the same ``mol2graph`` / ``mols2batch``
    featurization verbatim, so our surrogate matches the paper bit-for-bit.

    Why the atom graph and not RGFN's fragment graph: RGFN builds molecules
    reaction-by-reaction, so its *policy / flow predictor* reads the fragment
    (block) graph. The *proxy* only has to score a finished molecule's property,
    and both ``[bengio2021gflownet]`` and RGFN's own ``SehMoleculeProxy`` do that
    from the atom graph (SMILES -> RDKit mol). The reaction structure is
    irrelevant to the proxy.

Documented divergences from the publications (flag for Balam validation):
    1. Target. We predict our docking **neosubstrate differential** (the DDB1
       cooperativity bonus for 6TD3), not AutoDock sEH binding affinity. This is
       the project's novel oracle signal, not something inherited from a paper.
    2. Trainable. The shipped ``SehMoleculeProxy`` is inference-only with frozen
       pretrained weights. We add ``fit()`` so ``M`` can be warm-started on the
       seed set ``D_0`` and refit on the full history each round — which is
       precisely what ``[bengio2021gflownet]`` Alg. 1 prescribes.
    3. Init. We deliberately do **not** load any pretrained weights. We reuse
       only the ``MPNNet`` *architecture* and train it from scratch: fresh random
       init, then fit on ``D_0`` and refit each round. This matches the paper,
       which "initializes an MPNN proxy" and trains it on ``D_0`` — it does not
       warm-start from the sEH weights (note ``load_original_model`` /
       ``SEHProxyWrapper`` from ``seh_proxy.py`` are never imported here). The
       sEH proxy predicts a different target (sEH affinity), so its weights would
       be the wrong prior anyway.
    4. Label scaling & sign. ``[bengio2021gflownet]`` A.5 scales ``O(x)`` to be
       positive before fitting ``M`` and, for docking (lower-is-better), "takes
       its opposite". We instead standardise labels (subtract mean / divide std)
       at ``fit`` time for a stable prediction scale, and let the Reward handle
       sign + positivity: the base config uses *exponential* boosting, so the
       reward ``exp(signed_value * beta)`` is positive for any real prediction,
       and ``higher_is_better`` (a constructor arg) sets the sign. For the 6TD3
       ``ddb1_dvina`` differential, ``higher_is_better=False`` (Vina binding
       energy: more negative = better) — so we do NOT negate the label; the
       Reward negates ``signed_value`` for us.
    5. Scale. The molecule-domain run in the paper uses ``|D_0| = 2000`` and 200
       freshly-docked molecules per round. Ours are gin-configurable and default
       to whatever labelled data we have; the gap is expected and logged.
"""

from typing import List, Optional

import gin
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem

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
class LearnedGlueProxy(CachedProxyBase[ReactionState]):
    """A refit-able atom-graph MPNN proxy (Bengio-2021 / RGFN ``MPNNet``).

    The active-learning loop calls :meth:`fit` to (re)train the network on the
    accumulated oracle labels, then queries it as a normal RGFN proxy during GFN
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
        higher_is_better: bool = True,
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
            val_fraction: fraction of the dataset held out for early stopping
                (the paper uses a fixed 3000-molecule validation set; we hold out
                a fraction instead — documented divergence for small datasets).
            higher_is_better: MUST match the oracle whose labels this proxy is fit
                on. For the 6TD3 ``ddb1_dvina`` differential this is **False**
                (Vina binding energy: more negative = better). The loop asserts
                this equals the oracle's flag to catch mismatches.
            clip: symmetric sanity bound on (standardised) predictions, ``[-clip,
                +clip]``. This is NOT a positivity assumption — the base config
                uses *exponential* reward boosting, so the GFN reward
                (``exp(signed_value * beta)``) is positive regardless of sign;
                the clip only guards against runaway outliers blowing up ``exp``.
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
        # Predictions live in standardised label space (mean 0, std 1); these
        # stats are learned at fit() time and kept for logging.
        self._label_mean: float = 0.0
        self._label_std: float = 1.0
        self._is_fitted: bool = False

        # Invalid (early-terminal) states get the worst-case standardised value so
        # they always receive the lowest reward. Sign-aware: with lower-is-better
        # the worst case is the *largest* value (reward ~ exp(-value*beta)).
        self.cache = {ReactionStateEarlyTerminal(None): self._failed_value}

    @property
    def _failed_value(self) -> float:
        """Worst-case standardised prediction for invalid molecules (sign-aware)."""
        return -self.clip if self._higher_is_better else self.clip

    def _build_model(self) -> MPNNet:
        torch.manual_seed(self.seed)
        # Fresh random-init MPNNet -- NO pretrained weights are loaded (we never
        # call seh_proxy.load_original_model). num_vec=0 / num_feat=_NUM_FEAT
        # mirror that function's *shape* only, so the network input matches
        # mol2graph's node-feature width exactly.
        return MPNNet(
            num_feat=_NUM_FEAT,
            num_vec=0,
            dim=self.dim,
            num_out_per_mol=1,
            num_conv_steps=self.num_conv_steps,
        ).to(self.device)

    @property
    def is_non_negative(self) -> bool:
        # Predictions are standardised (mean ~0) and may be negative. That's fine
        # for the base config's *exponential* reward boosting; this proxy does not
        # support "linear" boosting (which requires non-negative proxy values).
        return False

    @property
    def higher_is_better(self) -> bool:
        # Set to match the oracle being fit (False for the 6TD3 ddb1_dvina
        # binding-energy differential, where more negative = better).
        return self._higher_is_better

    # ------------------------------------------------------------------ training
    def fit(self, smiles: List[str], labels: List[float]) -> dict:
        """Refit the MPNN on accumulated (SMILES, oracle-label) pairs.

        Implements the "Fit M on D_{i-1}" step of ``[bengio2021gflownet]`` Alg. 1.
        Labels are standardised (subtract mean / divide std) so the prediction
        scale is stable across rounds; sign/positivity of the reward is handled
        downstream by the Reward (via ``higher_is_better`` + exponential boosting).
        The standardisation stats are stored for logging.

        Returns a small dict of fit metrics (best val MSE, epochs, n) for logging.
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
                f"LearnedGlueProxy.fit needs >=2 valid (smiles,label) pairs, got {len(graphs)}."
            )

        y_arr = np.asarray(ys, dtype=np.float64)
        self._label_mean = float(y_arr.mean())
        self._label_std = float(y_arr.std()) or 1.0
        y_std = (y_arr - self._label_mean) / self._label_std
        targets = torch.tensor(y_std, dtype=torch.float, device=self.device).view(-1, 1)

        # Reproducible train/val split for early stopping.
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(graphs))
        n_val = max(1, int(round(self.val_fraction * len(graphs))))
        val_idx, train_idx = set(perm[:n_val].tolist()), perm[n_val:].tolist()
        if not train_idx:  # tiny datasets: train on everything, validate on it too
            train_idx = list(range(len(graphs)))
            val_idx = set(train_idx)

        # Fresh weights each round (paper refits; we do not anneal from last round
        # to avoid compounding drift — documented choice, revisit on Balam).
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
                vals = self.model(batch).view(-1).cpu().numpy().tolist()
                preds = list(vals)
            it = iter(preds)
            for ok in valid:
                # Standardised prediction, clamped to +/-clip as numerical safety
                # (NOT a positivity assumption — sign is handled by the Reward via
                # higher_is_better). Invalid molecules get the worst-case value.
                out.append(
                    float(np.clip(next(it), -self.clip, self.clip)) if ok else self._failed_value
                )
        return out

    def set_device(self, device: str, recursive: bool = True):
        self.device = device
        self.model.to(device)
