"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary classifier that maps the
feature vectors produced by ``aggregation.py`` to a truthful (0) /
hallucinated (1) label.  Called from ``solution.py`` via
``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) keep their
required signatures.

Design notes (this implementation)
----------------------------------
The dataset is small (689 rows) and the feature vector is comparatively
high-dimensional (~2.7k dims with the default aggregation).  In that
regime aggressive regularisation matters far more than network depth,
so we use:

* a 2-hidden-layer MLP with GELU activations and dropout (0.4 / 0.3);
* AdamW with weight decay 1e-3 and a fixed learning-rate schedule;
* **early stopping** on a 15% internal validation slice carved off
  inside ``fit`` (separate from the validation slice the evaluator
  hands to ``fit_hyperparameters``);
* class-balanced ``BCEWithLogitsLoss`` (``pos_weight = n_neg / n_pos``);
* threshold tuning for **accuracy** (the competition's primary metric)
  with F1 as a tiebreaker.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
_HIDDEN_1 = 256
_HIDDEN_2 = 64
_DROPOUT_1 = 0.4
_DROPOUT_2 = 0.3
_LR = 1e-3
_WEIGHT_DECAY = 1e-3
_MAX_EPOCHS = 400
_PATIENCE = 30
_INNER_VAL_FRAC = 0.15
_SEED = 1337


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Architecture (built lazily in :meth:`fit` once ``input_dim`` is known):

        Linear(in -> 256) -> GELU -> Dropout(0.4)
        -> Linear(256 -> 64) -> GELU -> Dropout(0.3)
        -> Linear(64 -> 1)

    Features are pre-scaled with :class:`StandardScaler`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None       # built lazily in fit()
        self._scaler = StandardScaler()
        self._threshold: float = 0.5                 # tuned by fit_hyperparameters()

    # ------------------------------------------------------------------
    # Network definition
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers.

        Args:
            input_dim: Feature vector dimensionality.
        """
        self._net = nn.Sequential(
            nn.Linear(input_dim, _HIDDEN_1),
            nn.GELU(),
            nn.Dropout(_DROPOUT_1),
            nn.Linear(_HIDDEN_1, _HIDDEN_2),
            nn.GELU(),
            nn.Dropout(_DROPOUT_2),
            nn.Linear(_HIDDEN_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``."""
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors with early stopping.

        Carves a small internal validation slice from ``(X, y)`` to drive
        early stopping; this is *not* the same as the ``X_val`` the
        evaluator later passes to :meth:`fit_hyperparameters`.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``;
               0 = truthful, 1 = hallucinated.

        Returns:
            ``self``.
        """
        # Set seeds for reproducibility of the inner train loop.
        torch.manual_seed(_SEED)
        np.random.seed(_SEED)

        y = np.asarray(y).astype(np.int64)
        X = np.asarray(X).astype(np.float32)

        # ── 1. Standardise features ─────────────────────────────────────────
        X_scaled = self._scaler.fit_transform(X).astype(np.float32)

        # ── 2. Inner train / val split for early stopping ──────────────────
        n = len(y)
        n_pos = int(y.sum())
        # If either class has fewer than 2 samples a stratified split would fail.
        # Decide whether to stratify based on this.
        stratify = y if (n_pos >= 2 and (n - n_pos) >= 2 and n >= 10) else None
        if n >= 10:
            X_tr, X_iv, y_tr, y_iv = train_test_split(
                X_scaled, y,
                test_size=_INNER_VAL_FRAC,
                random_state=_SEED,
                stratify=stratify,
            )
        else:
            # Tiny corner case: fall back to training on everything.
            X_tr, X_iv, y_tr, y_iv = X_scaled, X_scaled, y, y

        # ── 3. Build the network ───────────────────────────────────────────
        self._build_network(X_scaled.shape[1])
        assert self._net is not None

        X_tr_t = torch.from_numpy(X_tr).float()
        y_tr_t = torch.from_numpy(y_tr.astype(np.float32))
        X_iv_t = torch.from_numpy(X_iv).float()
        y_iv_t = torch.from_numpy(y_iv.astype(np.float32))

        # ── 4. Class-balanced loss ─────────────────────────────────────────
        n_pos_tr = int(y_tr.sum())
        n_neg_tr = len(y_tr) - n_pos_tr
        pos_weight = torch.tensor(
            [n_neg_tr / max(n_pos_tr, 1)], dtype=torch.float32
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Validation loss is left unweighted so that the early-stop signal
        # is comparable to the true validation BCE.
        val_criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY
        )

        # ── 5. Training loop with early stopping ───────────────────────────
        best_val_loss = float("inf")
        best_state: dict | None = None
        epochs_no_improve = 0

        for epoch in range(_MAX_EPOCHS):
            self.train()
            optimizer.zero_grad()
            logits = self(X_tr_t)
            loss = criterion(logits, y_tr_t)
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_logits = self(X_iv_t)
                val_loss = val_criterion(val_logits, y_iv_t).item()

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= _PATIENCE:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        self.eval()

        # ── 6. Tune threshold on the inner validation slice ────────────────
        # ``solution.py`` trains the *final* probe (used for predictions.csv)
        # without ever calling ``fit_hyperparameters``.  Tuning the
        # threshold here ensures the final probe also has a well-chosen
        # threshold by default.  ``fit_hyperparameters`` may later
        # override this on a larger validation set.
        if len(X_iv) >= 4 and len(np.unique(y_iv)) == 2:
            with torch.no_grad():
                iv_logits = self(X_iv_t)
                iv_probs = torch.sigmoid(iv_logits).cpu().numpy()
            self._tune_threshold_from_probs(iv_probs, y_iv)

        return self

    def _tune_threshold_from_probs(
        self, probs: np.ndarray, y_true: np.ndarray
    ) -> None:
        """Shared threshold-tuning helper.

        Maximises accuracy, with F1 as tiebreaker.
        """
        sorted_p = np.unique(probs)
        midpoints = (
            (sorted_p[:-1] + sorted_p[1:]) / 2.0
            if len(sorted_p) > 1
            else np.array([])
        )
        candidates = np.unique(
            np.concatenate([midpoints, np.linspace(0.0, 1.0, 101)])
        )

        best_threshold = 0.5
        best_acc = -1.0
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            acc = accuracy_score(y_true, y_pred_t)
            f1 = f1_score(y_true, y_pred_t, zero_division=0)
            if (acc > best_acc) or (acc == best_acc and f1 > best_f1):
                best_acc = acc
                best_f1 = f1
                best_threshold = float(t)
        self._threshold = best_threshold

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------
    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set.

        Maximises **accuracy** (the competition's primary metric), with
        F1 as a tiebreaker.  The chosen threshold is stored in
        ``self._threshold`` and used by subsequent ``predict`` calls.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self``.
        """
        probs = self.predict_proba(X_val)[:, 1]
        self._tune_threshold_from_probs(probs, np.asarray(y_val).astype(int))
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 is the
            probability of class 1 (hallucinated).
        """
        X_scaled = self._scaler.transform(np.asarray(X).astype(np.float32))
        X_t = torch.from_numpy(X_scaled.astype(np.float32))
        self.eval()
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).cpu().numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
