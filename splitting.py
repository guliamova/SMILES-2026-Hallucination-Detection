"""
splitting.py — Train / validation / test split utilities (student-implemented).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It returns a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Strategy implemented here
-------------------------
We use **stratified K-fold cross-validation with group awareness**.

* **Groups** are defined by a hash of the first 300 characters of each
  ``prompt`` (the shared context portion).  Roughly 18% of the rows in
  ``data/dataset.csv`` share a context with at least one other row, so
  putting two rows from the same context into different folds would
  artificially inflate test performance.  Group K-fold keeps every group
  fully within a single fold.

* When ``df`` is unavailable we fall back to a plain stratified K-fold.

* For every test fold a small **validation slice** (default 10% of the
  remaining train portion) is carved out, again with group awareness,
  to support threshold tuning in ``HallucinationProbe.fit_hyperparameters``.

The number of folds defaults to ``5`` (≈140 test samples per fold for
the provided 689-row dataset).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)


N_SPLITS = 5
VAL_FRAC = 0.10           # fraction of (train + val) reserved for validation
GROUP_PREFIX_CHARS = 300  # how many chars of `prompt` define a "context group"


def _context_groups(df: pd.DataFrame) -> np.ndarray:
    """Hash the first ``GROUP_PREFIX_CHARS`` characters of each prompt into a group id.

    Rows that share the same context (same SQuAD-style passage and
    instruction header) end up in the same group.  Group K-fold below
    keeps all rows of a given group within a single fold so that test
    performance is not inflated by context leakage from training.
    """
    prompts = df["prompt"].astype(str).str[:GROUP_PREFIX_CHARS]
    return np.array(
        [int(hashlib.md5(p.encode("utf-8")).hexdigest()[:12], 16) for p in prompts],
        dtype=np.int64,
    )


def _stratified_group_kfold(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> list[np.ndarray]:
    """Stratified group K-fold: each group fully within one fold, classes balanced.

    We implement a simple, deterministic version because some
    scikit-learn releases ship ``StratifiedGroupKFold`` and some don't,
    and we want to avoid the dependency drift.  The algorithm:

    1. Aggregate per-group: dominant label and group size.
    2. Sort groups by size (largest first, ties broken by hash for
       determinism) — large groups are placed first to balance folds.
    3. Greedily assign each group to the fold that currently has the
       lowest count of its dominant label.

    Returns a list of ``n_splits`` index arrays, one per fold (the
    *test* indices of that fold).
    """
    rng = np.random.default_rng(random_state)
    unique_groups = np.unique(groups)

    # Per-group dominant label and size.
    group_label = {}
    group_size = {}
    for g in unique_groups:
        mask = groups == g
        labs = y[mask]
        # Dominant label of the group.
        group_label[g] = int(np.bincount(labs.astype(int)).argmax())
        group_size[g] = int(mask.sum())

    # Order by size desc; jitter ties for determinism.
    jitter = rng.uniform(0, 1, size=len(unique_groups))
    order = sorted(
        range(len(unique_groups)),
        key=lambda i: (-group_size[unique_groups[i]], jitter[i]),
    )

    # Fold counters per label.
    fold_counts = [{0: 0, 1: 0} for _ in range(n_splits)]
    group_to_fold: dict[int, int] = {}
    for i in order:
        g = unique_groups[i]
        lab = group_label[g]
        # Pick the fold that currently has the fewest samples of this label.
        best_fold = int(np.argmin([fold_counts[f][lab] for f in range(n_splits)]))
        group_to_fold[int(g)] = best_fold
        fold_counts[best_fold][lab] += group_size[g]

    # Materialise the per-fold test indices.
    folds: list[np.ndarray] = []
    for f in range(n_splits):
        mask = np.array([group_to_fold[int(g)] == f for g in groups])
        folds.append(np.where(mask)[0])
    return folds


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,   # kept for API compatibility; unused in K-fold mode
    val_size: float = 0.15,    # kept for API compatibility; unused
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into ``N_SPLITS`` (train, val, test) folds.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
        df:           Optional full DataFrame (same row order as ``y``).
                      When provided, enables context-aware grouping.
        test_size:    Unused (kept for API compatibility with the skeleton).
        val_size:     Unused (kept for API compatibility with the skeleton).
        random_state: Random seed for reproducibility.

    Returns:
        A list of ``N_SPLITS`` ``(idx_train, idx_val, idx_test)`` tuples
        of integer index arrays.  Each tuple's ``idx_val`` is a small
        slice carved out of the non-test portion for threshold tuning.
    """
    n = len(y)
    idx = np.arange(n)
    y = np.asarray(y).astype(int)

    if df is not None and "prompt" in df.columns:
        groups = _context_groups(df)
        test_folds = _stratified_group_kfold(
            y, groups, n_splits=N_SPLITS, random_state=random_state
        )
    else:
        groups = None
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
        test_folds = [te for _, te in skf.split(idx, y)]

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold_i, idx_test in enumerate(test_folds):
        idx_pool = np.setdiff1d(idx, idx_test, assume_unique=False)
        y_pool = y[idx_pool]

        # Carve a small validation slice from the pool.
        if groups is not None:
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=VAL_FRAC,
                random_state=random_state + fold_i,
            )
            tr_rel, va_rel = next(gss.split(idx_pool, y_pool, groups=groups[idx_pool]))
        else:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=VAL_FRAC,
                random_state=random_state + fold_i,
            )
            tr_rel, va_rel = next(sss.split(idx_pool, y_pool))

        idx_train = idx_pool[tr_rel]
        idx_val = idx_pool[va_rel]
        splits.append((idx_train, idx_val, idx_test))

    return splits
