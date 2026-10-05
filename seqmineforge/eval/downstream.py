"""Downstream utility: sequence classification on mined-pattern features.

The question this answers: *does compressing the pattern set via closure cost
any predictive power?*  A closed-pattern basis is information-preserving in
theory, but the theory says nothing about what a regularised linear model does
with it.  So the claim is checked empirically on a held-out split.

Leakage discipline (the highest-frequency source of wrong numbers in this repo's
whole family):

* Miners run on the **train fold only** -- no pattern may be selected using
  holdout information.
* The label is derived from the sequence, then the split is stratified.
* The classifier and its hyper-parameters are fit on train only.
* Every extractor under comparison sees the *same* train/test split.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.errors import BackendUnavailableError
from ..core.seed import rng
from ..core.types import SequenceDatabase

__all__ = [
    "DatasetSplits",
    "available_sklearn",
    "logistic_accuracy",
    "make_binary_labels",
    "stratified_split",
]


def available_sklearn() -> bool:
    """Whether the Tier-0 logistic-regression backend can be used."""
    try:
        import sklearn.linear_model
        import sklearn.preprocessing  # noqa: F401
    except ImportError:
        return False
    return True


def make_binary_labels(db: SequenceDatabase, *, positive_items: set[int]) -> np.ndarray:
    """Label = 1 iff the sequence contains any item in ``positive_items``.

    Deterministic given the database, no RNG involved.
    """
    if not positive_items:
        raise ValueError("positive_items must be non-empty")
    y = np.zeros(db.n_sequences, dtype=np.int64)
    for i, seq in enumerate(db.sequences):
        if any(int(v) in positive_items for v in seq):
            y[i] = 1
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError(
            "label construction produced a single class -- choose a different "
            f"positive_items set (got {int(y.sum())} positives of {len(y)})"
        )
    return y


def stratified_split(
    y: np.ndarray, *, test_size: float, seed: int, stream: int = 41
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/test index split.

    Stratification matters because a random split on a skewed label would easily
    produce a fold with no positives, silently turning accuracy into a constant.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must lie in (0, 1), got {test_size}")
    gen = rng(seed, stream=stream)
    train: list[int] = []
    test: list[int] = []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        if idx.size == 0:
            continue
        perm = gen.permutation(idx)
        n_test = round(test_size * idx.size)
        n_test = max(1, min(n_test, idx.size - 1)) if idx.size > 1 else 0
        test.extend(int(v) for v in perm[:n_test])
        train.extend(int(v) for v in perm[n_test:])
    train_arr = np.sort(np.asarray(train, dtype=np.int64))
    test_arr = np.sort(np.asarray(test, dtype=np.int64))
    if train_arr.size == 0 or test_arr.size == 0:
        raise ValueError("stratified split produced an empty fold")
    if np.intersect1d(train_arr, test_arr).size:
        raise AssertionError("train/test overlap -- split is broken")
    return train_arr, test_arr


def logistic_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    max_iter: int = 2000,
) -> float:
    """Hold-out accuracy of an L2 logistic regression.

    The scaler is fit on **train only**; fitting it on the full matrix leaks
    holdout mean/variance into the model.  Falls back to a pure-NumPy
    ridge-regularised least-squares classifier when scikit-learn is unavailable,
    so the pipeline stays runnable offline.

    Raises
    ------
    BackendUnavailableError
        Never -- the fallback is always available.  Kept in the signature
        history: an earlier version raised here, which made the demo fail hard
        in an offline environment.  The offline path is now first-class.
    """
    try:
        return _sklearn_accuracy(x_train, y_train, x_test, y_test, seed=seed, max_iter=max_iter)
    except ImportError:
        return _numpy_ridge_accuracy(x_train, y_train, x_test, y_test)


def _sklearn_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    max_iter: int,
) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)  # train-only fit: no leakage
    xt = scaler.transform(x_train)
    xv = scaler.transform(x_test)
    model = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        random_state=seed % (2**31 - 1),
    )
    model.fit(xt, y_train)
    pred = model.predict(xv)
    return float((pred == y_test).mean())


def _numpy_ridge_accuracy(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Offline fallback: standardised ridge regression on a +-1 target.

    Same feature convention as the Tier-0 path (train-only standardisation), so
    the two backends remain comparable rather than measuring different things.
    """
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd < 1e-12] = 1.0
    xt = (x_train - mu) / sd
    xv = (x_test - mu) / sd
    if xt.shape[1] == 0:
        raise BackendUnavailableError("no features to fit (empty pattern set)")
    target = (2.0 * y_train - 1.0).astype(np.float64)
    gram = xt.T @ xt + 1e-6 * np.eye(xt.shape[1])
    coef = np.linalg.solve(gram, xt.T @ target)
    scores = xv @ coef
    pred = (scores > 0).astype(np.int64)
    return float((pred == y_test).mean())


class DatasetSplits:
    """Container keeping train/test *sequences* and labels aligned."""

    __slots__ = ("test_idx", "test_seqs", "train_idx", "train_seqs", "y_test", "y_train")

    def __init__(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        db: SequenceDatabase,
        y: np.ndarray,
    ) -> None:
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.train_seqs = tuple(db.sequences[int(i)] for i in train_idx)
        self.test_seqs = tuple(db.sequences[int(i)] for i in test_idx)
        self.y_train = y[train_idx]
        self.y_test = y[test_idx]
