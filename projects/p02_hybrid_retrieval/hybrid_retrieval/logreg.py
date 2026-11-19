"""L2-regularized logistic regression in numpy (Newton / IRLS).

scikit-learn is not available in this environment, so this is a small,
exact replacement: standardize features, then Newton's method on the
penalized log-likelihood. With ~10 features it converges in < 15 steps
and is fully deterministic.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class LogisticRegression:
    """Binary classifier: p(y=1|x) = sigmoid(w . standardize(x) + b)."""

    def __init__(self, l2: float = 1.0, max_iter: int = 50, tol: float = 1e-8, balanced: bool = True) -> None:
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.balanced = balanced
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be (n, d) and y must have length n")
        if len(np.unique(y)) < 2:
            raise ValueError("need both positive and negative examples")
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        Z = np.hstack([(X - self.mean_) / self.scale_, np.ones((len(X), 1))])
        sw = np.ones(len(y))
        if self.balanced:
            pos = y.sum()
            neg = len(y) - pos
            sw = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
        d = Z.shape[1]
        reg = np.full(d, self.l2)
        reg[-1] = 0.0  # do not penalize the intercept
        w = np.zeros(d)
        for it in range(1, self.max_iter + 1):
            p = _sigmoid(Z @ w)
            grad = Z.T @ (sw * (p - y)) + reg * w
            hess = (Z * (sw * p * (1 - p))[:, None]).T @ Z + np.diag(reg + 1e-9)
            step = np.linalg.solve(hess, grad)
            w -= step
            self.n_iter_ = it
            if np.max(np.abs(step)) < self.tol:
                break
        self.coef_ = w[:-1]
        self.intercept_ = float(w[-1])
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        Z = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        return Z @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def raw_weights(self) -> tuple[np.ndarray, float]:
        """Weights in the original (unstandardized) feature space."""
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        w = self.coef_ / self.scale_
        return w, float(self.intercept_ - (self.mean_ / self.scale_) @ self.coef_)
