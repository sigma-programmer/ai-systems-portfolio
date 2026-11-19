"""Judge calibration against human labels.

Implements, in pure numpy (scikit-learn is not available in this env):

* agreement metrics: accuracy, Cohen's kappa, Brier score, ECE
* Platt scaling: 1-D logistic regression fitted by Newton/IRLS with a tiny
  L2 penalty
* isotonic regression: pool-adjacent-violators
* decision-threshold tuning to maximise kappa

``calibrate_judge`` reports *cross-validated* post-calibration agreement so
the improvement is not an artefact of fitting and scoring on the same data.
"""
from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel


def accuracy(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    return float(np.mean(a == b)) if len(a) else 0.0


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary/categorical label vectors."""
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    if len(a) == 0:
        return 0.0
    labels = np.union1d(a, b)
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == k) * np.mean(b == k) for k in labels))
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


class Calibrator(Protocol):
    def fit(self, s: np.ndarray, y: np.ndarray) -> "Calibrator": ...
    def predict(self, s: np.ndarray) -> np.ndarray: ...


class PlattCalibrator:
    """p = sigmoid(a*s + b), fitted by Newton's method on the log-loss."""

    def __init__(self, l2: float = 1e-3, iters: int = 100):
        self.l2, self.iters = l2, iters
        self.a, self.b = 1.0, 0.0

    def fit(self, s: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        s, y = np.asarray(s, float), np.asarray(y, float)
        # Platt's smoothed targets avoid infinite weights on separable data.
        n1, n0 = y.sum(), len(y) - y.sum()
        t = np.where(y > 0.5, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))
        X = np.column_stack([s, np.ones_like(s)])
        w = np.zeros(2)
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-(X @ w)))
            g = X.T @ (p - t) + self.l2 * w
            H = X.T @ (X * (p * (1 - p))[:, None]) + self.l2 * np.eye(2)
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-10:
                break
        self.a, self.b = float(w[0]), float(w[1])
        return self

    def predict(self, s: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.a * np.asarray(s, float) + self.b)))


class IsotonicCalibrator:
    """Monotone non-decreasing step function fitted by pool-adjacent-violators."""

    def __init__(self) -> None:
        self.x_: np.ndarray = np.array([0.0, 1.0])
        self.y_: np.ndarray = np.array([0.0, 1.0])

    def fit(self, s: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        s, y = np.asarray(s, float), np.asarray(y, float)
        order = np.argsort(s, kind="mergesort")
        xs, ys = s[order], y[order]
        # Blocks: [sum_y, weight, x_min, x_max]
        blocks: list[list[float]] = []
        for x, v in zip(xs, ys):
            blocks.append([v, 1.0, x, x])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                v2, w2, _, hi = blocks.pop()
                blocks[-1][0] += v2
                blocks[-1][1] += w2
                blocks[-1][3] = hi
        xk, yk = [], []
        for v, w, lo, hi in blocks:
            xk += [lo, hi]
            yk += [v / w, v / w]
        self.x_, self.y_ = np.array(xk), np.array(yk)
        return self

    def predict(self, s: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(s, float), self.x_, self.y_)


def tune_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """Threshold on scores/probabilities that maximises Cohen's kappa."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    cands = np.unique(np.concatenate([p, [0.5]]))
    best_t, best_k = 0.5, -2.0
    for t in cands:
        k = cohen_kappa((p >= t).astype(int), y)
        if k > best_k + 1e-12 or (abs(k - best_k) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_k = float(t), k
    return best_t


class Agreement(BaseModel):
    accuracy: float
    kappa: float
    brier: float
    ece: float
    threshold: float


class CalibrationReport(BaseModel):
    method: str
    n: int
    positive_rate: float
    pre: Agreement
    post: Agreement
    folds: int
    final_params: dict[str, float]


def _agreement(p: np.ndarray, y: np.ndarray, t: float) -> Agreement:
    pred = (p >= t).astype(int)
    return Agreement(accuracy=accuracy(pred, y), kappa=cohen_kappa(pred, y),
                     brier=brier(p, y), ece=ece(p, y), threshold=t)


def make_calibrator(method: Literal["platt", "isotonic"]) -> Calibrator:
    return PlattCalibrator() if method == "platt" else IsotonicCalibrator()


def calibrate_judge(
    scores: np.ndarray,
    labels: np.ndarray,
    method: Literal["platt", "isotonic"] = "platt",
    folds: int = 5,
    seed: int = 0,
    tune: bool = True,
) -> tuple[CalibrationReport, Calibrator, float]:
    """Measure judge/human agreement before and after calibration.

    Pre-calibration uses the judge's raw score with the naive 0.5 cut.
    Post-calibration is k-fold cross-validated: the calibrator (and, if
    ``tune``, the kappa-optimal threshold) is fitted on k-1 folds and scored on
    the held-out fold. Returns (report, calibrator fitted on all data, threshold).
    """
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if len(s) != len(y) or len(s) < folds:
        raise ValueError("need at least `folds` labelled examples with matching lengths")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(s))
    fold_of = np.empty(len(s), int)
    fold_of[perm] = np.arange(len(s)) % folds
    cv_p = np.zeros(len(s))
    cv_pred = np.zeros(len(s), int)
    for f in range(folds):
        tr, te = fold_of != f, fold_of == f
        cal = make_calibrator(method).fit(s[tr], y[tr])
        t = tune_threshold(cal.predict(s[tr]), y[tr]) if tune else 0.5
        cv_p[te] = cal.predict(s[te])
        cv_pred[te] = (cv_p[te] >= t).astype(int)
    final = make_calibrator(method).fit(s, y)
    final_t = tune_threshold(final.predict(s), y) if tune else 0.5
    post = Agreement(accuracy=accuracy(cv_pred, y), kappa=cohen_kappa(cv_pred, y),
                     brier=brier(cv_p, y), ece=ece(cv_p, y), threshold=final_t)
    params = {"a": final.a, "b": final.b} if isinstance(final, PlattCalibrator) else {"knots": float(len(final.x_))}
    report = CalibrationReport(method=method, n=len(s), positive_rate=float(y.mean()),
                               pre=_agreement(s, y, 0.5), post=post, folds=folds, final_params=params)
    return report, final, final_t
