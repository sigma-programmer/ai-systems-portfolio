"""Stratified resampling statistics for suite comparison.

Runs are stratified by task: an agent may be much better at one task than
another, so naive pooling would let a change in the *task mix* masquerade
as a regression. The statistic is therefore the stratum-weighted mean
difference  sum_k w_k * (mean_cand_k - mean_base_k)  with weights w_k taken
from the baseline's task mix, and both the bootstrap and the permutation
test resample only within each task.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class DiffResult:
    delta: float
    ci_low: float
    ci_high: float
    p_one_sided: float  # P(delta at least this bad under H0), direction-aware
    p_two_sided: float


def _group(values: np.ndarray, strata: np.ndarray) -> dict[str, np.ndarray]:
    return {k: values[strata == k] for k in np.unique(strata)}


def stratified_diff(
    base: np.ndarray,
    base_strata: np.ndarray,
    cand: np.ndarray,
    cand_strata: np.ndarray,
    higher_is_better: bool = True,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> DiffResult:
    """Mean difference with stratified bootstrap CI and stratified permutation p-values.

    Only strata present in both samples are used.
    """
    base, cand = np.asarray(base, float), np.asarray(cand, float)
    bg = _group(base, np.asarray(base_strata))
    cg = _group(cand, np.asarray(cand_strata))
    keys = sorted(set(bg) & set(cg))
    if not keys:
        raise ValueError("no common strata between baseline and candidate")
    nb = sum(len(bg[k]) for k in keys)
    w = {k: len(bg[k]) / nb for k in keys}
    obs = float(sum(w[k] * (cg[k].mean() - bg[k].mean()) for k in keys))

    rng = np.random.default_rng(seed)
    B = n_resamples
    boot_d = np.zeros(B)
    perm_d = np.zeros(B)
    for k in keys:
        b, c = bg[k], cg[k]
        boot_d += w[k] * (c[rng.integers(0, len(c), (B, len(c)))].mean(axis=1)
                          - b[rng.integers(0, len(b), (B, len(b)))].mean(axis=1))
        pooled = np.concatenate([b, c])
        shuffled = rng.permuted(np.tile(pooled, (B, 1)), axis=1)
        perm_d += w[k] * (shuffled[:, len(b):].mean(axis=1) - shuffled[:, : len(b)].mean(axis=1))
    alpha = 1 - confidence
    lo, hi = np.quantile(boot_d, [alpha / 2, 1 - alpha / 2])
    eps = 1e-12
    if higher_is_better:
        p1 = (np.sum(perm_d <= obs + eps) + 1) / (B + 1)
    else:
        p1 = (np.sum(perm_d >= obs - eps) + 1) / (B + 1)
    p2 = (np.sum(np.abs(perm_d) >= abs(obs) - eps) + 1) / (B + 1)
    return DiffResult(delta=obs, ci_low=float(lo), ci_high=float(hi),
                      p_one_sided=float(min(1.0, p1)), p_two_sided=float(min(1.0, p2)))


Direction = Literal["higher", "lower"]
