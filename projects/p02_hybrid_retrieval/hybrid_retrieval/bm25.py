"""Okapi BM25 over an inverted index, implemented from scratch.

Per-term impact arrays are precomputed at index time (the BM25 term weight
for every posting), so query-time scoring is a handful of numpy scatter-adds.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable, Sequence

import numpy as np

from .text import analyze


def top_k_indices(scores: np.ndarray, k: int, valid: np.ndarray | None = None) -> np.ndarray:
    """Indices of the k largest scores (descending), restricted to `valid` rows.

    Ties are broken by lower index so results are deterministic.
    """
    if valid is not None:
        cand = np.flatnonzero(valid)
    else:
        cand = np.arange(scores.shape[0])
    if cand.size == 0 or k <= 0:
        return np.zeros(0, dtype=np.int64)
    sub = scores[cand]
    if k < cand.size:
        part = np.argpartition(-sub, k - 1)[:k]
        cand, sub = cand[part], sub[part]
    order = np.lexsort((cand, -sub))
    return cand[order]


class BM25Index:
    """Okapi BM25 (Robertson/Sparck Jones idf with +1 smoothing, never negative)."""

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        analyzer: Callable[[str], list[str]] = analyze,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.analyzer = analyzer
        self.n_docs = 0
        self.avgdl = 0.0
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.df: dict[str, int] = {}
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, texts: Sequence[str]) -> "BM25Index":
        tokenized = [self.analyzer(t) for t in texts]
        self.n_docs = len(tokenized)
        self.doc_len = np.array([len(t) for t in tokenized], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(tokenized):
            for term, tf in Counter(toks).items():
                raw[term].append((i, tf))
        self.df = {t: len(p) for t, p in raw.items()}
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        self._postings = {}
        for term, plist in raw.items():
            idx = np.fromiter((d for d, _ in plist), dtype=np.int64, count=len(plist))
            tf = np.fromiter((f for _, f in plist), dtype=np.float32, count=len(plist))
            impact = self.idf(term) * tf * (self.k1 + 1) / (tf + norm[idx])
            self._postings[term] = (idx, impact.astype(np.float32))
        return self

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def score_all(self, query: str) -> np.ndarray:
        """BM25 score of every document (0 for docs sharing no term)."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for term, qtf in Counter(self.analyzer(query)).items():
            post = self._postings.get(term)
            if post is None:
                continue
            idx, impact = post
            scores[idx] += impact * qtf
        return scores

    def search(self, query: str, k: int = 10, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Top-k (doc_index, score) among docs with a positive score (and in `mask`)."""
        scores = self.score_all(query)
        valid = scores > 0
        if mask is not None:
            valid &= mask
        return [(int(i), float(scores[i])) for i in top_k_indices(scores, k, valid)]
