"""Query-document interaction features for the cross-encoder-style reranker.

A true cross-encoder reads query and document *jointly*. Offline, we
approximate that with features computed over the (query, doc) pair rather
than over each side independently: lexical coverage, IDF-weighted coverage,
bigram/phrase matches, term proximity, fuzzy (typo-tolerant) matches, title
matches (best char-trigram similarity, so "Veltrra" still matches
"veltra"), plus the first-stage BM25 and dense scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np

from .bm25 import BM25Index
from .text import analyze, bigrams, char_ngrams, jaccard

FEATURE_NAMES: tuple[str, ...] = (
    "bm25_log",
    "bm25_rel",
    "dense_cos",
    "dense_rel",
    "term_coverage",
    "idf_coverage",
    "bigram_match",
    "proximity",
    "fuzzy_coverage",
    "title_coverage",
    "title_fuzzy",
)


@dataclass
class _DocView:
    tokens: list[str]
    token_set: frozenset[str]
    positions: dict[str, list[int]]
    bigram_set: set[tuple[str, str]]
    title_set: frozenset[str]


def min_window(positions: Sequence[Sequence[int]]) -> int:
    """Length of the smallest token window containing one position from every list."""
    events = sorted((p, t) for t, plist in enumerate(positions) for p in plist)
    need = len(positions)
    counts: dict[int, int] = {}
    best = 10**9
    left = 0
    for right, (pos, t) in enumerate(events):
        counts[t] = counts.get(t, 0) + 1
        while len(counts) == need:
            lpos, lt = events[left]
            best = min(best, pos - lpos + 1)
            counts[lt] -= 1
            if counts[lt] == 0:
                del counts[lt]
            left += 1
    return best


@dataclass
class FeatureExtractor:
    """Computes the FEATURE_NAMES vector for every (query, candidate) pair."""

    bm25: BM25Index
    titles: Sequence[str]
    texts: Sequence[str]
    vectors: np.ndarray
    fuzzy_floor: float = 0.2
    _views: dict[int, _DocView] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._vocab = sorted(self.bm25.df)
        self._fuzzy = lru_cache(maxsize=50_000)(self._fuzzy_uncached)

    def _view(self, i: int) -> _DocView:
        v = self._views.get(i)
        if v is None:
            toks = analyze(self.texts[i])
            pos: dict[str, list[int]] = {}
            for p, t in enumerate(toks):
                pos.setdefault(t, []).append(p)
            v = _DocView(toks, frozenset(toks), pos, bigrams(toks), frozenset(analyze(self.titles[i])))
            self._views[i] = v
        return v

    def _fuzzy_uncached(self, term: str) -> dict[str, float]:
        """Vocabulary terms similar to `term` -> char-trigram Jaccard (exact match = 1.0).

        Precomputed once per query term against the whole vocabulary, so
        per-candidate fuzzy matching is just dictionary lookups.
        """
        grams = char_ngrams(term)
        out = {term: 1.0}
        for w in self._vocab:
            if w != term and abs(len(w) - len(term)) <= 3:
                sim = jaccard(grams, char_ngrams(w))
                if sim >= self.fuzzy_floor:
                    out[w] = sim
        return out

    @staticmethod
    def _best(sims: dict[str, float], tokens: frozenset[str]) -> float:
        return max((sims.get(t, 0.0) for t in tokens), default=0.0)

    def extract(self, query: str, qvec: np.ndarray, candidates: Sequence[int]) -> np.ndarray:
        n = len(candidates)
        X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
        if n == 0:
            return X
        q_terms = list(dict.fromkeys(analyze(query)))
        q_bigrams = bigrams(analyze(query))
        idf = {t: self.bm25.idf(t) for t in q_terms}
        idf_total = sum(idf.values()) or 1.0
        fuzzy_sets = {t: self._fuzzy(t) for t in q_terms}
        cand = np.asarray(candidates, dtype=np.int64)
        bm = self.bm25.score_all(query)[cand].astype(np.float64)
        cos = (self.vectors[cand] @ qvec.astype(np.float32)).astype(np.float64)
        X[:, 0] = np.log1p(bm)
        X[:, 1] = bm / bm.max() if bm.max() > 0 else 0.0
        X[:, 2] = cos
        X[:, 3] = cos - cos.max()
        nq = max(len(q_terms), 1)
        for row, i in enumerate(cand):
            v = self._view(int(i))
            matched = [t for t in q_terms if t in v.token_set]
            X[row, 4] = len(matched) / nq
            X[row, 5] = sum(idf[t] for t in matched) / idf_total
            X[row, 6] = len(q_bigrams & v.bigram_set) / len(q_bigrams) if q_bigrams else 0.0
            if len(matched) >= 2:
                X[row, 7] = len(matched) / min_window([v.positions[t] for t in matched])
            X[row, 8] = sum(self._best(fuzzy_sets[t], v.token_set) for t in q_terms) / nq
            X[row, 9] = sum(1 for t in q_terms if t in v.title_set) / nq
            X[row, 10] = sum(self._best(fuzzy_sets[t], v.title_set) for t in q_terms) / nq
        return X
