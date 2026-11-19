"""Hybrid sparse+dense retriever: Okapi BM25 fused with HashingEmbedder cosine.

Fusion is a convex combination of max-normalised BM25 and raw cosine
similarity. Unlike reciprocal-rank fusion this keeps *absolute* score
magnitudes, which the confidence calibrator later relies on: a query whose
best BM25 hit is weak should look weak, not get rank 1 = score 1.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from aikit import HashingEmbedder

from .corpus import Document
from .text import content_terms


@dataclass(frozen=True)
class RetrievedPassage:
    doc_id: str
    title: str
    text: str
    bm25: float
    dense: float
    score: float


class BM25:
    """Classic Okapi BM25 over stemmed content terms."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.max_idf = math.log(1 + (n + 0.5) / 0.5) if n else 1.0

    def term_idf(self, term: str) -> float:
        """IDF of a term; unseen terms get the maximum possible IDF."""
        return self.idf.get(term, self.max_idf)

    def scores(self, query_terms: Sequence[str]) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float64)
        for i, (tf, dl) in enumerate(zip(self.docs, self.lengths)):
            s = 0.0
            for t in set(query_terms):
                f = tf.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                s += self.idf[t] * f * (self.k1 + 1) / denom
            out[i] = s
        return out


class HybridRetriever:
    """BM25 + dense hybrid retriever over a list of documents (one passage each)."""

    def __init__(self, docs: Sequence[Document], alpha: float = 0.6,
                 embedder: HashingEmbedder | None = None):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.docs = list(docs)
        self.alpha = alpha
        self.embedder = embedder or HashingEmbedder()
        self.doc_terms = [content_terms(d.title + " " + d.text) for d in self.docs]
        self.bm25 = BM25(self.doc_terms)
        self.matrix = self.embedder.embed([d.title + ". " + d.text for d in self.docs])
        self.by_id = {d.doc_id: d for d in self.docs}

    def search(self, query: str, k: int = 4) -> list[RetrievedPassage]:
        """Top-k passages by fused score (empty list for empty corpus/query)."""
        if not self.docs or not query.strip():
            return []
        sparse = self.bm25.scores(content_terms(query))
        top = float(sparse.max()) if sparse.size else 0.0
        sparse_n = sparse / top if top > 0 else sparse
        dense = self.matrix @ self.embedder.embed_one(query)
        fused = self.alpha * sparse_n + (1 - self.alpha) * np.clip(dense, 0.0, 1.0)
        order = np.argsort(-fused, kind="stable")[:k]
        return [
            RetrievedPassage(self.docs[i].doc_id, self.docs[i].title, self.docs[i].text,
                             float(sparse[i]), float(dense[i]), float(fused[i]))
            for i in order
        ]
