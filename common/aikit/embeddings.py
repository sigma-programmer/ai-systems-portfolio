"""Offline dense embeddings via signed feature hashing.

Features: word unigrams, word bigrams, char 3/4-grams of each word, plus
a tiny built-in synonym map so semantically-close words share features.
Not a neural model, but stable, fast, and good enough to make dense vs.
sparse retrieval behave *differently* (char n-grams give fuzzy matching
that BM25 lacks). Swap in any object with `.embed(list[str]) -> ndarray`.
"""
from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np

from .tokens import tokenize_words

_STOP = frozenset(
    "a an the of to in on for and or is are was were be been it this that with as by at from "
    "what which who how do does did can could should would will i you we they he she its".split()
)

# Small concept map: words in the same group share a hashed concept feature.
_CONCEPTS = [
    {"car", "cars", "automobile", "vehicle", "vehicles", "auto"},
    {"buy", "purchase", "purchasing", "acquire", "bought"},
    {"error", "errors", "bug", "bugs", "failure", "fault", "exception", "crash"},
    {"fast", "quick", "speed", "latency", "rapid", "performance"},
    {"cost", "price", "pricing", "expense", "cheap", "expensive", "spend"},
    {"doctor", "physician", "medical", "clinician"},
    {"refund", "reimburse", "reimbursement", "return", "money back"},
    {"login", "signin", "authenticate", "authentication", "password", "credentials"},
    {"delete", "remove", "erase", "drop"},
    {"start", "begin", "launch", "boot", "initialize"},
    {"ceo", "chief executive", "founder", "leader"},
    {"company", "firm", "corporation", "organization", "startup"},
]
_CONCEPT_OF: dict[str, int] = {w: i for i, g in enumerate(_CONCEPTS) for w in g}


def _h(feature: str, dim: int) -> tuple[int, float]:
    d = hashlib.blake2b(feature.encode(), digest_size=8).digest()
    v = int.from_bytes(d, "little")
    return v % dim, (1.0 if (v >> 63) & 1 else -1.0)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self._cache: dict[str, np.ndarray] = {}

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = [w for w in tokenize_words(text) if w not in _STOP]
        feats: list[tuple[str, float]] = []
        for w in words:
            feats.append(("w:" + w, 1.0))
            if w in _CONCEPT_OF:
                feats.append((f"c:{_CONCEPT_OF[w]}", 1.5))
            padded = f"<{w}>"
            for n in (3, 4):
                for i in range(len(padded) - n + 1):
                    feats.append((f"g{n}:" + padded[i : i + n], 0.25))
        for a, b in zip(words, words[1:]):
            feats.append((f"b:{a}_{b}", 0.75))
        return feats

    def embed_one(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]
        v = np.zeros(self.dim, dtype=np.float32)
        for f, w in self._features(text):
            idx, sign = _h(f, self.dim)
            v[idx] += sign * w
        n = float(np.linalg.norm(v))
        if n > 0:
            v /= n
        self._cache[text] = v
        return v

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed_one(t) for t in texts])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray | float:
    """Cosine similarity; vectors or (n,d)x(d,) / (n,d)x(m,d)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    def _norm(x):
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    if a.ndim == 1 and b.ndim == 1:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na and nb else 0.0
    return _norm(np.atleast_2d(a)) @ _norm(np.atleast_2d(b)).T
