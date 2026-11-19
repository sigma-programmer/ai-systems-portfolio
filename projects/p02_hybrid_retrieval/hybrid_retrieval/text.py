"""Text analysis helpers shared by BM25 and the reranker features."""
from __future__ import annotations

from functools import lru_cache

from aikit import tokenize_words

STOPWORDS = frozenset(
    """a an the of to in on for and or is are was were be been being it its this that these
    those with as by at from what which who whom how do does did can could should would will
    shall may might must i me my you your we our they their he she him her them not no but if
    then so than too very just about into over after before up down out off again there here
    when where why all any both each few more most other some such only own same s t don
    am has have had having get got""".split()
)


def analyze(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords. No stemming on purpose:
    the lexical/semantic gap is exactly what hybrid retrieval exists to cover."""
    return [t for t in tokenize_words(text) if t not in STOPWORDS]


def bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


@lru_cache(maxsize=200_000)
def char_ngrams(word: str, n: int = 3) -> frozenset[str]:
    """Character n-grams of a single word with boundary markers."""
    padded = f"<{word}>"
    if len(padded) <= n:
        return frozenset({padded})
    return frozenset(padded[i : i + n] for i in range(len(padded) - n + 1))


def jaccard(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)
