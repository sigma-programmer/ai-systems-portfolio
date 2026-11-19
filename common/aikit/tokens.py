"""Deterministic token approximation.

Real tokenizers are model-specific; for budgeting we need a fast, stable,
slightly *conservative* estimate. We count word/punctuation pieces and
long words as multiple sub-tokens (~4 chars each), which tracks BPE counts
within ~10-15% on English prose and over-estimates rather than under.
"""
from __future__ import annotations

import math
import re

_PIECE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WORD = re.compile(r"[a-z0-9]+")


def _piece_cost(piece: str) -> int:
    if len(piece) <= 4:
        return 1
    return math.ceil(len(piece) / 4)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return sum(_piece_cost(p) for p in _PIECE.findall(text))


def truncate_to_tokens(text: str, max_tokens: int, suffix: str = " …") -> str:
    """Truncate on piece boundaries so the result fits in max_tokens (incl. suffix)."""
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    budget = max_tokens - count_tokens(suffix)
    if budget <= 0:
        return ""
    used = 0
    end = 0
    for m in _PIECE.finditer(text):
        c = _piece_cost(m.group())
        if used + c > budget:
            break
        used += c
        end = m.end()
    return text[:end].rstrip() + suffix


def tokenize_words(text: str) -> list[str]:
    """Lowercased alphanumeric word tokens (for BM25, overlap metrics, etc.)."""
    return _WORD.findall(text.lower())
