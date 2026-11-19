"""Small text utilities: sentence splitting, name normalisation, keyword stems."""
from __future__ import annotations

import re

from aikit import tokenize_words

_ABBREVIATIONS = frozenset(
    {"inc", "corp", "ltd", "co", "dr", "mr", "mrs", "ms", "st", "jr", "sr", "vs", "no", "u.s", "e.g", "i.e"}
)
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")

_ORG_SUFFIXES = frozenset({"inc", "corp", "corporation", "ltd", "llc", "co", "plc", "gmbh", "limited"})

STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been being it its this that these those with as by at "
    "from what which who whom whose how do does did can could should would will i you we they he she his her "
    "their them our us me my where when why there here into than then so such not no yes after before about "
    "has have had also".split()
)


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, without breaking after common abbreviations."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENT_BOUNDARY.split(text)
    sentences: list[str] = []
    for part in parts:
        if sentences:
            prev = sentences[-1]
            last_word = re.findall(r"([A-Za-z.]+)\.$", prev)
            if last_word and last_word[0].lower().rstrip(".") in _ABBREVIATIONS:
                sentences[-1] = prev + " " + part
                continue
        sentences.append(part)
    return [s.strip() for s in sentences if s.strip()]


def normalize_name(name: str) -> str:
    """Canonical lookup key: lowercase, no punctuation, no leading 'the', no corporate suffix."""
    s = name.lower().replace("&", " and ")
    s = re.sub(r"'s\b", "", s)
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    toks = s.split()
    if toks and toks[0] == "the":
        toks = toks[1:]
    while len(toks) > 1 and toks[-1] in _ORG_SUFFIXES:
        toks = toks[:-1]
    return " ".join(toks)


def slugify(name: str) -> str:
    """Stable identifier for an entity name."""
    return re.sub(r"[^a-z0-9]+", "_", normalize_name(name)).strip("_") or "entity"


def keyword_stems(text: str, prefix: int = 5) -> set[str]:
    """Content-word stems (fixed-length prefixes) — crude but robust to inflection.

    'acquired', 'acquisition', 'acquires' all map to 'acqui'.
    """
    return {w[:prefix] for w in tokenize_words(text) if w not in STOPWORDS and len(w) > 2}
