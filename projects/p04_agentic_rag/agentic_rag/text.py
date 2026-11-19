"""Text utilities shared by every stage: tokenizing, stemming, synonyms, numbers.

All matching in the pipeline (BM25, relevance grading, entailment) goes through
the same `content_terms` normaliser so that a term counted as "covered" at
retrieval time is counted the same way at verification time.
"""
from __future__ import annotations

import re

from aikit import tokenize_words

STOPWORDS = frozenset(
    """a an the of to in on for and or is are was were be been being it its this that these those
    with as by at from what which who whom whose how when where why do does did done can could should
    would will shall may might must i you we they he she me my your our their them his her us
    has have had having get gets got there here than then so such into onto about over under also
    any all some each per much many long often quickly soon tell please know just very more most own same
    if but not no nor too only out up down off again further once s t""".split()
)

# Question-word phrases that indicate the expected answer type.
NUMERIC_CUES = ("how many", "how much", "how long", "how tall", "how far", "how quickly",
                "how fast", "when", "what year", "how old", "what percent")
PERSON_CUES = ("who",)
PLACE_CUES = ("where",)

NEGATIONS = frozenset({"not", "no", "never", "none", "cannot", "neither", "nor", "without"})
_NEG_CONTRACTION = re.compile(r"\b\w+n't\b", re.I)

# Small domain thesaurus used for query expansion and for coverage matching.
# Keys and values are surface words; they are stemmed on load.
_SYNONYM_GROUPS: list[set[str]] = [
    {"vacation", "pto", "holiday", "leave", "time off"},
    {"ceo", "chief executive", "chief executive officer", "head"},
    {"cost", "price", "retail", "dollar", "pay"},
    {"salary", "compensation", "wage"},
    {"tall", "height", "high", "highest", "elevation"},
    {"life", "last", "duration", "lifetime"},
    {"weigh", "weight", "gram", "heavy"},
    {"found", "founding", "establish", "start"},
    {"headquarter", "base", "office", "located"},
    {"remote", "remotely", "home"},
    {"return", "refund", "money back"},
    {"create", "invent", "design", "author"},
    {"raise", "funding", "round", "invest"},
    {"bug bounty", "vulnerability reward"},
    {"waterproof", "water resistant", "rain"},
    {"employee", "staff", "worker"},
    {"available", "open"},
    {"caffeine", "stimulant"},
    {"moon", "satellite"},
]

_NUM_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w])")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_CAP_RE = re.compile(r"\b[A-Z][a-zA-Z0-9-]+\b")


def stem(word: str) -> str:
    """Tiny suffix-stripping stemmer (consistent, not linguistically perfect)."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    for suf in ("ing", "ed", "es", "s", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and not w.endswith("ss"):
            return w[: -len(suf)]
    return w


def content_terms(text: str) -> list[str]:
    """Stemmed, stopword-free word tokens (numbers kept verbatim)."""
    return [stem(w) for w in tokenize_words(text) if w not in STOPWORDS]


def _stem_phrase(phrase: str) -> str:
    return " ".join(stem(w) for w in tokenize_words(phrase))


SYNONYMS: dict[str, frozenset[str]] = {}
for _group in _SYNONYM_GROUPS:
    _stemmed = {_stem_phrase(p) for p in _group}
    for _p in _stemmed:
        SYNONYMS[_p] = frozenset(_stemmed - {_p})


def synonyms_of(term: str) -> frozenset[str]:
    """Stemmed synonym phrases for a stemmed term (may contain spaces)."""
    return SYNONYMS.get(term, frozenset())


SYNONYM_CREDIT = 0.8


def term_match(term: str, text_terms: set[str], text_stem_joined: str) -> float:
    """1.0 for an exact (stemmed) match, SYNONYM_CREDIT for a synonym match, else 0."""
    if term in text_terms:
        return 1.0
    for syn in synonyms_of(term):
        if " " in syn:
            if f" {syn} " in f" {text_stem_joined} ":
                return SYNONYM_CREDIT
        elif syn in text_terms:
            return SYNONYM_CREDIT
    return 0.0


def term_present(term: str, text_terms: set[str], text_stem_joined: str) -> bool:
    """True if `term` or any synonym occurs in a text (given its term set / joined stems)."""
    return term_match(term, text_terms, text_stem_joined) > 0.0


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences on terminal punctuation."""
    parts = [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]
    return parts


def numbers_in(text: str) -> set[str]:
    """Normalised numbers (thousands separators removed) appearing in text."""
    return {m.group().replace(",", "") for m in _NUM_RE.finditer(text)}


def is_negated(text: str) -> bool:
    """Whether a sentence contains an explicit negation cue."""
    words = set(tokenize_words(text))
    return bool(words & NEGATIONS) or bool(_NEG_CONTRACTION.search(text))


def entities_in(text: str) -> set[str]:
    """Capitalised tokens that are not sentence-initial (cheap proper-noun proxy)."""
    ents: set[str] = set()
    for sent in split_sentences(text) or [text]:
        for i, m in enumerate(_CAP_RE.finditer(sent)):
            if m.start() == 0:
                continue
            ents.add(m.group().lower())
    return ents


def expects_number(question: str) -> bool:
    q = question.lower()
    return any(cue in q for cue in NUMERIC_CUES)


def expects_person(question: str) -> bool:
    return question.lower().strip().startswith(PERSON_CUES)


def expects_place(question: str) -> bool:
    return question.lower().strip().startswith(PLACE_CUES)
