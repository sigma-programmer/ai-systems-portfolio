"""Query rewriting: decomposition, synonym expansion and keyword extraction.

The controller calls `rewrite` only for information needs that retrieval has
not yet satisfied. Strategies escalate with the attempt number:

1. decompose multi-part questions into self-contained sub-questions
   (pronouns resolved to the question's named entities); if the need is
   atomic, expand it with thesaurus synonyms instead;
2. keyword extraction: keep the highest-IDF content terms plus synonyms, which
   drops chatty words that drag BM25 toward generic documents.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from aikit import LLM

from .grader import TermStats
from .text import STOPWORDS, content_terms, entities_in, synonyms_of

_WH = r"(?:how|what|who|whom|when|where|which|why|is|are|was|were|does|do|did|can|could)"
_SPLIT_RE = re.compile(rf"\s*(?:\?\s+|;\s*|,?\s+and\s+(?={_WH}\b)|,?\s+also\s+(?={_WH}\b))", re.I)
_PRONOUN_RE = re.compile(r"\b(it|its|they|their|them|he|she|his|her)\b", re.I)
_CAP_WORD = re.compile(r"\b[A-Z][a-zA-Z0-9-]+\b")


@dataclass
class RewritePlan:
    """New needs replacing the unmet ones, plus the search queries to issue."""

    needs: list[str]
    queries: dict[str, list[str]] = field(default_factory=dict)
    strategy: str = "none"


class QueryRewriter(Protocol):
    def rewrite(self, question: str, unmet: list[str], attempt: int) -> RewritePlan: ...


def decompose(question: str) -> list[str]:
    """Split a multi-part question into self-contained sub-questions."""
    parts = [p.strip(" ?.,") for p in _SPLIT_RE.split(question) if p and p.strip(" ?.,")]
    if len(parts) <= 1:
        return [question.strip()]
    first = parts[0]
    # Named entities from the first clause (non-initial capitalised words) are used to
    # resolve pronouns in later clauses or appended as context when a clause lacks them.
    ents = [m.group() for m in _CAP_WORD.finditer(first)
            if m.start() > 0 and m.group().lower() not in STOPWORDS]
    out = [first + "?"]
    for p in parts[1:]:
        if ents and _PRONOUN_RE.search(p):
            p = _PRONOUN_RE.sub(lambda m: ents[0] + ("'s" if m.group().lower() in
                                                     ("its", "their", "his", "her") else ""), p)
        elif ents and not (entities_in("X " + p) & {e.lower() for e in ents}):
            p = f"{p} {ents[0]}"
        out.append(p[0].upper() + p[1:] + "?")
    return out


def expand_synonyms(query: str) -> str:
    """Append thesaurus synonyms of the query's content terms."""
    extra: list[str] = []
    for t in dict.fromkeys(content_terms(query)):
        extra.extend(sorted(synonyms_of(t)))
    return query if not extra else f"{query} {' '.join(dict.fromkeys(extra))}"


def keywords(query: str, stats: TermStats | None = None, k: int = 4) -> str:
    """Top-k content terms by corpus IDF (or by order if no stats) plus synonyms."""
    terms = list(dict.fromkeys(content_terms(query)))
    if stats is not None:
        terms.sort(key=lambda t: -stats.idf(t))
    top = terms[:k]
    syns = [s for t in top for s in sorted(synonyms_of(t))]
    return " ".join(dict.fromkeys(top + syns))


class HeuristicRewriter:
    """Offline rewriter implementing the escalating strategies above."""

    def __init__(self, stats: TermStats | None = None):
        self.stats = stats

    def rewrite(self, question: str, unmet: list[str], attempt: int) -> RewritePlan:
        if attempt <= 1:
            needs: list[str] = []
            for need in unmet:
                needs.extend(decompose(need))
            if len(needs) > len(unmet):
                return RewritePlan(needs, {n: [n, expand_synonyms(n)] for n in needs}, "decompose")
            return RewritePlan(list(unmet), {n: [expand_synonyms(n)] for n in unmet},
                               "synonym_expansion")
        return RewritePlan(list(unmet), {n: [keywords(n, self.stats)] for n in unmet},
                           "keyword_extraction")


class LLMRewriter:
    """LLM rewriter; expects JSON {"sub_questions": [...], "queries": {sub_q: [..]}}."""

    SYSTEM = ("You rewrite search queries for a retrieval system. Split multi-part questions "
              "into self-contained sub-questions and propose keyword-rich queries for each. "
              "Reply as JSON: {\"sub_questions\": [..], \"queries\": {\"<sub_q>\": [..]}}.")

    def __init__(self, llm: LLM):
        self.llm = llm

    def rewrite(self, question: str, unmet: list[str], attempt: int) -> RewritePlan:
        prompt = (f"REWRITE QUERY (attempt {attempt})\nOriginal question: {question}\n"
                  f"Unanswered parts: {unmet}")
        data = self.llm.complete(prompt, system=self.SYSTEM, json_mode=True).json()
        subs = [str(s) for s in data.get("sub_questions") or unmet]
        raw_q = data.get("queries") or {}
        queries = {s: [str(q) for q in raw_q.get(s, [s])] or [s] for s in subs}
        return RewritePlan(subs, queries, "llm")
