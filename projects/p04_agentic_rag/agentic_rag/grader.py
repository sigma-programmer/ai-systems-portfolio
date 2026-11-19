"""Relevance grading: does a passage actually contain what an information need asks for?

The heuristic grader computes *IDF-weighted concept coverage*: each content
term of the need (with synonyms) is weighted by how rare its concept is in the
corpus, and we measure what fraction of that weight the passage covers. Rare
terms the corpus has never seen (e.g. "salary" in a handbook with no pay data)
get maximum weight, so a passage that matches only the generic words of a
question ("Lumora", "employees") scores low. That single property is what
makes abstention on near-miss questions work.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from aikit import LLM, tokenize_words

from .retriever import RetrievedPassage
from .text import content_terms, stem, term_match, term_present


@dataclass
class Grade:
    doc_id: str
    relevance: float
    coverage: float
    missing_terms: list[str] = field(default_factory=list)
    relevant: bool = False


class RelevanceGrader(Protocol):
    threshold: float

    def grade(self, need: str, passage: RetrievedPassage) -> Grade: ...


@dataclass
class _TextView:
    terms: set[str]
    joined: str

    @classmethod
    def of(cls, text: str) -> "_TextView":
        return cls(set(content_terms(text)), " ".join(stem(w) for w in tokenize_words(text)))


class TermStats:
    """Concept-level document frequencies over a corpus (synonym-aware)."""

    def __init__(self, texts: Sequence[str]):
        self.views = [_TextView.of(t) for t in texts]
        self.n = len(self.views)
        self._idf: dict[str, float] = {}

    def idf(self, term: str) -> float:
        if term not in self._idf:
            df = sum(1 for v in self.views if term_present(term, v.terms, v.joined))
            self._idf[term] = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
        return self._idf[term]

    def coverage(self, query: str, text: str) -> tuple[float, list[str]]:
        """IDF-weighted fraction of query concepts present in text, plus missing terms."""
        terms = list(dict.fromkeys(content_terms(query)))
        surface: dict[str, str] = {}
        for w in tokenize_words(query):
            surface.setdefault(stem(w), w)
        if not terms:
            return 0.0, []
        view = _TextView.of(text)
        total = got = 0.0
        missing: list[tuple[float, str]] = []
        for t in terms:
            w = self.idf(t)
            total += w
            credit = term_match(t, view.terms, view.joined)
            if credit:
                got += w * credit
            else:
                missing.append((w, t))
        missing.sort(key=lambda x: -x[0])
        return (got / total if total else 0.0), [surface.get(t, t) for _, t in missing]


class HeuristicGrader:
    """Offline grader: relevance = 0.85 * concept coverage + 0.15 * dense similarity."""

    def __init__(self, stats: TermStats, threshold: float = 0.65):
        self.stats = stats
        self.threshold = threshold

    def grade(self, need: str, passage: RetrievedPassage) -> Grade:
        cov, missing = self.stats.coverage(need, passage.title + " " + passage.text)
        rel = 0.85 * cov + 0.15 * max(passage.dense, 0.0)
        return Grade(passage.doc_id, rel, cov, missing, rel >= self.threshold)


class LLMGrader:
    """LLM-backed grader; expects JSON {"score": 0..1, "missing": [..]}."""

    SYSTEM = ("You grade whether a passage contains the information needed to answer a "
              "question. Reply as JSON: {\"score\": <0..1>, \"missing\": [<missing concepts>]}.")

    def __init__(self, llm: LLM, threshold: float = 0.6):
        self.llm = llm
        self.threshold = threshold

    def grade(self, need: str, passage: RetrievedPassage) -> Grade:
        prompt = f"GRADE RELEVANCE\nQuestion: {need}\nPassage [{passage.doc_id}]: {passage.text}"
        data = self.llm.complete(prompt, system=self.SYSTEM, json_mode=True).json()
        score = float(max(0.0, min(1.0, data.get("score", 0.0))))
        missing = [str(m) for m in data.get("missing", [])]
        return Grade(passage.doc_id, score, score, missing, score >= self.threshold)
