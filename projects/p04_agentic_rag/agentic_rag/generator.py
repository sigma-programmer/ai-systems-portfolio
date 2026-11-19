"""Answer generation: every output sentence is a `Claim` carrying [doc_id] citations.

The offline `ExtractiveGenerator` selects, per information need, the source
sentence(s) with the best IDF-weighted coverage of that need (with small
answer-type bonuses: a number for "how many/when" questions, a name for "who"
questions). Because it copies sentences verbatim, it cannot hallucinate; it
is also used as the constrained *repair* generator when a free-form generator
produced unsupported claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from aikit import LLM

from .grader import Grade, TermStats
from .retriever import RetrievedPassage
from .text import (entities_in, expects_number, expects_person, expects_place, numbers_in,
                   split_sentences)

Evidence = dict[str, list[tuple[RetrievedPassage, Grade]]]


@dataclass
class Claim:
    """One answer sentence and the documents it cites."""

    text: str
    citations: list[str]
    need: str | None = None
    # Ground-truth bookkeeping for adversarial evaluation (never read by the pipeline).
    injected: bool = False
    mutation: str | None = None
    meta: dict = field(default_factory=dict)

    def render(self) -> str:
        cites = "".join(f"[{c}]" for c in self.citations)
        return f"{self.text.rstrip('.')}. {cites}".strip()


class AnswerGenerator(Protocol):
    def generate(self, question: str, evidence: Evidence) -> list[Claim]: ...


class ExtractiveGenerator:
    """Pick the best-supported source sentences for each need."""

    def __init__(self, stats: TermStats, max_passages: int = 2, second_ratio: float = 0.9):
        self.stats = stats
        self.max_passages = max_passages
        self.second_ratio = second_ratio

    def _score(self, need: str, sentence: str, passage_rel: float) -> float:
        cov, _ = self.stats.coverage(need, sentence)
        bonus = 0.0
        if expects_number(need) and numbers_in(sentence):
            bonus += 0.15
        if (expects_person(need) or expects_place(need)) and entities_in("X " + sentence):
            bonus += 0.15
        return cov + bonus + 0.05 * passage_rel

    def generate(self, question: str, evidence: Evidence) -> list[Claim]:
        claims: list[Claim] = []
        seen: set[str] = set()
        for need, graded in evidence.items():
            cands: list[tuple[float, str, str]] = []
            for passage, grade in graded[: self.max_passages]:
                for sent in split_sentences(passage.text):
                    cands.append((self._score(need, sent, grade.relevance), sent, passage.doc_id))
            if not cands:
                continue
            cands.sort(key=lambda c: -c[0])
            best = cands[0]
            picks = [best]
            if len(cands) > 1 and cands[1][0] >= self.second_ratio * best[0]:
                picks.append(cands[1])
            for _, sent, doc_id in picks:
                if sent in seen:
                    continue
                seen.add(sent)
                claims.append(Claim(sent, [doc_id], need))
        return claims


_CITE_RE = re.compile(r"\[([A-Za-z0-9_\-]+)\]")


def parse_cited_sentences(text: str) -> list[Claim]:
    """Parse LLM output of the form 'Sentence one [doc-a]. Sentence two [doc-b][doc-c].'"""
    claims: list[Claim] = []
    for line in text.splitlines():
        for chunk in re.split(r"(?<=\])\s*(?:\.\s*)?(?=[A-Z])|(?<=[.!?])\s+(?=[A-Z])", line):
            cites = _CITE_RE.findall(chunk)
            body = _CITE_RE.sub("", chunk).strip(" .")
            if body:
                claims.append(Claim(body + ".", list(dict.fromkeys(cites))))
    return claims


class LLMGenerator:
    """Free-form LLM generator instructed to cite every sentence."""

    SYSTEM = ("Answer using ONLY the passages. Every sentence must end with the id of the "
              "passage that supports it in square brackets, e.g. 'X is Y [doc-1].' If the "
              "passages do not contain the answer, say so.")

    def __init__(self, llm: LLM):
        self.llm = llm

    def generate(self, question: str, evidence: Evidence) -> list[Claim]:
        seen: dict[str, RetrievedPassage] = {}
        for graded in evidence.values():
            for p, _ in graded:
                seen.setdefault(p.doc_id, p)
        ctx = "\n".join(f"[{p.doc_id}] {p.text}" for p in seen.values())
        prompt = f"GENERATE ANSWER\nQuestion: {question}\nPassages:\n{ctx}"
        return parse_cited_sentences(self.llm.complete(prompt, system=self.SYSTEM).text)
