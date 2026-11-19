"""Naive vector RAG baseline: embed chunks, retrieve top-k, read an answer.

The offline reader is a standard extractive heuristic: candidate answers are
entities of the expected type (who -> PERSON, where -> LOCATION, ...) found in
the retrieved chunks, excluding entities named in the question, scored by
retrieval rank and by keyword overlap of the sentence they appear in. It is a
fair stand-in for "retrieve then read" — and, like real vector RAG, it can
only reason over what the top-k chunks happen to contain.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from pydantic import BaseModel, Field

from aikit import LLM, HashingEmbedder

from .extraction import find_entity_spans
from .models import Chunk, EntityType
from .query import _expected_type
from .text import keyword_stems, normalize_name, split_sentences

READER_SYSTEM = "Answer the question using only the passages provided. Reply with the answer entity only."


class VectorAnswer(BaseModel):
    question: str
    answer: str | None
    candidates: list[str] = Field(default_factory=list)
    retrieved: list[str] = Field(default_factory=list)


class VectorRAG:
    """Brute-force cosine retrieval over chunk embeddings + extractive reader."""

    def __init__(self, chunks: list[Chunk], k: int = 3, embedder: HashingEmbedder | None = None, llm: LLM | None = None):
        self.chunks = chunks
        self.k = k
        self.embedder = embedder or HashingEmbedder()
        self.llm = llm
        self.matrix = self.embedder.embed([c.text for c in chunks])

    def retrieve(self, question: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = self.embedder.embed([question])[0]
        sims = self.matrix @ q
        order = np.argsort(-sims, kind="stable")[: k or self.k]
        return [(self.chunks[i], float(sims[i])) for i in order]

    def answer(self, question: str) -> VectorAnswer:
        hits = self.retrieve(question)
        retrieved = [c.id for c, _ in hits]
        if self.llm is not None:
            passages = "\n\n".join(f"[{c.id}] {c.text}" for c, _ in hits)
            text = self.llm.complete(f"Passages:\n{passages}\n\nQuestion: {question}", system=READER_SYSTEM).text.strip()
            return VectorAnswer(question=question, answer=text, candidates=[text], retrieved=retrieved)

        want = _expected_type(question)
        q_norm = " " + normalize_name(question) + " "
        q_stems = keyword_stems(question)
        scores: dict[str, float] = defaultdict(float)
        for rank, (chunk, _sim) in enumerate(hits):
            for sent in split_sentences(chunk.text):
                overlap = len(q_stems & keyword_stems(sent))
                for span in find_entity_spans(sent):
                    key = normalize_name(span.text)
                    if not key or f" {key} " in q_norm:
                        continue
                    if want is not None and span.type not in (want, EntityType.OTHER):
                        continue
                    scores[span.text] += (1.0 / (rank + 1)) * (1.0 + overlap)
        ranked = sorted(scores, key=lambda s: (-scores[s], s))
        return VectorAnswer(question=question, answer=ranked[0] if ranked else None, candidates=ranked[:5], retrieved=retrieved)
