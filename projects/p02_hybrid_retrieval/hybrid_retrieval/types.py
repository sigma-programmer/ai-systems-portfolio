"""Core data types shared across the retrieval stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A retrievable unit of text with a stable id and filterable metadata."""

    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Title and body joined; this is what both indexes see."""
        return f"{self.title}. {self.text}" if self.title else self.text


@dataclass
class Hit:
    """A scored candidate. `index` is the row in the searcher's doc table.

    `components` keeps every stage's score (bm25, dense, fused, rerank) so
    results are explainable after fusion and reranking.
    """

    index: int
    doc_id: str
    score: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class Query:
    """An evaluation query with graded relevance judgments (doc_id -> grade)."""

    id: str
    text: str
    qrels: dict[str, int]
    qtype: str = "generic"
    filter: dict[str, Any] | None = None
