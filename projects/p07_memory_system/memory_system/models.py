"""Data records for the three memory tiers."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class WorkingItem:
    """One turn held in the working scratchpad."""

    id: str
    session_id: str
    role: str
    text: str
    created: float
    last_access: float
    tokens: int
    salience: float
    access_count: int = 0


@dataclass
class Episode:
    """A compressed window of turns that left working memory."""

    id: str
    session_id: str
    start: float
    end: float
    summary: str
    raw_turns: list[str]
    raw_tokens: int
    summary_tokens: int
    salience: float
    last_access: float
    access_count: int = 0
    consolidated: bool = False

    @property
    def compression_ratio(self) -> float:
        return self.summary_tokens / self.raw_tokens if self.raw_tokens else 1.0


@dataclass
class Fact:
    """A distilled (subject, predicate, value) belief about the user or world."""

    id: str
    subject: str
    predicate: str
    value: str
    text: str
    created: float
    last_access: float
    source_episode: str | None = None
    confidence: float = 0.7
    salience: float = 0.7
    access_count: int = 0
    status: str = "active"  # "active" | "superseded"
    superseded_by: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject, self.predicate)


@dataclass
class RecallHit:
    tier: str
    id: str
    text: str
    score: float
    similarity: float
    importance: float
    recency: float
