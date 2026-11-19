"""Typed data model shared by every stage of the GraphRAG pipeline."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Coarse entity classes produced by the extractors."""

    PERSON = "PERSON"
    ORG = "ORG"
    LOCATION = "LOCATION"
    PRODUCT = "PRODUCT"
    OTHER = "OTHER"


class RelationType(str, Enum):
    """Canonical relation vocabulary. RELATED_TO is the co-occurrence fallback."""

    FOUNDED = "FOUNDED"
    ACQUIRED = "ACQUIRED"
    CEO_OF = "CEO_OF"
    CTO_OF = "CTO_OF"
    CFO_OF = "CFO_OF"
    LOCATED_IN = "LOCATED_IN"
    INVESTED_IN = "INVESTED_IN"
    PARTNERED_WITH = "PARTNERED_WITH"
    WORKS_AT = "WORKS_AT"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    PRODUCES = "PRODUCES"
    SUPPLIES = "SUPPLIES"
    ALUMNUS_OF = "ALUMNUS_OF"
    RELATED_TO = "RELATED_TO"


#: Relations that hold in both directions; traversal ignores edge direction.
SYMMETRIC_RELATIONS = frozenset({RelationType.PARTNERED_WITH, RelationType.RELATED_TO})

#: Loose (subject_types, object_types) signatures used for type inference and
#: for rejecting implausible extractions. ``None`` means "any type".
RELATION_SIGNATURES: dict[RelationType, tuple[frozenset | None, frozenset | None]] = {
    RelationType.FOUNDED: (frozenset({EntityType.PERSON, EntityType.ORG}), frozenset({EntityType.ORG})),
    RelationType.ACQUIRED: (frozenset({EntityType.ORG}), frozenset({EntityType.ORG})),
    RelationType.CEO_OF: (frozenset({EntityType.PERSON}), frozenset({EntityType.ORG})),
    RelationType.CTO_OF: (frozenset({EntityType.PERSON}), frozenset({EntityType.ORG})),
    RelationType.CFO_OF: (frozenset({EntityType.PERSON}), frozenset({EntityType.ORG})),
    RelationType.LOCATED_IN: (frozenset({EntityType.ORG}), frozenset({EntityType.LOCATION})),
    RelationType.INVESTED_IN: (frozenset({EntityType.ORG, EntityType.PERSON}), frozenset({EntityType.ORG})),
    RelationType.PARTNERED_WITH: (frozenset({EntityType.ORG}), frozenset({EntityType.ORG})),
    RelationType.WORKS_AT: (frozenset({EntityType.PERSON}), frozenset({EntityType.ORG})),
    RelationType.SUBSIDIARY_OF: (frozenset({EntityType.ORG}), frozenset({EntityType.ORG})),
    RelationType.PRODUCES: (frozenset({EntityType.ORG}), frozenset({EntityType.PRODUCT})),
    RelationType.SUPPLIES: (frozenset({EntityType.ORG}), frozenset({EntityType.ORG, EntityType.PRODUCT})),
    RelationType.ALUMNUS_OF: (frozenset({EntityType.PERSON}), frozenset({EntityType.ORG})),
    RelationType.RELATED_TO: (None, None),
}


class Document(BaseModel):
    """A raw input document."""

    id: str
    title: str = ""
    text: str


class Chunk(BaseModel):
    """A contiguous, sentence-aligned piece of a document."""

    id: str
    doc_id: str
    index: int
    text: str
    n_tokens: int


class Mention(BaseModel):
    """A surface occurrence of an entity inside a chunk."""

    text: str
    type: EntityType
    chunk_id: str
    sentence_index: int = 0
    start: int = 0
    end: int = 0


class ExtractedRelation(BaseModel):
    """A (subject, relation, object) triple using surface strings."""

    subject: str
    relation: RelationType
    object: str
    chunk_id: str
    evidence: str = ""
    confidence: float = 1.0


class ExtractionResult(BaseModel):
    """Everything an extractor found in one chunk."""

    chunk_id: str
    mentions: list[Mention] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    #: (alias, canonical-surface) pairs found via "X (Y)" / "also known as".
    aliases: list[tuple[str, str]] = Field(default_factory=list)


class Entity(BaseModel):
    """A resolved, deduplicated entity (a graph node)."""

    id: str
    name: str
    type: EntityType
    aliases: list[str] = Field(default_factory=list)
    mention_count: int = 0
    source_chunks: list[str] = Field(default_factory=list)
