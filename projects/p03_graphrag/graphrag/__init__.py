"""graphrag: unstructured text -> knowledge graph -> multi-hop and global answers.

Offline by default (heuristic extraction, extractive summaries, template
semantic parsing); every LLM step is pluggable through `aikit.LLM`.
"""
from .baseline import VectorAnswer, VectorRAG
from .chunking import SentenceChunker
from .community import Community, CommunitySummarizer, detect_communities, verbalize
from .corpus import DOCUMENTS, GLOBAL_QUESTIONS, GOLD_TRIPLES, QUESTIONS, QAItem
from .extraction import Extractor, HeuristicExtractor, LLMExtractor
from .graph import KnowledgeGraph
from .models import (
    Chunk, Document, Entity, EntityType, ExtractedRelation, ExtractionResult, Mention, RelationType,
)
from .pipeline import GraphRAG
from .query import (
    EntityRef, GlobalAnswer, GlobalSearch, Hop, LocalAnswer, LocalSearch, PathStep, QuestionParser,
)
from .resolution import EntityResolver, Resolution

__all__ = [
    "VectorAnswer", "VectorRAG", "SentenceChunker", "Community", "CommunitySummarizer", "detect_communities",
    "verbalize", "DOCUMENTS", "GLOBAL_QUESTIONS", "GOLD_TRIPLES", "QUESTIONS", "QAItem", "Extractor",
    "HeuristicExtractor", "LLMExtractor", "KnowledgeGraph", "Chunk", "Document", "Entity", "EntityType",
    "ExtractedRelation", "ExtractionResult", "Mention", "RelationType", "GraphRAG", "EntityRef", "GlobalAnswer",
    "GlobalSearch", "Hop", "LocalAnswer", "LocalSearch", "PathStep", "QuestionParser", "EntityResolver",
    "Resolution",
]
