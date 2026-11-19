"""End-to-end GraphRAG pipeline: documents -> knowledge graph -> answers."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from aikit import LLM

from .chunking import SentenceChunker
from .community import Community, CommunitySummarizer, detect_communities
from .extraction import Extractor, HeuristicExtractor
from .graph import KnowledgeGraph
from .models import Chunk, Document, ExtractionResult
from .query import GlobalAnswer, GlobalSearch, LocalAnswer, LocalSearch
from .resolution import EntityResolver

_GLOBAL_CUES = re.compile(
    r"\b(summari[sz]e|overview|overall|main themes?|themes|trends?|landscape|across|in general|"
    r"which companies|what companies|major|key players)\b",
    re.I,
)


class GraphRAG:
    """Build a knowledge graph from documents and answer questions over it.

    Args:
        extractor: entity/relation extractor (offline heuristic by default,
            or :class:`~graphrag.extraction.LLMExtractor`).
        llm: optional LLM used for community summaries, answer phrasing and
            global-search map/reduce. ``None`` keeps everything offline.
        chunker: document chunker.
        seed: Louvain seed (community detection is otherwise nondeterministic).
    """

    def __init__(
        self,
        extractor: Extractor | None = None,
        llm: LLM | None = None,
        chunker: SentenceChunker | None = None,
        resolver: EntityResolver | None = None,
        seed: int = 42,
        community_level: int = 0,
    ):
        self.extractor = extractor or HeuristicExtractor()
        self.llm = llm
        self.chunker = chunker or SentenceChunker()
        self.resolver = resolver or EntityResolver()
        self.seed = seed
        self.community_level = community_level
        self.chunks: list[Chunk] = []
        self.extractions: list[ExtractionResult] = []
        self.kg = KnowledgeGraph()
        self.communities: list[Community] = []
        self.timings_ms: dict[str, float] = {}

    def index(self, documents: list[Document]) -> "GraphRAG":
        """Chunk, extract, resolve, build the graph, detect and summarise communities."""
        t = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal t
            now = time.perf_counter()
            self.timings_ms[name] = round((now - t) * 1000, 2)
            t = now

        self.chunks, self.extractions = [], []
        for doc in documents:
            doc_chunks = self.chunker.chunk(doc)
            self.chunks.extend(doc_chunks)
            self.extractions.extend(self.extractor.extract_document(doc_chunks))
        lap("chunk_extract")
        resolution = self.resolver.resolve(self.extractions)
        lap("resolve")
        self.kg = KnowledgeGraph.build(self.extractions, resolution, {c.id: c.text for c in self.chunks})
        lap("graph")
        self.communities = detect_communities(self.kg, seed=self.seed)
        CommunitySummarizer(llm=self.llm).summarize_all(self.kg, self.communities)
        lap("communities")
        return self

    # ------------------------------------------------------------------ querying
    def local_search(self, question: str, max_hops: int = 2) -> LocalAnswer:
        return LocalSearch(self.kg, llm=self.llm, max_hops=max_hops).search(question)

    def global_search(self, question: str, level: int | None = None) -> GlobalAnswer:
        lvl = self.community_level if level is None else level
        return GlobalSearch(self.kg, self.communities, level=lvl, llm=self.llm).search(question)

    def query(self, question: str) -> LocalAnswer | GlobalAnswer:
        """Route: parseable entity questions -> local; corpus-wide questions -> global."""
        local = LocalSearch(self.kg, llm=None)
        if local.parser.parse(question) is not None:
            return self.local_search(question)
        if _GLOBAL_CUES.search(question) or not local.link_mentions(question):
            return self.global_search(question)
        return self.local_search(question)

    # ------------------------------------------------------------------ export
    def to_json(self) -> dict[str, Any]:
        data = self.kg.to_dict()
        data["communities"] = [c.model_dump() for c in self.communities]
        data["stats"] = self.stats()
        return data

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    def stats(self) -> dict[str, Any]:
        levels: dict[int, int] = {}
        for c in self.communities:
            levels[c.level] = levels.get(c.level, 0) + 1
        return {
            **self.kg.stats(),
            "chunks": len(self.chunks),
            "communities_per_level": {str(k): v for k, v in sorted(levels.items())},
            "rejected_relations": len(self.kg.rejected),
        }
