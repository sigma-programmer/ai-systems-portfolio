"""The hybrid search pipeline: filter -> sparse + dense -> fuse -> rerank.

    query ─┬─> BM25 (inverted index) ──────┐
           │                               ├─> fusion (RRF | weighted) ─> rerank ─> top-k
           └─> embed ─> Flat | IVF index ──┘
    metadata filter ─> bitmask applied inside both retrievers (pre-retrieval)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field

from aikit import HashingEmbedder

from .bm25 import BM25Index
from .dense import FlatIndex, IVFIndex
from .features import FeatureExtractor
from .filters import Filter, MetadataStore
from .fusion import reciprocal_rank_fusion, weighted_score_fusion
from .rerank import CrossEncoderReranker, Reranker
from .types import Document, Hit


class SearchConfig(BaseModel):
    """Every latency/accuracy knob of the stack in one validated object."""

    mode: Literal["sparse", "dense", "hybrid"] = "hybrid"
    fusion: Literal["rrf", "weighted"] = "rrf"
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="dense weight for weighted fusion")
    rrf_k: int = Field(60, ge=1)
    candidate_k: int = Field(50, ge=1, description="candidates pulled from each retriever")
    dense_index: Literal["flat", "ivf"] = "flat"
    nprobe: int = Field(4, ge=1)
    rerank: bool = False
    rerank_depth: int = Field(20, ge=1, description="how many fused hits the reranker rescores")
    name: str | None = None

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        parts = [self.mode]
        if self.mode != "sparse":
            parts.append(self.dense_index + (f"@{self.nprobe}" if self.dense_index == "ivf" else ""))
        if self.mode == "hybrid":
            parts.append(self.fusion if self.fusion == "rrf" else f"w{self.alpha:g}")
        parts.append(f"c{self.candidate_k}")
        if self.rerank:
            parts.append(f"rr{self.rerank_depth}")
        return "/".join(parts)


@dataclass
class SearchResponse:
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    candidates_considered: int = 0

    @property
    def doc_ids(self) -> list[str]:
        return [h.doc_id for h in self.hits]


class HybridSearcher:
    """Owns the indexes for one corpus and executes `SearchConfig`s against them."""

    def __init__(
        self,
        docs: Sequence[Document],
        embedder: Any | None = None,
        ivf_lists: int = 16,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ) -> None:
        if len({d.id for d in docs}) != len(docs):
            raise ValueError("document ids must be unique")
        self.docs = list(docs)
        self.embedder = embedder or HashingEmbedder()
        texts = [d.full_text for d in self.docs]
        t0 = time.perf_counter()
        self.bm25 = BM25Index(k1=bm25_k1, b=bm25_b).fit(texts)
        self.vectors = self.embedder.embed(texts) if texts else np.zeros((0, self.embedder.dim), np.float32)
        self.flat = FlatIndex(self.vectors)
        self.ivf = IVFIndex(self.vectors, n_lists=ivf_lists)
        self.metadata = MetadataStore([d.metadata for d in self.docs])
        self.extractor = FeatureExtractor(self.bm25, [d.title for d in self.docs], texts, self.vectors)
        self.reranker: Reranker = CrossEncoderReranker(self.extractor)
        self.index_build_ms = (time.perf_counter() - t0) * 1000

    def set_reranker(self, reranker: Reranker) -> None:
        self.reranker = reranker

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.embed([query])[0]

    def _hit(self, idx: int, score: float, comps: dict[str, float]) -> Hit:
        return Hit(idx, self.docs[idx].id, score, comps)

    def candidates(
        self,
        query: str,
        config: SearchConfig,
        flt: Filter | dict | None = None,
        qvec: np.ndarray | None = None,
        timings: dict[str, float] | None = None,
    ) -> list[Hit]:
        """First-stage retrieval + fusion (no reranking), best first."""
        timings = timings if timings is not None else {}
        t = time.perf_counter()
        mask = self.metadata.mask(flt)
        timings["filter"] = (time.perf_counter() - t) * 1000
        lists: dict[str, list[tuple[int, float]]] = {}
        if config.mode in ("sparse", "hybrid"):
            t = time.perf_counter()
            lists["bm25"] = self.bm25.search(query, config.candidate_k, mask)
            timings["sparse"] = (time.perf_counter() - t) * 1000
        if config.mode in ("dense", "hybrid"):
            t = time.perf_counter()
            if qvec is None:
                qvec = self.embed_query(query)
            if config.dense_index == "ivf":
                lists["dense"] = self.ivf.search(qvec, config.candidate_k, mask, nprobe=config.nprobe)
            else:
                lists["dense"] = self.flat.search(qvec, config.candidate_k, mask)
            timings["dense"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        if config.mode != "hybrid":
            (name, ranked), = lists.items()
            hits = [self._hit(i, s, {name: s}) for i, s in ranked]
        else:
            raw = {name: dict(r) for name, r in lists.items()}
            if config.fusion == "rrf":
                fused = reciprocal_rank_fusion(lists, k=config.rrf_k)
            else:
                fused = weighted_score_fusion(lists, {"dense": config.alpha, "bm25": 1 - config.alpha})
            hits = []
            for i, s, _parts in fused:
                comps = {name: raw[name].get(i, 0.0) for name in raw}
                comps["fused"] = s
                hits.append(self._hit(i, s, comps))
        timings["fusion"] = (time.perf_counter() - t) * 1000
        return hits

    def search(
        self,
        query: str,
        k: int = 10,
        config: SearchConfig | None = None,
        filter: Filter | dict | None = None,
    ) -> SearchResponse:
        config = config or SearchConfig()
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        qvec = self.embed_query(query) if (config.mode != "sparse" or config.rerank) else None
        hits = self.candidates(query, config, filter, qvec, timings)
        considered = len(hits)
        if config.rerank and hits:
            t = time.perf_counter()
            depth = max(config.rerank_depth, k)
            head = self.reranker.rerank(query, qvec, hits[:depth])
            hits = head + hits[depth:]
            timings["rerank"] = (time.perf_counter() - t) * 1000
        timings["total"] = (time.perf_counter() - t0) * 1000
        return SearchResponse(hits[:k], timings, considered)
