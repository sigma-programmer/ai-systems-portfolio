"""hybrid_retrieval: BM25 + dense retrieval, fusion, reranking, filtering, evaluation."""
import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _Path

if _ilu.find_spec("aikit") is None:  # running outside pytest: locate the monorepo's common/
    _common = _Path(__file__).resolve().parents[3] / "common"
    if (_common / "aikit").is_dir():
        _sys.path.insert(0, str(_common))
from .bm25 import BM25Index
from .corpus import build_corpus, build_queries, load_benchmark, split_queries
from .dense import FlatIndex, IVFIndex, spherical_kmeans
from .evaluation import evaluate, ndcg_at_k, recall_at_k, reciprocal_rank
from .features import FEATURE_NAMES, FeatureExtractor
from .filters import And, Eq, Filter, In, MetadataStore, Not, Or, Range, parse_filter
from .fusion import normalize_scores, reciprocal_rank_fusion, weighted_score_fusion
from .logreg import LogisticRegression
from .pipeline import HybridSearcher, SearchConfig, SearchResponse
from .rerank import CrossEncoderReranker, LLMReranker, Reranker
from .tuning import default_sweep_configs, pareto_frontier, sweep, train_reranker
from .types import Document, Hit, Query

__all__ = [
    "BM25Index", "FlatIndex", "IVFIndex", "spherical_kmeans",
    "Document", "Hit", "Query",
    "Filter", "Eq", "In", "Range", "And", "Or", "Not", "parse_filter", "MetadataStore",
    "reciprocal_rank_fusion", "weighted_score_fusion", "normalize_scores",
    "FeatureExtractor", "FEATURE_NAMES", "LogisticRegression",
    "CrossEncoderReranker", "LLMReranker", "Reranker",
    "HybridSearcher", "SearchConfig", "SearchResponse",
    "evaluate", "recall_at_k", "reciprocal_rank", "ndcg_at_k",
    "build_corpus", "build_queries", "load_benchmark", "split_queries",
    "train_reranker", "sweep", "pareto_frontier", "default_sweep_configs",
]
