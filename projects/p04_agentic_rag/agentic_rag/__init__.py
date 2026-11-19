"""Self-correcting agentic RAG: rewrite, multi-step retrieval, grounded citations, abstention."""
from .adversarial import HallucinatingGenerator
from .controller import AgenticRAG, Confidence, RAGResult, Trace
from .corpus import Document, load_corpus
from .generator import Claim, ExtractiveGenerator, LLMGenerator, parse_cited_sentences
from .grader import HeuristicGrader, LLMGrader, TermStats
from .retriever import BM25, HybridRetriever
from .rewriter import HeuristicRewriter, LLMRewriter, decompose
from .verifier import HallucinationDetector, HeuristicEntailment, LLMEntailment

__all__ = [
    "AgenticRAG", "Confidence", "RAGResult", "Trace", "Document", "load_corpus", "Claim",
    "ExtractiveGenerator", "LLMGenerator", "parse_cited_sentences", "HeuristicGrader",
    "LLMGrader", "TermStats", "BM25", "HybridRetriever", "HeuristicRewriter", "LLMRewriter",
    "decompose", "HallucinationDetector", "HeuristicEntailment", "LLMEntailment",
    "HallucinatingGenerator",
]
