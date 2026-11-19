"""aikit: shared primitives for the AI systems portfolio.

Everything here works fully offline. When ANTHROPIC_API_KEY is set (and
AIKIT_OFFLINE is not), `get_llm()` returns a real Claude-backed client.
"""
from .tokens import count_tokens, truncate_to_tokens, tokenize_words
from .embeddings import Embedder, HashingEmbedder, cosine_sim
from .llm import LLM, LLMResponse, ClaudeLLM, MockLLM, get_llm, extract_json

__version__ = "1.0.0"

__all__ = [
    "count_tokens", "truncate_to_tokens", "tokenize_words",
    "Embedder", "HashingEmbedder", "cosine_sim",
    "LLM", "LLMResponse", "ClaudeLLM", "MockLLM", "get_llm", "extract_json",
]
