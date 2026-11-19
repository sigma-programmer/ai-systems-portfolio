"""context_assembler: master the context window instead of fighting it.

Typed context sources -> token budget allocation -> per-block compression ->
lost-in-the-middle-aware placement -> Anthropic Messages-API payload, with a
hard guarantee that the payload never exceeds ``window - reserve_output``.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:  # allow `python -m context_assembler.bench` without installing aikit
    import aikit  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))

from .allocator import Allocation, BlockDemand, BudgetAllocator, proportional_split, water_fill
from .assembler import AssembledContext, ContextAssembler
from .blocks import (
    history_block, memory_block, query_block, retrieval_block, system_block, tool_results_block, tools_block,
)
from .compressors import CompressionContext, CompressionResult, compress, relevance_scores
from .formatting import Formatter, fit_middle, fit_prefix
from .models import BlockKind, ContextBlock, ContextItem, ContextOverflowError, Placement, Strategy
from .placement import edge_order, order_items
from .render import count_payload_tokens, render_payload, to_prompt
from .report import AssemblyReport, BlockReport
from .summarize import ExtractiveSummarizer, LLMSummarizer, Summarizer

__all__ = [
    "Allocation", "BlockDemand", "BudgetAllocator", "proportional_split", "water_fill",
    "AssembledContext", "ContextAssembler",
    "history_block", "memory_block", "query_block", "retrieval_block", "system_block", "tool_results_block",
    "tools_block", "CompressionContext", "CompressionResult", "compress", "relevance_scores",
    "Formatter", "fit_middle", "fit_prefix",
    "BlockKind", "ContextBlock", "ContextItem", "ContextOverflowError", "Placement", "Strategy",
    "edge_order", "order_items", "count_payload_tokens", "render_payload", "to_prompt",
    "AssemblyReport", "BlockReport", "ExtractiveSummarizer", "LLMSummarizer", "Summarizer",
]
