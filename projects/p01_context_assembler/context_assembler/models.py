"""Typed data model: context items, blocks, strategies and errors."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class BlockKind(str, Enum):
    """Where a block's content lives in the rendered Messages-API payload."""

    SYSTEM = "system"            # system prompt (static, cacheable prefix)
    MEMORY = "memory"            # long-term memories -> appended to system
    TOOL_DEFS = "tool_defs"      # tool definitions -> top-level `tools`
    HISTORY = "history"          # prior conversation turns -> `messages`
    RETRIEVAL = "retrieval"      # retrieved documents -> final user turn
    TOOL_RESULT = "tool_result"  # tool outputs -> final user turn
    QUERY = "query"              # the current user request -> end of final user turn


class Strategy(str, Enum):
    """How a block shrinks when it receives less budget than it wants."""

    NONE = "none"                        # keep as-is; drop trailing items if forced
    DROP = "drop"                        # all-or-nothing
    TRUNCATE = "truncate"                # keep in order, cut the tail
    TRUNCATE_MIDDLE = "truncate_middle"  # keep each item's head and tail
    DROP_OLDEST = "drop_oldest"          # sliding window over history
    SUMMARIZE = "summarize"              # compress overflow into a summary
    TOP_K = "top_k"                      # rerank by relevance, keep the best


class Placement(str, Enum):
    """Ordering of ranked items inside a block (counters lost-in-the-middle)."""

    RELEVANCE = "relevance"  # best first
    EDGES = "edges"          # best at both ends, weakest in the middle
    ORIGINAL = "original"    # input order


class ContextOverflowError(RuntimeError):
    """Raised when the window cannot hold even the minimal payload."""


class ContextItem(BaseModel):
    """One unit of content inside a block: a message, memory, document, tool...

    Attributes:
        id: stable identifier (used in reports).
        text: the content. For tool definitions, the description.
        role: "user" / "assistant" for history items.
        score: upstream relevance score (e.g. from a retriever), any scale >= 0.
        importance: prior importance in [0, 1] (memories, pinned facts).
        timestamp: creation time (larger = newer) for recency weighting.
        pinned: never dropped by ranking strategies (still subject to emergencies).
        tool: Messages-API tool definition (``name``, ``description``, ``input_schema``).
        metadata: free-form attributes rendered as XML attributes for documents.
    """

    id: str
    text: str = ""
    role: str | None = None
    score: float | None = None
    importance: float = 0.5
    timestamp: float | None = None
    pinned: bool = False
    tool: dict[str, Any] | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class ContextBlock(BaseModel):
    """A typed source of context with its budgeting policy.

    Attributes:
        name: unique block name.
        kind: rendering destination (see :class:`BlockKind`).
        items: content units.
        priority: higher wins when budget is scarce (floors granted in priority order).
        required: hard-required blocks always receive their floor first.
        min_tokens: floor; ``None`` means "full demand" for required blocks and 0 otherwise.
        max_tokens: cap on the allocation, even if budget is plentiful.
        weight: share of surplus budget in proportional allocation.
        strategy: compression strategy when under-allocated.
        placement: item ordering for ranked blocks.
        top_k: maximum number of items kept by ranking strategies.
        mmr_lambda: if set, Maximal-Marginal-Relevance diversity (1.0 = pure relevance).
        allow_partial: whether ranking strategies may include a truncated last item.
        min_partial_tokens: smallest truncated item worth including.
        summary_ratio: for SUMMARIZE on history, the fraction of budget for the summary.
    """

    name: str
    kind: BlockKind
    items: list[ContextItem] = Field(default_factory=list)
    priority: int = 50
    required: bool = False
    min_tokens: int | None = None
    max_tokens: int | None = None
    weight: float = 1.0
    strategy: Strategy = Strategy.TRUNCATE
    placement: Placement = Placement.ORIGINAL
    top_k: int | None = None
    mmr_lambda: float | None = None
    allow_partial: bool = True
    min_partial_tokens: int = 24
    summary_ratio: float = 0.3

    @model_validator(mode="after")
    def _check(self) -> "ContextBlock":
        if self.weight < 0:
            raise ValueError("weight must be >= 0")
        if self.min_tokens is not None and self.min_tokens < 0:
            raise ValueError("min_tokens must be >= 0")
        if self.max_tokens is not None and self.max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if self.min_tokens is not None and self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens cannot exceed max_tokens")
        if not 0.0 <= self.summary_ratio <= 1.0:
            raise ValueError("summary_ratio must be in [0, 1]")
        if self.kind == BlockKind.TOOL_DEFS:
            for it in self.items:
                if not it.tool or "name" not in it.tool:
                    raise ValueError(f"tool item {it.id!r} needs a `tool` dict with a name")
        if self.kind == BlockKind.HISTORY:
            for it in self.items:
                if it.role not in ("user", "assistant"):
                    raise ValueError(f"history item {it.id!r} needs role 'user' or 'assistant'")
        return self
