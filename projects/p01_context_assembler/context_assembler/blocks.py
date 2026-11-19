"""Convenience constructors with sensible default policies per source type.

Defaults (priority / required / strategy / weight):

=============  ===  ========  ===============  ======
block          pri  required  strategy         weight
=============  ===  ========  ===============  ======
system         100  yes       truncate         -
query          100  yes       truncate_middle  -
tool_defs       80  no        top_k (no cut)   1.0
tool_results    75  no        truncate_middle  2.0
retrieval       70  no        top_k + edges    3.0
history         60  no        summarize        2.0
memories        50  no        top_k            1.0
=============  ===  ========  ===============  ======
"""
from __future__ import annotations

from typing import Any, Iterable

from .models import BlockKind, ContextBlock, ContextItem, Placement, Strategy


def system_block(text: str, name: str = "system", **kw: Any) -> ContextBlock:
    opts: dict[str, Any] = dict(priority=100, required=True, strategy=Strategy.TRUNCATE)
    opts.update(kw)
    return ContextBlock(name=name, kind=BlockKind.SYSTEM, items=[ContextItem(id=name, text=text)], **opts)


def query_block(text: str, name: str = "query", **kw: Any) -> ContextBlock:
    opts: dict[str, Any] = dict(priority=100, required=True, strategy=Strategy.TRUNCATE_MIDDLE)
    opts.update(kw)
    return ContextBlock(name=name, kind=BlockKind.QUERY, items=[ContextItem(id=name, text=text)], **opts)


def history_block(messages: Iterable[dict[str, Any]], name: str = "history", **kw: Any) -> ContextBlock:
    """``messages``: dicts with ``role`` and ``content`` (optionally ``id``, ``pinned``)."""
    items = [
        ContextItem(id=str(m.get("id", f"msg{i}")), role=m["role"], text=m["content"], timestamp=float(i),
                    pinned=bool(m.get("pinned", False)))
        for i, m in enumerate(messages)
    ]
    opts: dict[str, Any] = dict(priority=60, strategy=Strategy.SUMMARIZE, weight=2.0, min_tokens=0)
    opts.update(kw)
    return ContextBlock(name=name, kind=BlockKind.HISTORY, items=items, **opts)


def memory_block(memories: Iterable[dict[str, Any] | str], name: str = "memories", **kw: Any) -> ContextBlock:
    """``memories``: strings or dicts with ``text`` and optional ``importance``/``timestamp``/``id``."""
    items = []
    for i, m in enumerate(memories):
        m = {"text": m} if isinstance(m, str) else m
        items.append(ContextItem(id=str(m.get("id", f"mem{i}")), text=m["text"],
                                 importance=float(m.get("importance", 0.5)), timestamp=m.get("timestamp"),
                                 pinned=bool(m.get("pinned", False))))
    opts: dict[str, Any] = dict(priority=50, strategy=Strategy.TOP_K, placement=Placement.RELEVANCE,
                                allow_partial=False, weight=1.0)
    opts.update(kw)
    return ContextBlock(name=name, kind=BlockKind.MEMORY, items=items, **opts)


def retrieval_block(docs: Iterable[dict[str, Any]], name: str = "retrieval", **kw: Any) -> ContextBlock:
    """``docs``: dicts with ``id``, ``text`` and optional ``score``/``source``/``title``."""
    items = []
    for i, d in enumerate(docs):
        meta = {k: str(d[k]) for k in ("source", "title") if k in d}
        items.append(ContextItem(id=str(d.get("id", f"doc{i}")), text=d["text"], score=d.get("score"),
                                 metadata=meta))
    opts: dict[str, Any] = dict(priority=70, strategy=Strategy.TOP_K, placement=Placement.EDGES, weight=3.0,
                                mmr_lambda=0.8)
    opts.update(kw)
    return ContextBlock(name=name, kind=BlockKind.RETRIEVAL, items=items, **opts)


def tools_block(tools: Iterable[dict[str, Any]], name: str = "tools", pinned: Iterable[str] = (), **kw: Any) -> ContextBlock:
    """``tools``: Messages-API tool definitions (``name``, ``description``, ``input_schema``)."""
    pins = set(pinned)
    items = [
        ContextItem(id=t["name"], text=t.get("description", ""), tool=t, pinned=t["name"] in pins) for t in tools
    ]
    opts: dict[str, Any] = dict(priority=80, strategy=Strategy.TOP_K, allow_partial=False, weight=1.0)
    opts.update(kw)
    return ContextBlock(name=name, kind=BlockKind.TOOL_DEFS, items=items, **opts)


def tool_results_block(results: Iterable[dict[str, Any]], name: str = "tool_results", **kw: Any) -> ContextBlock:
    """``results``: dicts with ``name`` (tool name), ``content`` and optional ``id``."""
    items = [
        ContextItem(id=str(r.get("id", r["name"] if i == 0 else f"{r['name']}#{i}")), text=r["content"],
                    timestamp=float(i), metadata={"tool": r["name"]})
        for i, r in enumerate(results)
    ]
    opts: dict[str, Any] = dict(priority=75, strategy=Strategy.TRUNCATE_MIDDLE, weight=2.0)
    opts.update(kw)
    return ContextBlock(name=name, kind=BlockKind.TOOL_RESULT, items=items, **opts)
