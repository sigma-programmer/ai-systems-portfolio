"""Render assembled sections to an Anthropic Messages-API-shaped payload, and count it."""
from __future__ import annotations

from typing import Any

from .formatting import BASE_OVERHEAD, MESSAGE_OVERHEAD, SECTION_SEP, Counter, tool_json


def render_payload(
    system_parts: list[str],
    memory_parts: list[str],
    tools: list[dict[str, Any]],
    history: list[tuple[str, str]],
    final_parts: list[str],
    max_output_tokens: int,
    cache_static_prefix: bool = True,
) -> dict[str, Any]:
    """Build ``{"system", "messages", "tools", "max_tokens"}``.

    * The system prompt is the stable prefix; with ``cache_static_prefix`` it
      becomes its own text block marked ``cache_control: ephemeral`` (as is the
      last tool definition), so repeated calls reuse the prompt cache. Memories
      follow in a separate, uncached block because they change per request.
    * Adjacent messages with the same role are merged, so the list alternates
      and starts with ``user`` (compressors already drop leading assistant turns).
    * Retrieved documents and tool results come right before the query in the
      final user turn: the query is always the last thing the model reads.
    """
    system_text = SECTION_SEP.join(p for p in system_parts if p)
    memory_text = SECTION_SEP.join(p for p in memory_parts if p)
    payload: dict[str, Any] = {"max_tokens": max_output_tokens}
    if cache_static_prefix:
        blocks: list[dict[str, Any]] = []
        if system_text:
            blocks.append({"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}})
        if memory_text:
            blocks.append({"type": "text", "text": memory_text})
        if blocks:
            payload["system"] = blocks
    else:
        joined = SECTION_SEP.join(p for p in (system_text, memory_text) if p)
        if joined:
            payload["system"] = joined

    messages: list[dict[str, Any]] = []
    for role, text in history + [("user", SECTION_SEP.join(p for p in final_parts if p))]:
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] = SECTION_SEP.join(x for x in (messages[-1]["content"], text) if x)
        else:
            messages.append({"role": role, "content": text})
    payload["messages"] = messages

    if tools:
        rendered = [dict(t) for t in tools]
        if cache_static_prefix:
            rendered[-1]["cache_control"] = {"type": "ephemeral"}
        payload["tools"] = rendered
    return payload


def _text_of(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    return [b.get("text", "") for b in content if isinstance(b, dict)]


def count_payload_tokens(payload: dict[str, Any], counter: Counter) -> int:
    """Input-token estimate of a payload, consistent with :class:`Formatter` costs."""
    total = BASE_OVERHEAD
    if "system" in payload:
        total += sum(counter(t) for t in _text_of(payload["system"]))
    for msg in payload.get("messages", []):
        total += MESSAGE_OVERHEAD + sum(counter(t) for t in _text_of(msg["content"]))
    for tool in payload.get("tools", []):
        total += counter(tool_json({k: v for k, v in tool.items() if k != "cache_control"}))
    return total


def to_prompt(payload: dict[str, Any]) -> tuple[str, str]:
    """Flatten a payload into ``(system, prompt)`` for the text-only :class:`aikit.LLM` protocol."""
    system = SECTION_SEP.join(_text_of(payload.get("system", "")))
    parts = []
    if payload.get("tools"):
        names = ", ".join(t["name"] for t in payload["tools"])
        parts.append(f"(Available tools: {names})")
    for msg in payload["messages"]:
        parts.append(f"{msg['role'].capitalize()}: {SECTION_SEP.join(_text_of(msg['content']))}")
    return system, SECTION_SEP.join(parts)
