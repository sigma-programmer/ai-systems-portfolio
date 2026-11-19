"""LLM abstraction with a real Claude backend and a deterministic mock.

Every project depends only on the `LLM` protocol:
    llm.complete(prompt, system=None, max_tokens=..., json_mode=False) -> LLMResponse

`ClaudeLLM` calls the Anthropic Messages API (claude-opus-5-5 by default,
adaptive thinking, server-side refusal fallback). `MockLLM` routes prompts to
registered handlers so tests and demos are deterministic and free.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .tokens import count_tokens

DEFAULT_MODEL = "claude-opus-5-5"


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = "mock"
    stop_reason: str = "end_turn"

    def json(self):
        """Parse the first JSON object/array found in the text."""
        return extract_json(self.text)


def extract_json(text: str):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in LLM output: {text[:200]!r}")


class LLM(Protocol):
    model: str

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse: ...


class LLMRefusal(RuntimeError):
    pass


class ClaudeLLM:
    """Claude via the official Anthropic SDK.

    Uses adaptive thinking (the only mode on Opus 5.5) and the server-side
    refusal fallback beta so a safety-classifier refusal is retried on a
    fallback model instead of surfacing as an empty answer.
    """

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium", client=None):
        import anthropic  # imported lazily so offline use needs no SDK

        self.model = model
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def complete(self, prompt, system=None, max_tokens=2048, json_mode=False) -> LLMResponse:
        sys_prompt = system or ""
        if json_mode:
            sys_prompt += "\nRespond with a single valid JSON value and nothing else."
        kwargs = dict(
            model=self.model,
            max_tokens=max(max_tokens, 1024),
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if sys_prompt.strip():
            kwargs["system"] = sys_prompt.strip()
        resp = self.client.beta.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            raise LLMRefusal(str(getattr(resp, "stop_details", None)))
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=resp.model,
            stop_reason=resp.stop_reason or "end_turn",
        )


Handler = Callable[[str, str | None], str]


@dataclass
class MockLLM:
    """Deterministic LLM stand-in.

    Register handlers with `on(pattern, fn_or_str)`; the first handler whose
    regex matches the prompt (or system prompt) produces the reply. Records
    every call in `.calls` so tests can assert on prompts.
    """

    model: str = "mock-llm"
    default: Handler | str = "OK"
    handlers: list[tuple[re.Pattern, Handler | str]] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def on(self, pattern: str, reply: Handler | str) -> "MockLLM":
        self.handlers.append((re.compile(pattern, re.I | re.S), reply))
        return self

    def complete(self, prompt, system=None, max_tokens=2048, json_mode=False) -> LLMResponse:
        self.calls.append({"prompt": prompt, "system": system, "json_mode": json_mode})
        reply: Handler | str = self.default
        haystack = f"{system or ''}\n{prompt}"
        for pat, r in self.handlers:
            if pat.search(haystack):
                reply = r
                break
        text = reply(prompt, system) if callable(reply) else reply
        if not isinstance(text, str):
            text = json.dumps(text)
        return LLMResponse(
            text=text,
            input_tokens=count_tokens(haystack),
            output_tokens=count_tokens(text),
            model=self.model,
        )


def get_llm(mock: MockLLM | None = None, **kwargs) -> LLM:
    """Real Claude if credentials exist and AIKIT_OFFLINE is unset, else mock."""
    if os.environ.get("AIKIT_OFFLINE") or not (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        return mock or MockLLM()
    return ClaudeLLM(**kwargs)
