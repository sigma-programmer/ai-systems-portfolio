"""Simulated LLM backends implementing the `aikit.LLM` protocol.

Each backend has a seeded RNG, a latency model (base + per-output-token +
log-normal jitter) and independent probabilities of timeout, rate-limit and
5xx-style failure. Latency is *virtual* (reported, not slept), so a 500-request
replay runs in milliseconds yet produces realistic p50/p95 numbers.
`outage(start, end)` makes calls fail deterministically in a request-index window.
"""
from __future__ import annotations

import hashlib
import math
import random

from aikit import LLMResponse, count_tokens


class BackendError(RuntimeError):
    kind = "error"


class BackendTimeout(BackendError):
    kind = "timeout"


class RateLimited(BackendError):
    kind = "rate_limited"


class BackendUnavailable(BackendError):
    kind = "unavailable"


class SimulatedBackend:
    def __init__(self, model: str, base_latency_ms: float = 400.0, ms_per_token: float = 8.0,
                 jitter: float = 0.25, failure_rate: float = 0.0, timeout_rate: float = 0.0,
                 rate_limit_rate: float = 0.0, timeout_ms: float = 10_000.0, seed: int = 0,
                 verbosity: float = 1.0):
        self.model = model
        self.base_latency_ms = base_latency_ms
        self.ms_per_token = ms_per_token
        self.jitter = jitter
        self.failure_rate = failure_rate
        self.timeout_rate = timeout_rate
        self.rate_limit_rate = rate_limit_rate
        self.timeout_ms = timeout_ms
        self.verbosity = verbosity
        self.rng = random.Random(seed)
        self.calls = 0
        self.last_latency_ms = 0.0
        self._outages: list[tuple[int, int]] = []

    def outage(self, start_call: int, end_call: int) -> None:
        """Fail every call whose index (0-based, per backend) is in [start, end)."""
        self._outages.append((start_call, end_call))

    def _answer(self, prompt: str, max_tokens: int) -> str:
        digest = hashlib.blake2b(prompt.encode(), digest_size=4).hexdigest()
        n_out = min(max_tokens, int(30 + self.verbosity * 0.6 * count_tokens(prompt)))
        filler = ("data " * max(1, n_out - 8)).strip()  # "data" = exactly one token
        return f"[{self.model}#{digest}] {filler}"

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 512,
                 json_mode: bool = False) -> LLMResponse:
        idx = self.calls
        self.calls += 1
        r = self.rng.random()
        if any(s <= idx < e for s, e in self._outages):
            self.last_latency_ms = 50.0
            raise BackendUnavailable(f"{self.model} outage")
        if r < self.timeout_rate:
            self.last_latency_ms = self.timeout_ms
            raise BackendTimeout(f"{self.model} timed out")
        if r < self.timeout_rate + self.rate_limit_rate:
            self.last_latency_ms = 30.0
            raise RateLimited(f"{self.model} 429")
        if r < self.timeout_rate + self.rate_limit_rate + self.failure_rate:
            self.last_latency_ms = 120.0
            raise BackendUnavailable(f"{self.model} 503")
        text = self._answer(prompt, max_tokens)
        out_tokens = count_tokens(text)
        noise = math.exp(self.rng.gauss(0.0, self.jitter))
        self.last_latency_ms = (self.base_latency_ms + self.ms_per_token * out_tokens) * noise
        if self.last_latency_ms > self.timeout_ms:
            self.last_latency_ms = self.timeout_ms
            raise BackendTimeout(f"{self.model} exceeded {self.timeout_ms}ms")
        return LLMResponse(text=text, input_tokens=count_tokens((system or "") + prompt),
                           output_tokens=out_tokens, model=self.model)


def default_backends(seed: int = 0, failure_rate: float = 0.0) -> dict[str, SimulatedBackend]:
    """Haiku/Sonnet/Opus simulators with latency profiles that scale with model size."""
    return {
        "claude-haiku-5-5": SimulatedBackend("claude-haiku-5-5", 250, 4, seed=seed + 1,
                                             failure_rate=failure_rate, verbosity=0.8),
        "claude-sonnet-5-5": SimulatedBackend("claude-sonnet-5-5", 450, 9, seed=seed + 2,
                                              failure_rate=failure_rate, verbosity=1.0),
        "claude-opus-5-5": SimulatedBackend("claude-opus-5-5", 700, 15, seed=seed + 3,
                                            failure_rate=failure_rate, verbosity=1.2),
    }
