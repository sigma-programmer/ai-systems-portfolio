"""Gateway facade: budget check → cache → complexity routing → degradation chain.

    request ─► CostTracker.decide(tenant) ──reject──► cache.nearest() or 402-style error
                    │ ok / downgrade      (also: pre-flight cost estimate vs cap headroom
                    │                      → downgrade to cheapest tier, or reject)
                    ▼
               SemanticCache.get(tenant, prompt) ──hit──► response (cost $0)
                    │ miss
                    ▼
               ComplexityAnalyzer → tier → Router → model (downgrade ⇒ cheapest tier)
                    ▼
               degradation chain [primary, fallbacks...] skipping open circuit breakers
                    │ all failed
                    ▼
               cache.nearest() (looser floor, flagged degraded) or graceful error
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping

from aikit import LLM, count_tokens

from .budget import CircuitBreaker, CostTracker
from .cache import SemanticCache
from .clock import Clock, SystemClock
from .complexity import ComplexityAnalyzer, ComplexityReport
from .pricing import DEFAULT_TIER_MODEL, MODELS, OPUS, TIER_ORDER, cost_of

DEFAULT_CHAINS: dict[str, list[str]] = {
    "claude-opus-5-5": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
    "claude-sonnet-5-5": ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"],
    "claude-haiku-5-5": ["claude-haiku-5-5", "claude-sonnet-5-5"],
}


class Router:
    """Maps complexity tiers to models; can cap the tier (cost kill-switch)."""

    def __init__(self, tier_model: Mapping[str, str] | None = None):
        self.tier_model = dict(tier_model or DEFAULT_TIER_MODEL)

    def route(self, tier: str, max_tier: str | None = None) -> str:
        if max_tier is not None and TIER_ORDER.index(tier) > TIER_ORDER.index(max_tier):
            tier = max_tier
        return self.tier_model[tier]


@dataclass
class GatewayRequest:
    prompt: str
    tenant: str = "default"
    max_tokens: int = 512
    system: str | None = None


@dataclass
class GatewayResponse:
    text: str | None
    ok: bool
    model: str | None
    cached: bool
    cost_usd: float
    latency_ms: float
    tier: str | None
    decision: str
    degraded: bool = False
    error: str | None = None
    attempts: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    complexity: ComplexityReport | None = None
    cache_similarity: float | None = None
    cache_source_meta: dict | None = None


class Gateway:
    def __init__(self, backends: Mapping[str, LLM], *, analyzer: ComplexityAnalyzer | None = None,
                 router: Router | None = None, cache: SemanticCache | None = None,
                 tracker: CostTracker | None = None, chains: Mapping[str, list[str]] | None = None,
                 breaker_threshold: int = 3, breaker_cooldown: float = 30.0,
                 cache_latency_ms: float = 2.0, clock: Clock | None = None):
        self.clock = clock or SystemClock()
        self.backends = dict(backends)
        self.analyzer = analyzer or ComplexityAnalyzer()
        self.router = router or Router()
        self.cache = cache if cache is not None else SemanticCache(clock=self.clock)
        self.tracker = tracker or CostTracker(clock=self.clock)
        self.chains = {k: list(v) for k, v in (chains or DEFAULT_CHAINS).items()}
        self.breakers = {m: CircuitBreaker(breaker_threshold, breaker_cooldown, self.clock)
                         for m in self.backends}
        self.cache_latency_ms = cache_latency_ms
        self.metrics: dict[str, int] = {"requests": 0, "cache_hits": 0, "fallbacks": 0,
                                        "degraded_cache": 0, "errors": 0, "rejected": 0,
                                        "downgraded": 0}

    # ------------------------------------------------------------------ helpers
    def _from_cache(self, hit, decision: str, t0: float, degraded: bool, tier: str | None,
                    report: ComplexityReport | None = None) -> GatewayResponse:
        return GatewayResponse(hit.entry.response, True, hit.entry.model, True, 0.0,
                               self.cache_latency_ms + (time.perf_counter() - t0) * 1000, tier,
                               decision, degraded, complexity=report,
                               cache_similarity=hit.similarity, cache_source_meta=hit.entry.meta)

    @staticmethod
    def _estimate(model: str, req: GatewayRequest) -> float:
        """Conservative pre-call cost estimate (output ≈ 30 + 0.8×input, capped)."""
        if model not in MODELS:
            return 0.0
        in_tok = count_tokens((req.system or "") + req.prompt)
        return cost_of(model, in_tok, min(req.max_tokens, int(30 + 0.8 * in_tok)))

    def _chain(self, model: str) -> list[str]:
        chain = self.chains.get(model, [model])
        return [m for m in chain if m in self.backends] or [model]

    # ------------------------------------------------------------------ main entry
    def handle(self, req: GatewayRequest, meta: dict | None = None) -> GatewayResponse:
        t0 = time.perf_counter()
        self.metrics["requests"] += 1
        decision = self.tracker.decide(req.tenant)
        if decision == "reject":
            self.metrics["rejected"] += 1
            hit = self.cache.nearest(req.tenant, req.prompt)
            if hit:
                self.metrics["degraded_cache"] += 1
                return self._from_cache(hit, decision, t0, True, None)
            return GatewayResponse(None, False, None, False, 0.0,
                                   (time.perf_counter() - t0) * 1000, None, decision,
                                   error="budget_exceeded")

        hit = self.cache.get(req.tenant, req.prompt)
        if hit:
            self.metrics["cache_hits"] += 1
            return self._from_cache(hit, decision, t0, False, None)

        report = self.analyzer.analyze(req.prompt)
        max_tier = TIER_ORDER[0] if decision == "downgrade" else None
        if max_tier:
            self.metrics["downgraded"] += 1
        primary = self.router.route(report.tier, max_tier)
        # Pre-flight: never start a call whose *estimated* cost would breach the hard cap.
        headroom = self.tracker.headroom(req.tenant)
        if headroom is not None and self._estimate(primary, req) > headroom:
            cheapest = self.router.route(TIER_ORDER[0])
            if self._estimate(cheapest, req) > headroom:
                self.metrics["rejected"] += 1
                hit = self.cache.nearest(req.tenant, req.prompt)
                if hit:
                    self.metrics["degraded_cache"] += 1
                    return self._from_cache(hit, "reject", t0, True, report.tier, report)
                return GatewayResponse(None, False, None, False, 0.0,
                                       (time.perf_counter() - t0) * 1000, report.tier, "reject",
                                       error="budget_exceeded_preflight", complexity=report)
            decision, max_tier, primary = "downgrade", TIER_ORDER[0], cheapest
            self.metrics["downgraded"] += 1
        chain = self._chain(primary)
        if max_tier:  # never fall back *up* to a pricier model while cost-capped
            chain = [m for m in chain if MODELS[m].tier == TIER_ORDER[0]] or chain[:1]

        attempts: list[dict] = []
        virtual_ms = 0.0
        for model in chain:
            breaker = self.breakers.setdefault(model, CircuitBreaker(clock=self.clock))
            if not breaker.allow():
                attempts.append({"model": model, "outcome": "skipped_open_breaker"})
                continue
            backend = self.backends[model]
            try:
                resp = backend.complete(req.prompt, system=req.system, max_tokens=req.max_tokens)
            except Exception as exc:  # noqa: BLE001 - any backend failure triggers fallback
                breaker.failure()
                virtual_ms += float(getattr(backend, "last_latency_ms", 0.0))
                attempts.append({"model": model, "outcome": getattr(exc, "kind", "error"),
                                 "error": str(exc)})
                continue
            breaker.success()
            virtual_ms += float(getattr(backend, "last_latency_ms", 0.0))
            in_tok = resp.input_tokens or count_tokens(req.prompt)
            out_tok = resp.output_tokens or count_tokens(resp.text)
            cost = cost_of(model, in_tok, out_tok) if model in MODELS else 0.0
            self.tracker.record(req.tenant, cost)
            self.cache.put(req.tenant, req.prompt, resp.text, model, meta)
            attempts.append({"model": model, "outcome": "ok"})
            fell_back = model != primary
            if fell_back:
                self.metrics["fallbacks"] += 1
            return GatewayResponse(resp.text, True, model, False, cost,
                                   virtual_ms + (time.perf_counter() - t0) * 1000, report.tier,
                                   decision, fell_back, None, attempts, in_tok, out_tok, report)

        hit = self.cache.nearest(req.tenant, req.prompt)
        if hit:
            self.metrics["degraded_cache"] += 1
            r = self._from_cache(hit, decision, t0, True, report.tier, report)
            r.attempts = attempts
            r.latency_ms += virtual_ms
            return r
        self.metrics["errors"] += 1
        return GatewayResponse(None, False, None, False, 0.0,
                               virtual_ms + (time.perf_counter() - t0) * 1000, report.tier,
                               decision, True, "all_backends_failed", attempts, complexity=report)

    def baseline_cost(self, prompt: str, output_tokens: int) -> float:
        """What the same call would cost on always-Opus with no cache."""
        return OPUS.cost(count_tokens(prompt), output_tokens)
