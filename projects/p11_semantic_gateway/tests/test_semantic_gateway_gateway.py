"""Gateway integration tests: routing, caching, kill-switches, degradation, LLM protocol, bench."""
from __future__ import annotations

import json

import pytest

from aikit import MockLLM
from semantic_gateway import (CostTracker, Gateway, GatewayRequest, ManualClock, NullCache,
                              Router, SemanticCache, SimulatedBackend, SpendRateBreaker,
                              TenantBudget, default_backends)
from semantic_gateway.bench import benchmark

HARD = "Prove that the square root of 2 is irrational, step by step."


@pytest.fixture
def clock():
    return ManualClock()


def make(clock, **kw):
    kw.setdefault("cache", SemanticCache(clock=clock))
    return Gateway(kw.pop("backends", None) or default_backends(seed=1), clock=clock, **kw)


def test_routes_by_complexity(clock):
    gw = make(clock)
    assert gw.handle(GatewayRequest("What is the capital of France?")).model == "claude-haiku-5-5"
    assert gw.handle(GatewayRequest("Summarize the causes of World War I in a paragraph.")
                     ).model == "claude-sonnet-5-5"
    r = gw.handle(GatewayRequest(HARD))
    assert r.model == "claude-opus-5-5" and r.tier == "complex" and r.cost_usd > 0
    assert r.complexity is not None and r.attempts[-1]["outcome"] == "ok"


def test_cache_hit_is_free_and_tenant_scoped(clock):
    gw = make(clock)
    first = gw.handle(GatewayRequest(HARD, tenant="a"))
    hit = gw.handle(GatewayRequest("Give a rigorous step by step proof that the square root "
                                   "of 2 is irrational.", tenant="a"))
    assert hit.cached and hit.cost_usd == 0 and hit.text == first.text
    other = gw.handle(GatewayRequest(HARD, tenant="b"))
    assert not other.cached
    assert gw.metrics["cache_hits"] == 1


def test_numeric_variant_does_not_hit(clock):
    gw = make(clock)
    a = gw.handle(GatewayRequest("Convert 5 km to miles."))
    b = gw.handle(GatewayRequest("Convert 8 km to miles."))
    assert not b.cached and a.text != b.text


def test_soft_cap_downgrades_and_never_falls_back_upward(clock):
    tracker = CostTracker(budgets={"t": TenantBudget(1.0, 0.5)}, clock=clock)
    tracker.record("t", 0.6)
    backends = default_backends(seed=1)
    backends["claude-haiku-5-5"].failure_rate = 1.0
    gw = make(clock, tracker=tracker, backends=backends, cache=NullCache())
    r = gw.handle(GatewayRequest(HARD, tenant="t"))
    assert r.decision == "downgrade" and not r.ok
    assert [a["model"] for a in r.attempts] == ["claude-haiku-5-5"]
    backends["claude-haiku-5-5"].failure_rate = 0.0
    clock.advance(60)
    r2 = gw.handle(GatewayRequest(HARD, tenant="t"))
    assert r2.ok and r2.model == "claude-haiku-5-5"


def test_hard_cap_rejects_but_serves_cache(clock):
    tracker = CostTracker(clock=clock)
    gw = make(clock, tracker=tracker)
    first = gw.handle(GatewayRequest(HARD, tenant="t"))
    assert first.ok
    tracker.budgets["t"] = TenantBudget(0.0001)  # cap now below today's spend
    again = gw.handle(GatewayRequest(HARD, tenant="t"))
    assert again.decision == "reject" and again.ok and again.cached and again.degraded
    new = gw.handle(GatewayRequest("What is the capital of Japan?", tenant="t"))
    assert not new.ok and new.error == "budget_exceeded"
    assert gw.metrics["rejected"] == 2


def test_preflight_estimate_downgrades_or_rejects(clock):
    tracker = CostTracker(budgets={"t": TenantBudget(0.0005)}, clock=clock)
    gw = make(clock, tracker=tracker, cache=NullCache())
    r = gw.handle(GatewayRequest(HARD, tenant="t"))  # Opus estimate > cap, Haiku fits
    assert r.ok and r.model == "claude-haiku-5-5" and r.decision == "downgrade"
    tracker.budgets["t"] = TenantBudget(tracker.spent_today("t") + 1e-7)
    r2 = gw.handle(GatewayRequest("What is the capital of Japan?", tenant="t"))
    assert not r2.ok and r2.error == "budget_exceeded_preflight"
    assert tracker.headroom("t") > 0 and tracker.headroom("nobody") is None


def test_spend_rate_breaker_forces_cheap_tier(clock):
    br = SpendRateBreaker(0.000001, window=60, cooldown=600, clock=clock)
    gw = make(clock, tracker=CostTracker(rate_breaker=br, clock=clock), cache=NullCache())
    gw.handle(GatewayRequest("hello"))
    r = gw.handle(GatewayRequest(HARD))
    assert r.decision == "downgrade" and r.model == "claude-haiku-5-5"


def test_fallback_chain_and_breaker(clock):
    backends = default_backends(seed=1)
    backends["claude-opus-5-5"].failure_rate = 1.0
    gw = make(clock, backends=backends, cache=NullCache(), breaker_threshold=2,
              breaker_cooldown=30)
    r = gw.handle(GatewayRequest(HARD))
    assert r.ok and r.model == "claude-sonnet-5-5" and r.degraded
    assert r.attempts[0] == {"model": "claude-opus-5-5", "outcome": "unavailable",
                             "error": "claude-opus-5-5 503"}
    gw.handle(GatewayRequest(HARD + " Again."))
    r3 = gw.handle(GatewayRequest(HARD + " Third."))
    assert r3.attempts[0]["outcome"] == "skipped_open_breaker"
    assert gw.breakers["claude-opus-5-5"].trips == 1
    clock.advance(31)
    backends["claude-opus-5-5"].failure_rate = 0.0
    r4 = gw.handle(GatewayRequest(HARD + " Fourth."))
    assert r4.model == "claude-opus-5-5"
    assert gw.breakers["claude-opus-5-5"].state == "closed"


def test_total_outage_degrades_to_cache_then_errors(clock):
    backends = {m: SimulatedBackend(m, seed=i) for i, m in enumerate(
        ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"))}
    gw = make(clock, backends=backends)
    gw.handle(GatewayRequest("Summarize the causes of World War I."))
    for b in backends.values():
        b.failure_rate = 1.0
    r = gw.handle(GatewayRequest("Summarize what caused World War I."))
    assert r.ok and r.cached and r.degraded and r.attempts
    r2 = gw.handle(GatewayRequest("What is the capital of Japan?"))
    assert not r2.ok and r2.error == "all_backends_failed"
    assert gw.metrics["errors"] == 1


def test_mockllm_backends_satisfy_protocol(clock):
    llm = MockLLM(model="claude-haiku-5-5", default="Paris.")
    gw = Gateway({"claude-haiku-5-5": llm}, clock=clock, cache=SemanticCache(clock=clock))
    r = gw.handle(GatewayRequest("What is the capital of France?", system="Be brief."))
    assert r.ok and r.text == "Paris." and r.cost_usd > 0
    assert llm.calls[0]["system"] == "Be brief."
    # a complex prompt routes to Opus, which is absent: chain filters to available backends
    r2 = gw.handle(GatewayRequest(HARD))
    assert not r2.ok or r2.model == "claude-haiku-5-5"


def test_custom_router_and_baseline_cost(clock):
    gw = make(clock, router=Router({"simple": "claude-sonnet-5-5",
                                    "moderate": "claude-sonnet-5-5",
                                    "complex": "claude-sonnet-5-5"}))
    assert gw.handle(GatewayRequest(HARD)).model == "claude-sonnet-5-5"
    assert gw.router.route("complex", max_tier="simple") == "claude-sonnet-5-5"
    assert Router().route("complex", max_tier="moderate") == "claude-sonnet-5-5"
    assert gw.baseline_cost("hello world", 100) == pytest.approx((4 * 4 + 100 * 20) / 1e6)


def test_benchmark_targets():
    m = benchmark(n=300)
    json.dumps(m)
    assert m["main"]["savings_vs_baseline_pct"] >= 40
    assert m["main"]["false_hit_rate"] <= 0.02
    assert m["main"]["cache_hit_rate"] > 0.2
    assert m["degradation"]["success_rate"] >= 0.98
    ks = m["kill_switch"]
    assert ks["initech_rejected"] > 0
    assert ks["initech_spend_usd"] < ks["initech_daily_cap_usd"] * 1.5
