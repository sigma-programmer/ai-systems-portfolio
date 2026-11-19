"""Unit tests: pricing, complexity analysis, semantic cache guards, budgets, breakers, backends."""
from __future__ import annotations

import pytest

from semantic_gateway import (HAIKU, OPUS, SONNET, BackendTimeout, BackendUnavailable,
                              CircuitBreaker, ComplexityAnalyzer, CostTracker, ManualClock,
                              NullCache, RateLimited, SemanticCache, SimulatedBackend,
                              SpendRateBreaker, TenantBudget, cost_of, guard_reason)
from semantic_gateway.cache import slot_substitution
from semantic_gateway.workload import INTENTS, generate


# ---------------------------------------------------------------- pricing
def test_prices_per_mtok():
    assert cost_of("claude-haiku-5-5", 1_000_000, 0) == pytest.approx(0.10)
    assert cost_of("claude-haiku-5-5", 0, 1_000_000) == pytest.approx(0.50)
    assert SONNET.cost(1_000_000, 1_000_000) == pytest.approx(12.0)
    assert OPUS.cost(1_000_000, 1_000_000) == pytest.approx(24.0)
    assert HAIKU.tier == "simple"
    with pytest.raises(KeyError):
        cost_of("gpt-x", 1, 1)


# ---------------------------------------------------------------- complexity
@pytest.fixture(scope="module")
def analyzer():
    return ComplexityAnalyzer()


@pytest.mark.parametrize("prompt,tier", [
    ("What is the capital of France?", "simple"),
    ("Convert 5 km to miles.", "simple"),
    ("Summarize the main causes of World War I in a paragraph.", "moderate"),
    ("Draft a polite email declining a meeting.", "moderate"),
    ("Prove that the square root of 2 is irrational, step by step.", "complex"),
    ("Design a rate limiter architecture and analyze the trade-offs of token bucket "
     "versus sliding window algorithms.", "complex"),
])
def test_complexity_tiers(analyzer, prompt, tier):
    assert analyzer.analyze(prompt).tier == tier


def test_complexity_features(analyzer):
    r = analyzer.analyze("```python\ndef f(x):\n    return x\n```\nWhy?")
    assert r.code_present
    multi = analyzer.analyze("1. What is X?\n2. What is Y?\n3. What is Z?")
    assert multi.parts >= 3
    assert analyzer.analyze("").score == 0.0
    assert ComplexityAnalyzer.entropy(["a", "a", "b", "b"]) == pytest.approx(1.0)
    assert ComplexityAnalyzer.entropy([]) == 0.0
    long_easy = analyzer.analyze("Please read this. " * 60)
    assert long_easy.tier != "complex"
    with pytest.raises(ValueError):
        ComplexityAnalyzer(0.6, 0.5)


def test_routing_accuracy_on_workload_labels(analyzer):
    pairs = [(i.tier, analyzer.analyze(p).tier) for i in INTENTS for p in i.paraphrases]
    acc = sum(a == b for a, b in pairs) / len(pairs)
    assert acc >= 0.9


# ---------------------------------------------------------------- cache guards
@pytest.mark.parametrize("a,b,reason", [
    ("Convert 5 km to miles.", "Convert 8 km to miles.", "number"),
    ("Which foods are safe for dogs?", "Which foods are not safe for dogs?", "negation"),
    ("How do I increase the font size?", "How do I decrease the font size?", "antonym"),
    ("Write a Go function that validates emails.",
     "Write a Rust function that validates emails.", "slot"),
    ("What is the capital of Spain?", "What is the capital of Peru?", "slot"),
])
def test_guard_rejects_lookalikes(a, b, reason):
    r = guard_reason(a, b)
    assert r is not None and reason in r


@pytest.mark.parametrize("a,b", [
    ("What is the capital of France?", "Tell me the capital city of France."),
    ("How do I increase the font size in VS Code?", "How can I increase font size in VS Code?"),
    ("Which foods are not safe for dogs?", "What foods are not safe for dogs to eat?"),
])
def test_guard_allows_paraphrases(a, b):
    assert guard_reason(a, b) is None


def test_slot_substitution_helper():
    assert slot_substitution("a b c", "a b c") is None
    assert slot_substitution("write a go function now", "write a rust function now") == (
        {"go"}, {"rust"})
    assert slot_substitution("totally different words", "nothing alike here") is None


def test_cache_paraphrase_hit_and_guarded_miss():
    clock = ManualClock()
    c = SemanticCache(clock=clock)
    c.put("t", "What is the capital of France?", "Paris", "m")
    hit = c.get("t", "Which city is the capital of France?")
    assert hit and hit.entry.response == "Paris" and not hit.exact
    assert c.get("t", "what is the capital of france").exact
    assert c.get("t", "What is the capital of Peru?") is None
    c.put("t", "Convert 5 km to miles.", "3.1", "m")
    assert c.get("t", "Convert 8 km to miles.") is None
    assert c.stats["hits"] == 2 and c.stats["misses"] == 2


def test_cache_tenant_isolation_ttl_and_lru():
    clock = ManualClock()
    c = SemanticCache(ttl=10, max_entries=2, clock=clock)
    c.put("a", "What is photosynthesis?", "x", "m")
    assert c.get("b", "What is photosynthesis?") is None
    clock.advance(11)
    assert c.get("a", "What is photosynthesis?") is None
    assert c.stats["expirations"] == 1
    for p in ("Define gravity.", "Define entropy.", "Define osmosis."):
        c.put("a", p, p, "m")
    assert len(c) == 2 and c.stats["evictions"] == 1
    assert c.get("a", "Define gravity.") is None


def test_cache_lru_recency_is_refreshed_by_get():
    c = SemanticCache(max_entries=2, clock=ManualClock())
    c.put("a", "Define gravity.", "g", "m")
    c.put("a", "Define entropy.", "e", "m")
    c.get("a", "Define gravity.")
    c.put("a", "Define osmosis.", "o", "m")
    assert c.get("a", "Define gravity.") is not None
    assert c.get("a", "Define entropy.") is None


def test_cache_nearest_fallback_and_null_cache():
    c = SemanticCache(clock=ManualClock())
    c.put("t", "Summarize the causes of World War I.", "S", "m")
    assert c.nearest("t", "Briefly summarize the causes of World War I.", floor=0.5)
    assert c.nearest("t", "Convert 5 km to miles.") is None
    n = NullCache()
    n.put("t", "x", "y", "m")
    assert n.get("t", "x") is None and n.nearest("t", "x") is None


# ---------------------------------------------------------------- budgets & breakers
def test_cost_tracker_soft_and_hard_caps_reset_daily():
    clock = ManualClock()
    tr = CostTracker(budgets={"t": TenantBudget(1.0, 0.8)}, clock=clock)
    assert tr.decide("t") == "ok"
    tr.record("t", 0.85)
    assert tr.decide("t") == "downgrade"
    tr.record("t", 0.2)
    assert tr.decide("t") == "reject"
    assert tr.decide("other") == "ok"  # no budget configured
    clock.advance(86_400)
    assert tr.decide("t") == "ok" and tr.spent_today("t") == 0
    assert tr.total_usd == pytest.approx(1.05)


def test_spend_rate_breaker_trips_and_recovers():
    clock = ManualClock()
    br = SpendRateBreaker(1.0, window=60, cooldown=100, clock=clock)
    tr = CostTracker(rate_breaker=br, clock=clock)
    tr.record("t", 0.6)
    assert not br.is_open()
    tr.record("t", 0.6)
    assert br.is_open() and br.trips == 1
    assert tr.decide("t") == "downgrade"
    clock.advance(101)
    assert not br.is_open() and br.rate() == 0.0
    assert tr.decide("t") == "ok"


def test_circuit_breaker_state_machine():
    clock = ManualClock()
    cb = CircuitBreaker(failure_threshold=2, cooldown=10, clock=clock)
    cb.failure()
    assert cb.state == "closed"
    cb.failure()
    assert cb.state == "open" and not cb.allow() and cb.trips == 1
    clock.advance(10)
    assert cb.state == "half_open" and cb.allow()
    cb.failure()  # probe fails -> re-open
    assert cb.state == "open" and cb.trips == 2
    clock.advance(10)
    cb.success()
    assert cb.state == "closed" and cb.failures == 0


# ---------------------------------------------------------------- backends
def test_simulated_backend_is_deterministic_and_fails_as_configured():
    a = SimulatedBackend("m", seed=1)
    b = SimulatedBackend("m", seed=1)
    ra, rb = a.complete("hello there"), b.complete("hello there")
    assert ra.text == rb.text and a.last_latency_ms == b.last_latency_ms > 0
    assert ra.output_tokens > 0 and ra.model == "m"
    with pytest.raises(BackendTimeout):
        SimulatedBackend("m", timeout_rate=1.0).complete("x")
    with pytest.raises(RateLimited):
        SimulatedBackend("m", rate_limit_rate=1.0).complete("x")
    with pytest.raises(BackendUnavailable):
        SimulatedBackend("m", failure_rate=1.0).complete("x")
    o = SimulatedBackend("m")
    o.outage(1, 2)
    o.complete("x")
    with pytest.raises(BackendUnavailable):
        o.complete("x")
    o.complete("x")
    with pytest.raises(BackendTimeout):
        SimulatedBackend("m", base_latency_ms=50_000).complete("x")


def test_workload_is_deterministic_and_mixed():
    w1, w2 = generate(200, seed=3), generate(200, seed=3)
    assert w1 == w2
    assert {w.tier for w in w1} == {"simple", "moderate", "complex"}
    assert len({w.tenant for w in w1}) == 3
    assert all(b.t > a.t for a, b in zip(w1, w1[1:]))
