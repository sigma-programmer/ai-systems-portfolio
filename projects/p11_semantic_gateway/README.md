# P11 — Semantic Cache + Routing Gateway

> You can slash inference costs by 40-60%.

## Problem
Most production LLM traffic is cheap: lookups, short rewrites, repeated questions phrased
slightly differently. Sending all of it to the most capable model, with no caching, wastes most of
the bill. The obvious fixes bring their own risks. A semantic cache can hand back the wrong answer
("Convert 8 km to miles" answered with the cached 5 km result). A router can starve hard prompts.
And with no spend controls, one runaway tenant or agent loop can burn a month's budget in an
hour. This gateway applies all three cost levers and puts guards on each.

## Architecture
```
 GatewayRequest(prompt, tenant)
        │
        ▼
 ┌───────────────┐  reject (hard daily cap) ──► cache.nearest() (flagged degraded) or error
 │ CostTracker   │  downgrade (≥80% of cap, or spend-rate breaker open) → cheapest tier only
 └──────┬────────┘
        ▼
 ┌───────────────┐  exact hit  ─┐
 │ SemanticCache │  cosine ≥ 0.75 + guards (numbers · negation · antonyms ·
 │ per tenant,   │  template-slot substitution · content overlap) ──► hit: $0
 │ TTL + LRU     │
 └──────┬────────┘ miss
        ▼
 ┌────────────────────┐ entropy · length · rare words · reasoning cues · code ·
 │ ComplexityAnalyzer │ sub-question count · generative cues  → score → tier
 └──────┬─────────────┘
        ▼
 ┌────────┐ simple→claude-haiku-5-5  moderate→claude-sonnet-5-5  complex→claude-opus-5-5
 │ Router │ + pre-flight cost estimate vs. remaining cap (downgrade or reject)
 └──────┬─┘
        ▼
 degradation chain, e.g. opus → sonnet → haiku. Each backend has a circuit breaker
 (closed → open after 3 consecutive failures → half-open after cooldown); open ones are skipped.
        │ all failed
        ▼
 cache.nearest() with a looser floor (degraded) ──► else graceful "all_backends_failed"
```
Backends implement the `aikit.LLM` protocol, so a `ClaudeLLM` or `MockLLM` drops in directly.
`SimulatedBackend` is a seeded simulator with a latency model (base + per-token + log-normal
jitter) and configurable timeout, rate-limit and 5xx rates, plus deterministic outage windows.
Its latency is virtual: reported, never slept.

| Model | $/MTok in | $/MTok out | Tier |
|---|---|---|---|
| claude-haiku-5-5 | 0.10 | 0.50 | simple |
| claude-sonnet-5-5 | 2.00 | 10.00 | moderate |
| claude-opus-5-5 | 4.00 | 20.00 | complex |

### Complexity score (`complexity.py`)
```
score = 0.12·min(H/7,1) + 0.13·log(1+tokens)/log(401) + 0.10·min(4·rare,1)
      + 0.35·min(reasoning_hits/2,1) + 0.18·code + 0.12·min((parts−1)/3,1) + 0.20·generative
tier  = simple < 0.30 ≤ moderate < 0.50 ≤ complex
```

## Key design decisions & tradeoffs
* **Reasoning cues outweigh length and entropy.** If length or entropy dominated, long but easy
  prompts ("summarise this paragraph") would land on Opus. Explicit cues ("prove", "derive",
  "step by step", "trade-off", code) are the strongest sign that a small model will fail.
* **The cache guards are what make a lower threshold safe.** Hash embeddings score
  "Write a Go function that …" and "Write a Rust function that …" at 0.83 similarity. The guards
  reject these pairs no matter how similar they look: different number sets, different negation
  parity, antonym flips (increase/decrease), and template-slot substitution (a shared
  prefix+suffix with different filler words). With the guards in place, the threshold could drop
  from 0.82 to 0.75. That caught more paraphrases and still produced zero false hits.
* **Tenant isolation is the cache namespace.** One tenant's cached answer never reaches another.
* **Kill-switches act before the call, not after.** A post-hoc cap overshoots by one request,
  which matters when that request is an Opus call. The pre-flight estimate (output ≈ 30 + 0.8 ×
  input tokens) first tries downgrading to Haiku and rejects only if Haiku also wouldn't fit. While
  cost-capped, the fallback chain never escalates to a pricier model.
* **Degradation order:** fallback model → nearest cached answer, marked `degraded=True` → a
  graceful error. Circuit breakers stop the gateway from paying the failure latency of a dead
  backend on every request.
* **Tradeoffs:**
  * "Explain why TCP and UDP differ" goes to Opus: "explain why" counts as a reasoning cue. Routing
    errors on this workload all go toward the more expensive model, which is the safe direction.
  * Paraphrases with little word overlap ("At what temperature does water boil?" vs. "boiling
    point of water") miss the cache. A neural embedder would raise the hit rate, and the guards
    would still apply.
  * Breaker "trips" count every reopening, including failed half-open probes, so a long outage
    shows many trips.
  * The workload is synthetic, so these numbers show how the mechanisms behave. They are not a
    prediction of savings on real traffic.

## How to run
```bash
AIKIT_OFFLINE=1 .venv/bin/python projects/p11_semantic_gateway/demo.py
.venv/bin/python -m pytest projects/p11_semantic_gateway
cd projects/p11_semantic_gateway && PYTHONPATH=../../common:. ../../.venv/bin/python -m semantic_gateway.bench
```
```python
from semantic_gateway import Gateway, GatewayRequest, default_backends
gw = Gateway(default_backends())          # or {"claude-opus-5-5": aikit.ClaudeLLM(), ...}
r = gw.handle(GatewayRequest("Prove that sqrt(2) is irrational, step by step.", tenant="acme"))
print(r.model, r.cost_usd, r.cached, r.attempts)
```

## Benchmark (measured, deterministic, ~0.3 s)
The workload has 500 requests across 3 tenants with Poisson arrivals and 161 distinct intents.
About half is a Zipf-popular head: 29 intents with paraphrases, plus numeric, negation and antonym
traps. The other half is a templated long tail, which supplies the same-template sibling traps.
Tier mix: 243 simple, 174 moderate, 83 complex. The baseline sends every request to Opus with no
cache and is priced on identical token accounting.

**Main run** (background failure rate of 1% per backend):

| Metric | Value |
|---|---|
| Gateway cost vs. always-Opus baseline | $0.1341 vs. $0.5094 |
| **Savings** | **73.7%** (target 40–60%+) |
| Router-only savings (cache off) | 58.5% |
| Cache-only savings (always Opus + cache) | 46.8% |
| Cache hit rate | 47.4% |
| **False-hit rate** | **0.0%** (0 of 237 hits) |
| Routing distribution (cache misses) | Haiku 117 · Sonnet 77 · Opus 69 |
| Routing accuracy vs. tier labels | 99.6% |
| Latency p50 / p95 / p99 (virtual) | 317 / 1625 / 2006 ms (cache hits ≈ 2 ms) |
| Success rate | 100% |

**Degradation run** (Opus down for 50 consecutive calls, Sonnet at 10% errors, Haiku at 3% rate
limits):

| Metric | Value |
|---|---|
| Success rate | 99.8% (1 hard error) |
| Fallbacks served by the next model in the chain | 70 |
| Opus calls skipped because its breaker was open | 45 (Opus breaker reopened 21 times) |
| Attempt outcomes | 263 ok · 38 unavailable · 1 rate-limited · 45 skipped |
| Latency p50 / p95 | 305 / 1148 ms (fallbacks land on faster models) |

**Kill-switch run** (tenant `initech` has a $0.005/day cap; global spend-rate breaker at
$0.012 per 60 s):

| Metric | Value |
|---|---|
| initech spend vs. cap | $0.004996 / $0.005 (the cap held, no overshoot) |
| initech requests rejected (pre-flight or hard cap) | 44 of 104 (1 served from cache) |
| Requests downgraded to Haiku (soft cap, pre-flight, or rate breaker) | 114 |
| Spend-rate breaker trips | 2 |
