"""Benchmark: replay a ~500-request synthetic workload through the gateway.

Scenarios (all deterministic, virtual latency):

1. **main** - full gateway (routing + semantic cache) with light background failure
   rates, vs an always-Opus/no-cache baseline priced on identical token counts; plus
   ablations "router only" and "cache only (always Opus)".
2. **degradation** - Opus hard outage for 40 consecutive calls and Sonnet 10% errors:
   fallback chains, breaker trips, success rate.
3. **kill-switch** - a tenant with a tiny daily cap and a global spend-rate breaker:
   downgrade/reject decisions and whether the cap holds.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter

from .backends import SimulatedBackend, default_backends
from .budget import CostTracker, SpendRateBreaker, TenantBudget
from .cache import NullCache, SemanticCache
from .clock import ManualClock
from .gateway import Gateway, GatewayRequest, GatewayResponse, Router
from .pricing import OPUS
from .workload import WorkItem, generate


def _pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 1) if s else 0.0


def _replay(gw: Gateway, clock: ManualClock, work: list[WorkItem]
            ) -> list[tuple[WorkItem, GatewayResponse]]:
    out = []
    for w in work:
        clock.t = w.t
        out.append((w, gw.handle(GatewayRequest(w.prompt, w.tenant), meta={"intent": w.intent})))
    return out


def _baseline_cost(work: list[WorkItem]) -> float:
    """Always Opus, no cache: same token accounting as the gateway path."""
    opus = SimulatedBackend(OPUS.model_id, verbosity=1.2)
    total = 0.0
    for w in work:
        r = opus.complete(w.prompt)
        total += OPUS.cost(r.input_tokens, r.output_tokens)
    return total


def _summarise(rows: list[tuple[WorkItem, GatewayResponse]]) -> dict:
    ok = [r for _, r in rows if r.ok]
    hits = [(w, r) for w, r in rows if r.cached and not r.degraded]
    false_hits = [(w, r) for w, r in hits
                  if (r.cache_source_meta or {}).get("intent") != w.intent]
    routed = [(w, r) for w, r in rows if not r.cached and r.tier]
    lat = [r.latency_ms for _, r in rows]
    return {
        "requests": len(rows),
        "success_rate": round(len(ok) / len(rows), 4),
        "cost_usd": round(sum(r.cost_usd for _, r in rows), 6),
        "cache_hit_rate": round(len(hits) / len(rows), 4),
        "false_hit_rate": round(len(false_hits) / len(hits), 4) if hits else 0.0,
        "false_hits": len(false_hits),
        "routing_distribution": dict(Counter(r.model for _, r in routed if r.model)),
        "routing_accuracy_vs_labels": round(
            sum(w.tier == r.tier for w, r in routed) / len(routed), 4) if routed else 0.0,
        "latency_ms": {"p50": _pct(lat, 50), "p95": _pct(lat, 95), "p99": _pct(lat, 99)},
    }


def _gateway(clock: ManualClock, *, cache=None, router=None, backends=None, tracker=None
             ) -> Gateway:
    return Gateway(backends or default_backends(seed=5, failure_rate=0.01),
                   cache=cache if cache is not None else SemanticCache(clock=clock, ttl=3600),
                   router=router, tracker=tracker or CostTracker(clock=clock), clock=clock)


def benchmark(n: int = 500, seed: int = 11) -> dict:
    work = generate(n, seed)
    baseline = _baseline_cost(work)

    clock = ManualClock()
    main = _summarise(_replay(_gateway(clock), clock, work))
    main["baseline_always_opus_usd"] = round(baseline, 6)
    main["savings_vs_baseline_pct"] = round(100 * (1 - main["cost_usd"] / baseline), 2)

    c2 = ManualClock()
    router_only = _summarise(_replay(_gateway(c2, cache=NullCache()), c2, work))
    c3 = ManualClock()
    opus_router = Router({t: OPUS.model_id for t in ("simple", "moderate", "complex")})
    cache_only = _summarise(_replay(_gateway(c3, router=opus_router), c3, work))
    ablations = {
        "router_only_savings_pct": round(100 * (1 - router_only["cost_usd"] / baseline), 2),
        "cache_only_savings_pct": round(100 * (1 - cache_only["cost_usd"] / baseline), 2),
    }

    # ---- degradation under injected failures
    c4 = ManualClock()
    backends = default_backends(seed=9)
    backends["claude-opus-5-5"].outage(10, 60)
    backends["claude-sonnet-5-5"].failure_rate = 0.10
    backends["claude-haiku-5-5"].rate_limit_rate = 0.03
    gw = _gateway(c4, backends=backends)
    deg_rows = _replay(gw, c4, work)
    degradation = _summarise(deg_rows)
    degradation.update({
        "fallbacks": gw.metrics["fallbacks"],
        "served_from_cache_after_failures": gw.metrics["degraded_cache"],
        "hard_errors": gw.metrics["errors"],
        "breaker_trips": {m: b.trips for m, b in gw.breakers.items()},
        "attempt_outcomes": dict(Counter(a["outcome"] for _, r in deg_rows for a in r.attempts)),
    })

    # ---- cost kill-switches
    c5 = ManualClock()
    cap = 0.005
    tracker = CostTracker(budgets={"initech": TenantBudget(daily_cap_usd=cap)},
                          rate_breaker=SpendRateBreaker(max_usd_per_window=0.012, window=60,
                                                        cooldown=120, clock=c5), clock=c5)
    gw5 = _gateway(c5, tracker=tracker)
    ks_rows = _replay(gw5, c5, work)
    initech = [r for w, r in ks_rows if w.tenant == "initech"]
    killswitch = {
        "initech_daily_cap_usd": cap,
        "initech_spend_usd": round(tracker.spent_today("initech"), 6),
        "initech_requests": len(initech),
        "initech_rejected": sum(r.decision == "reject" for r in initech),
        "initech_rejected_but_served_from_cache": sum(r.decision == "reject" and r.ok
                                                       for r in initech),
        "tracker_decisions_pre_routing": tracker.decisions,
        "gateway_rejected_total": gw5.metrics["rejected"],
        "gateway_downgraded_total": gw5.metrics["downgraded"],
        "rate_breaker_trips": tracker.rate_breaker.trips if tracker.rate_breaker else 0,
        "total_cost_usd": round(tracker.total_usd, 6),
    }

    return {"workload": {"requests": n, "distinct_intents": len({w.intent for w in work}),
                         "tier_mix": dict(Counter(w.tier for w in work))},
            "main": main, "ablations": ablations, "degradation": degradation,
            "kill_switch": killswitch}


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
