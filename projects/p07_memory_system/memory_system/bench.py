"""Benchmark: recall quality across sessions, compression, eviction, latency.

Replays `scenario.SESSIONS` (4 sessions over 20 simulated days, planted facts,
updates such as a move and a job change, plus small talk) into a SQLite file,
then opens a *fresh* `MemoryManager` on the same file at day 35 and asks the
`PROBES`. Compared against a naive baseline: a flat vector store of every raw
user turn ranked by cosine similarity (no tiers, no supersession, no
importance) — the usual "just embed the chat log" approach.
"""
from __future__ import annotations

import json
import statistics
import tempfile
import time
from pathlib import Path

from aikit import HashingEmbedder

from .clock import DAY, ManualClock
from .manager import MemoryConfig, MemoryManager
from .scenario import PROBES, QUERY_DAY, SESSIONS, Probe


def _relevant(text: str, probe: Probe) -> bool:
    return any(e.lower() in text.lower() for e in probe.expected)


def _stale(text: str, probe: Probe) -> bool:
    return any(s.lower() in text.lower() for s in probe.stale) and not _relevant(text, probe)


def _rank_metrics(ranked: list[list[str]], k: int = 3) -> dict[str, float]:
    p1 = p_k = r_k = mrr = stale = 0.0
    for texts, probe in zip(ranked, PROBES):
        rel = [_relevant(t, probe) for t in texts]
        p1 += rel[0] if rel else 0
        p_k += sum(rel[:k]) / k
        r_k += any(rel[:k])
        first = next((i for i, r in enumerate(rel) if r), None)
        mrr += 1 / (first + 1) if first is not None else 0
        first_stale = next((i for i, t in enumerate(texts) if _stale(t, probe)), None)
        if probe.stale and first_stale is not None and (first is None or first_stale < first):
            stale += 1
    n = len(PROBES)
    n_stale = sum(1 for p in PROBES if p.stale)
    return {"precision@1": round(p1 / n, 4), f"precision@{k}": round(p_k / n, 4),
            f"recall@{k}": round(r_k / n, 4), "mrr@5": round(mrr / n, 4),
            "stale_outranks_current": f"{int(stale)}/{n_stale}"}


def replay(db_path: str | Path, clock: ManualClock, config: MemoryConfig | None = None
           ) -> tuple[MemoryManager, list[float]]:
    """Write every scripted session into a MemoryManager; returns it and observe latencies."""
    mm = MemoryManager(db_path, clock=clock, config=config)
    start = clock.now()
    lat: list[float] = []
    for i, (day, turns) in enumerate(SESSIONS):
        clock.t = start + day * DAY
        mm.start_session(f"session-{i + 1}")
        for role, text in turns:
            clock.advance(45)
            t0 = time.perf_counter()
            mm.observe(text, role)
            lat.append((time.perf_counter() - t0) * 1000)
        mm.end_session()
        mm.evict()
    return mm, lat


def benchmark() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "memory.db"
        clock = ManualClock()
        start = clock.now()
        writer, observe_lat = replay(db, clock)
        write_stats = writer.stats()
        writer.close()

        # New process / new instance on the same file: cross-session recall.
        clock.t = start + QUERY_DAY * DAY
        reader = MemoryManager(db, clock=clock)
        reader.start_session("session-query")
        evict_counts = reader.evict()
        recall_lat: list[float] = []
        ranked: list[list[str]] = []
        tier_of_top: dict[str, int] = {}
        for probe in PROBES:
            t0 = time.perf_counter()
            hits = reader.recall(probe.query, k=5, reinforce=False)
            recall_lat.append((time.perf_counter() - t0) * 1000)
            ranked.append([h.text for h in hits])
            if hits:
                tier_of_top[hits[0].tier] = tier_of_top.get(hits[0].tier, 0) + 1
        profile = reader.profile()
        final_stats = reader.stats()
        reader.close()

    # Baseline: flat vector store over every raw user turn.
    emb = HashingEmbedder()
    turns = [t for _, ts in SESSIONS for r, t in ts if r == "user"]
    mat = emb.embed(turns)
    base_ranked = []
    for probe in PROBES:
        sims = mat @ emb.embed_one(probe.query)
        base_ranked.append([turns[i] for i in sims.argsort()[::-1][:5]])

    raw_tokens_total = sum(len(t.split()) for t in turns)
    return {
        "sessions": len(SESSIONS), "turns": sum(len(t) for _, t in SESSIONS),
        "probes": len(PROBES),
        "memory_system": _rank_metrics(ranked),
        "baseline_flat_vector_store": _rank_metrics(base_ranked),
        "top1_tier_distribution": tier_of_top,
        "compression_ratio_episodic": final_stats["compression_ratio"],
        "episodes": final_stats["episodes"],
        "facts_active": final_stats["facts_active"],
        "facts_superseded": final_stats["facts_superseded"],
        "profile_location": profile.get("location"),
        "profile_employer": profile.get("employer"),
        "eviction_counts": {
            "working_evictions_to_episodic": write_stats["counters"].get("working_evictions", 0),
            "working_expired_ttl": final_stats["counters"].get("working_expired", 0),
            "episodes_expired_ttl": final_stats["counters"].get("episodes_expired", 0),
            "superseded_facts_purged": final_stats["counters"].get("facts_purged", 0),
            "query_time_eviction": evict_counts,
        },
        "fact_actions": {k[6:]: v for k, v in final_stats["counters"].items()
                         if k.startswith("facts_") and k != "facts_purged"},
        "raw_user_words": raw_tokens_total,
        "latency_ms": {
            "observe_mean": round(statistics.fmean(observe_lat), 3),
            "observe_p95": round(sorted(observe_lat)[int(0.95 * (len(observe_lat) - 1))], 3),
            "recall_mean": round(statistics.fmean(recall_lat), 3),
            "recall_max": round(max(recall_lat), 3),
        },
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
