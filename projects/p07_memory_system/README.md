# P07 — 3-Tier Agent Memory System

> You can build agents that truly remember.

## Problem
A context window is not a memory. Stuffing the whole chat history in fails on cost and length;
"embed every message into a vector store" fails on *truth*: when a user says "I live in Berlin"
and three weeks later "I moved to Lisbon", a flat vector store happily returns both, and the
older one often ranks higher. Real agent memory needs (a) a bounded working set, (b) compressed
episodic history, (c) distilled facts with belief revision, (d) forgetting, and (e) persistence
across processes.

## Architecture
```
 observe(turn)
     │
     ▼
 ┌──────────────────────── WORKING (token-budgeted scratchpad) ───────────────────────┐
 │ over budget → look at oldest 50% (FIFO pressure) → evict LOWEST importance item    │
 └───────────────┬───────────────────────────────────────────────────────────────────┘
                 │ evicted / idle-TTL / end_session   (nothing is silently dropped)
                 ▼
 ┌──────────────────────── EPISODIC (compressed windows, timestamped) ────────────────┐
 │ ExtractiveSummarizer: 0.5·centrality + 0.5·salience, budget = 40% of raw tokens   │
 │ (or LLMSummarizer via aikit.LLM)          TTL 30d unless static importance ≥ 0.5  │
 └───────────────┬───────────────────────────────────────────────────────────────────┘
                 │ consolidate (episodic → semantic), raw turns then dropped
                 ▼
 ┌──────────────────────── SEMANTIC (facts / preferences) ────────────────────────────┐
 │ PatternFactExtractor ("my name is", "I work at", "I prefer X over Y",             │
 │ "remember that", "I no longer…") or LLMFactExtractor                               │
 │ upsert: duplicate→reinforce · single-valued→newer supersedes · likes⇄dislikes      │
 │ flip → newer wins · retraction → supersede; superseded purged after 14d            │
 └────────────────────────────────────────────────────────────────────────────────────┘
          all tiers write-through to SQLite (sessions, working, episodes, facts, counters)

 recall(q) = Σ tiers, rank by  w_tier · cos(q,m) · (0.4+0.6·static_importance) · (0.5+0.5·recency)
```

### Importance (documented formula, `importance.py`)
```
salience   = 0.25 + 0.35·explicit cue ("remember", "important", "don't forget"…)
                  + 0.20·personal-fact pattern + 0.15·emotional cue + 0.05·number − 0.15·filler
recency    = 2^(−age / half_life_tier)      half-lives: working 30 min, episodic 7 d, semantic 90 d
frequency  = 1 − exp(−access_count / 3)     (recall reinforces what it returns)
importance = 0.45·salience + 0.30·recency + 0.25·frequency       (eviction decisions)
static     = (0.45·salience + 0.25·frequency) / 0.70              (recall, × recency separately)
```

## Key design decisions & tradeoffs
* **Eviction = FIFO window × importance.** Pure FIFO drops the "remember my flight is Friday"
  turn; pure importance lets recent context get evicted. Restricting candidates to the oldest half
  and evicting the least important among them keeps both recency and salience.
* **Evicted ≠ deleted.** Overflow is promoted into episodes; facts are extracted *before* raw
  turns are discarded, so compression (≈0.37 of raw tokens) loses chit-chat, not facts.
* **Belief revision over append-only.** Single-valued predicates (name, employer, location,
  `favorite_*`, `*_name`) supersede; multi-valued ones (likes, allergy, note) dedup. Superseded
  facts are hidden from recall by default but kept (with `superseded_by`) for audit until TTL.
* **Multiplicative recall with floors.** Similarity is the gate; importance and recency modulate
  between 0.4–1.0× and 0.5–1.0×, so a 30-day-old fact that is exactly on topic still wins.
* **Boilerplate stripping for embeddings.** "The user's …" appears in every fact and every probe;
  removing speaker/user tokens before hashing-embedding lifted precision@1 from 0.67 to 0.83.
* **Deterministic, injectable clock.** All TTL/decay logic reads `clock.now()`; tests use
  `ManualClock` to jump 40 days in one line.
* **SQLite write-through, no stored vectors.** `HashingEmbedder` is deterministic, so vectors are
  recomputed on load; a crash mid-session leaves working items in the DB which the next
  `start_session` recovers into an episode.
* **Tradeoffs:** regex extraction is precise but low-recall (it will not catch "Lisbon has been home
  since May") — `LLMFactExtractor` is the drop-in fix. Hash embeddings confuse "name of the dog" with
  "the user's name" (the one probe it misses at rank 1).

## How to run
```bash
AIKIT_OFFLINE=1 .venv/bin/python projects/p07_memory_system/demo.py
.venv/bin/python -m pytest projects/p07_memory_system
cd projects/p07_memory_system && PYTHONPATH=../../common:. ../../.venv/bin/python -m memory_system.bench
```
```python
from memory_system import MemoryManager
mm = MemoryManager("agent.db"); mm.start_session()
mm.observe("My name is Ada and I prefer tea over coffee."); mm.end_session()
MemoryManager("agent.db").recall("what does the user drink?")   # new instance, same memories
```
Tests prove persistence across instances *and* across OS processes (a `subprocess` reads the DB).

## Benchmark (measured)
4 scripted sessions over 20 simulated days (38 turns, 12 planted facts incl. a city move and a
job change, plus small talk); 12 probes asked on day 35 by a **fresh MemoryManager on the same
SQLite file**. Baseline = flat vector store of all raw user turns (cosine only).

| Metric | 3-tier memory | Flat vector store |
|---|---|---|
| precision@1 | **0.833** | 0.417 |
| precision@3 | 0.389 | 0.278 |
| recall@3 | **1.000** | 0.750 |
| MRR@5 | **0.903** | 0.590 |
| stale value outranks current (Berlin/Nimbus) | **0 / 2** | 2 / 2 |

| Other | Value |
|---|---|
| Episodic compression ratio (summary/raw tokens) | 0.367 |
| Working evictions promoted to episodic | 18 |
| Episodes expired by TTL at day 35 | 3 (of 5; salient ones kept) |
| Superseded facts purged | 2 |
| Fact actions | 12 inserted, 1 superseded, 1 retracted |
| Latency | observe ≈0.4–0.7 ms (SQLite commit incl.), recall ≈0.2 ms |
