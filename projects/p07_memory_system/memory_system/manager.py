"""MemoryManager: the facade tying the three tiers, persistence and recall together.

Data flow::

    observe(turn) ─► WorkingMemory (token budget) ──evict (FIFO window, lowest importance)──┐
                                                                                           ▼
                        end_session / batch full ─► Episode (extractive or LLM summary) ─► SQLite
                                                           │ consolidate
                                                           ▼
                                   SemanticMemory (facts, dedup, newer supersedes older) ─► SQLite

Recall score for every candidate memory m and query q::

    score = tier_weight · max(cos(q, m), 0) · (0.4 + 0.6 · static_importance(m))
                                            · (0.5 + 0.5 · recency(m, tier half-life))

The floors (0.4, 0.5) keep similarity primary: an old, rarely used fact that
is exactly on-topic still beats fresh, salient but irrelevant chatter.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from aikit import HashingEmbedder, count_tokens

from .clock import DAY, HOUR, Clock, SystemClock
from .episodic import ExtractiveSummarizer, Summarizer
from .importance import ImportanceScorer, recency, salience_of
from .models import Episode, Fact, RecallHit, WorkingItem, new_id
from .semantic import FactExtractor, PatternFactExtractor, SemanticMemory
from .store import SQLiteStore
from .working import WorkingMemory

TIERS = ("semantic", "episodic", "working")
# Words that appear in nearly every memory/query about the user carry no signal for
# recall; stripping them before embedding stops "The user's ..." boilerplate from
# dominating cosine similarity.
_RECALL_NOISE = re.compile(r"\b(?:the user'?s?|users?|my|me|mine|i'm|i|we|our|your|you|"
                           r"assistant)\b:?", re.I)


def recall_text(text: str) -> str:
    """Normalise text for recall embedding (drop speaker/user boilerplate)."""
    return re.sub(r"\s+", " ", _RECALL_NOISE.sub(" ", text)).strip()


@dataclass
class MemoryConfig:
    token_budget: int = 80
    fifo_window: float = 0.5
    episode_max_turns: int = 6
    half_life: dict[str, float] = field(default_factory=lambda: {
        "working": 30 * 60.0, "episodic": 7 * DAY, "semantic": 90 * DAY})
    working_ttl: float = 2 * HOUR
    episodic_ttl: float = 30 * DAY
    episodic_keep_importance: float = 0.5
    superseded_ttl: float = 14 * DAY
    tier_weight: dict[str, float] = field(default_factory=lambda: {
        "semantic": 1.0, "episodic": 0.8, "working": 0.9})
    consolidate_on_close: bool = True
    drop_raw_after_consolidation: bool = True


class MemoryManager:
    def __init__(self, db_path: str | Path = ":memory:", *, clock: Clock | None = None,
                 config: MemoryConfig | None = None, embedder: HashingEmbedder | None = None,
                 summarizer: Summarizer | None = None, extractor: FactExtractor | None = None,
                 scorer: ImportanceScorer | None = None):
        self.clock = clock or SystemClock()
        self.cfg = config or MemoryConfig()
        self.embedder = embedder or HashingEmbedder()
        self.summarizer = summarizer or ExtractiveSummarizer(self.embedder)
        self.extractor = extractor or PatternFactExtractor()
        self.scorer = scorer or ImportanceScorer()
        self.store = SQLiteStore(db_path)
        self.working = WorkingMemory(self.clock, self.cfg.token_budget, self.cfg.fifo_window,
                                     self.cfg.half_life["working"], self.scorer)
        self.working.items = self.store.load_working()
        self.episodes: dict[str, Episode] = {e.id: e for e in self.store.load_episodes()}
        self.semantic = SemanticMemory()
        self.semantic.load(self.store.load_facts())
        self.session_id: str | None = None
        self._pending: list[WorkingItem] = []

    # ------------------------------------------------------------------ sessions
    def start_session(self, session_id: str | None = None) -> str:
        if self.session_id:
            self.end_session()
        sid = session_id or f"s_{uuid.uuid4().hex[:8]}"
        self.session_id = sid
        self.store.start_session(sid, self.clock.now())
        orphans = [i for i in self.working.items if i.session_id != sid]
        if orphans:  # crash recovery: a previous process never closed its session
            self.working.items = [i for i in self.working.items if i.session_id == sid]
            self.store.delete_working([i.id for i in orphans])
            self._close_episode(orphans)
            self.store.bump("recovered_working_items", len(orphans))
        return sid

    def end_session(self) -> Episode | None:
        if not self.session_id:
            return None
        rest = self.working.drain()
        self.store.delete_working([i.id for i in rest])
        ep = self._close_episode(self._pending + rest)
        self._pending = []
        self.store.end_session(self.session_id, self.clock.now())
        self.session_id = None
        return ep

    # ------------------------------------------------------------------ write path
    def observe(self, text: str, role: str = "user", salience: float | None = None
                ) -> WorkingItem:
        """Add a conversational turn to working memory (auto-starting a session)."""
        if not self.session_id:
            self.start_session()
        assert self.session_id is not None
        item, evicted = self.working.add(text, self.session_id, role, salience)
        self.store.save_working(item)
        if evicted:
            self.store.delete_working([i.id for i in evicted])
            self.store.bump("working_evictions", len(evicted))
            self._pending.extend(evicted)
            if len(self._pending) >= self.cfg.episode_max_turns:
                self._close_episode(self._pending)
                self._pending = []
        return item

    def _close_episode(self, items: Sequence[WorkingItem]) -> Episode | None:
        items = sorted(items, key=lambda i: i.created)
        if not items:
            return None
        turns = [f"{i.role}: {i.text}" for i in items]
        summary = self.summarizer.summarize(turns) or turns[-1]
        top = sorted((i.salience for i in items), reverse=True)[:2]
        ep = Episode(new_id("ep"), items[0].session_id, items[0].created, items[-1].created,
                     summary, turns, sum(count_tokens(t) for t in turns), count_tokens(summary),
                     sum(top) / len(top), self.clock.now())
        self.episodes[ep.id] = ep
        self.store.save_episode(ep)
        self.store.bump("episodes_created")
        if self.cfg.consolidate_on_close:
            self._consolidate_episode(ep)
        return ep

    # ------------------------------------------------------------------ consolidation
    def consolidate(self) -> dict[str, int]:
        """Episodic → semantic: extract facts from every unconsolidated episode."""
        totals: dict[str, int] = {}
        for ep in sorted(self.episodes.values(), key=lambda e: e.start):
            if not ep.consolidated:
                for k, v in self._consolidate_episode(ep).items():
                    totals[k] = totals.get(k, 0) + v
        return totals

    def _consolidate_episode(self, ep: Episode) -> dict[str, int]:
        counts: dict[str, int] = {}
        now = self.clock.now()
        for turn in ep.raw_turns:
            role, _, text = turn.partition(": ")
            if role != "user":
                continue
            for ex in self.extractor.extract(text):
                action, fact, changed = self.semantic.upsert(ex, now, ep.id, salience_of(text))
                counts[action] = counts.get(action, 0) + 1
                for f in changed + ([fact] if fact else []):
                    self.store.save_fact(f)
        ep.consolidated = True
        if self.cfg.drop_raw_after_consolidation:
            ep.raw_turns = []
        self.store.save_episode(ep)
        for k, v in counts.items():
            self.store.bump(f"facts_{k}", v)
        return counts

    # ------------------------------------------------------------------ eviction
    def evict(self) -> dict[str, int]:
        """Apply TTLs: idle working items → episodic; stale episodes and dead facts deleted."""
        now = self.clock.now()
        expired = self.working.expire(self.cfg.working_ttl)
        if expired:
            self.store.delete_working([i.id for i in expired])
            self._close_episode(expired)
        dead_eps = [e.id for e in self.episodes.values()
                    if now - e.last_access > self.cfg.episodic_ttl
                    and self.scorer.static(e.salience, e.access_count)
                    < self.cfg.episodic_keep_importance]
        for i in dead_eps:
            del self.episodes[i]
        self.store.delete_episodes(dead_eps)
        dead_facts = [f.id for f in self.semantic.facts.values()
                      if f.status == "superseded" and now - f.last_access > self.cfg.superseded_ttl]
        for i in dead_facts:
            del self.semantic.facts[i]
        self.store.delete_facts(dead_facts)
        counts = {"working_expired": len(expired), "episodes_expired": len(dead_eps),
                  "facts_purged": len(dead_facts)}
        for k, v in counts.items():
            self.store.bump(k, v)
        return counts

    # ------------------------------------------------------------------ recall
    def _candidates(self, tiers: Iterable[str], include_superseded: bool
                    ) -> list[tuple[str, object, str, float, int, float]]:
        out: list[tuple[str, object, str, float, int, float]] = []
        tiers = set(tiers)
        if "semantic" in tiers:
            for f in self.semantic.facts.values():
                if f.status == "active" or include_superseded:
                    out.append(("semantic", f, f.text, f.salience, f.access_count, f.last_access))
        if "episodic" in tiers:
            for e in self.episodes.values():
                out.append(("episodic", e, e.summary, e.salience, e.access_count, e.last_access))
        if "working" in tiers:
            for w in self.working.items:
                out.append(("working", w, w.text, w.salience, w.access_count, w.last_access))
        return out

    def recall(self, query: str, k: int = 5, tiers: Iterable[str] = TIERS,
               include_superseded: bool = False, reinforce: bool = True) -> list[RecallHit]:
        cands = self._candidates(tiers, include_superseded)
        if not cands or not query.strip():
            return []
        now = self.clock.now()
        sims = (self.embedder.embed([recall_text(c[2]) for c in cands])
                @ self.embedder.embed_one(recall_text(query)))
        hits: list[tuple[RecallHit, object]] = []
        for (tier, obj, text, sal, acc, last), sim in zip(cands, np.asarray(sims)):
            imp = self.scorer.static(sal, acc)
            rec = recency(now - last, self.cfg.half_life[tier])
            score = (self.cfg.tier_weight[tier] * max(float(sim), 0.0)
                     * (0.4 + 0.6 * imp) * (0.5 + 0.5 * rec))
            hits.append((RecallHit(tier, obj.id, text, score, float(sim), imp, rec), obj))
        hits.sort(key=lambda h: -h[0].score)
        top = hits[:k]
        if reinforce:
            for hit, obj in top:
                obj.access_count += 1
                obj.last_access = now
                if isinstance(obj, Fact):
                    self.store.save_fact(obj)
                elif isinstance(obj, Episode):
                    self.store.save_episode(obj)
                else:
                    self.store.save_working(obj)  # type: ignore[arg-type]
        return [h for h, _ in top]

    def context(self, query: str, k: int = 5, token_budget: int = 300) -> str:
        """Render recalled memories as a prompt block within a token budget."""
        lines, used = [], 0
        for h in self.recall(query, k):
            line = f"- [{h.tier}] {h.text}"
            cost = count_tokens(line)
            if used + cost > token_budget:
                break
            lines.append(line)
            used += cost
        return "\n".join(lines)

    # ------------------------------------------------------------------ introspection
    def profile(self) -> dict[str, list[str] | str]:
        return self.semantic.profile()

    def stats(self) -> dict[str, object]:
        eps = list(self.episodes.values())
        raw = sum(e.raw_tokens for e in eps)
        summ = sum(e.summary_tokens for e in eps)
        return {
            "working_items": len(self.working.items), "working_tokens": self.working.tokens,
            "episodes": len(eps), "facts_active": len(self.semantic.active()),
            "facts_superseded": len(self.semantic.facts) - len(self.semantic.active()),
            "compression_ratio": round(summ / raw, 4) if raw else 1.0,
            "counters": self.store.counters(),
        }

    def close(self) -> None:
        self.store.close()
