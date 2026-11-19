"""Semantic response cache with false-hit guards.

Lookup order: exact normalised-text hit → nearest neighbour by cosine
similarity (HashingEmbedder) above `threshold`, *then* the guard:

* the sets of numbers in both prompts must be identical ("5 km" ≠ "8 km");
* negation parity must match ("is X typed" ≠ "is X not typed");
* no antonym asymmetry (increase/decrease, min/max, before/after, ...);
* no template slot substitution (shared prefix+suffix, different filler words);
* content-word overlap (Jaccard over stopword-free words) >= `min_overlap`, so two
  prompts that only share a template ("What is the capital of X?") don't collide.

Entries are isolated per namespace (tenant), expire after `ttl` seconds and are
evicted LRU once a namespace holds `max_entries`.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from aikit import HashingEmbedder, tokenize_words

from .clock import Clock, SystemClock

_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_NEG = frozenset({"not", "no", "never", "without", "none", "cannot", "isn", "doesn", "don",
                  "aren", "wasn", "won", "shouldn", "can't", "nt"})
_ANTONYMS = [("increase", "decrease"), ("min", "max"), ("minimum", "maximum"),
             ("before", "after"), ("largest", "smallest"), ("highest", "lowest"),
             ("buy", "sell"), ("encrypt", "decrypt"), ("add", "remove"), ("enable", "disable"),
             ("first", "last"), ("north", "south"), ("east", "west"), ("celsius", "fahrenheit"),
             ("miles", "kilometers"), ("km", "miles")]
_STOP = frozenset("a an the of to in on for and or is are was be it this that with as by at "
                  "from what which who how do does can could should would will i you me my "
                  "please tell give explain".split())


def normalise(text: str) -> str:
    return " ".join(tokenize_words(text))


def _content(text: str) -> set[str]:
    return {w for w in tokenize_words(text) if w not in _STOP}


def slot_substitution(a: str, b: str, min_shared: float = 0.6) -> tuple[set[str], set[str]] | None:
    """Detect "same template, different slot filler" pairs.

    If the prompts share a common prefix + suffix covering >= `min_shared` of the longer
    one and the differing middles contain different content words, return those words.
    ("Write a Go function that ..." vs "Write a Rust function that ...").
    """
    ta, tb = tokenize_words(a), tokenize_words(b)
    if not ta or not tb or ta == tb:
        return None
    p = 0
    while p < min(len(ta), len(tb)) and ta[p] == tb[p]:
        p += 1
    s = 0
    while s < min(len(ta), len(tb)) - p and ta[-1 - s] == tb[-1 - s]:
        s += 1
    if p + s < min_shared * max(len(ta), len(tb)):
        return None
    mid_a = {w.rstrip("s") for w in ta[p:len(ta) - s] if w not in _STOP}
    mid_b = {w.rstrip("s") for w in tb[p:len(tb) - s] if w not in _STOP}
    if mid_a and mid_b and mid_a != mid_b:
        return mid_a, mid_b
    return None


def guard_reason(a: str, b: str, min_overlap: float = 0.5) -> str | None:
    """Why two semantically-similar prompts must NOT share a cached answer (None = safe)."""
    if {n.replace(",", "") for n in _NUM.findall(a)} != {n.replace(",", "") for n in _NUM.findall(b)}:
        return "number mismatch"
    wa, wb = tokenize_words(a.replace("n't", " not")), tokenize_words(b.replace("n't", " not"))
    if (sum(w in _NEG for w in wa) % 2) != (sum(w in _NEG for w in wb) % 2):
        return "negation mismatch"
    sa, sb = set(wa), set(wb)
    for x, y in _ANTONYMS:
        if ((x in sa) != (x in sb)) and ((y in sa) != (y in sb)):
            return f"antonym mismatch ({x}/{y})"
    slot = slot_substitution(a, b)
    if slot:
        return f"slot substitution ({sorted(slot[0])} vs {sorted(slot[1])})"
    ca, cb = _content(a), _content(b)
    if ca or cb:
        jac = len(ca & cb) / len(ca | cb)
        if jac < min_overlap:
            return f"low content overlap ({jac:.2f})"
    return None


@dataclass
class CacheEntry:
    prompt: str
    response: str
    model: str
    created: float
    vector: np.ndarray
    hits: int = 0
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CacheHit:
    entry: CacheEntry
    similarity: float
    exact: bool


class SemanticCache:
    def __init__(self, threshold: float = 0.75, ttl: float = 3600.0, max_entries: int = 1000,
                 min_overlap: float = 0.5, embedder: HashingEmbedder | None = None,
                 clock: Clock | None = None):
        self.threshold = threshold
        self.ttl = ttl
        self.max_entries = max_entries
        self.min_overlap = min_overlap
        self.embedder = embedder or HashingEmbedder()
        self.clock = clock or SystemClock()
        self._ns: dict[str, OrderedDict[str, CacheEntry]] = {}
        self.stats = {"hits": 0, "misses": 0, "guard_rejections": 0, "evictions": 0,
                      "expirations": 0}

    def _bucket(self, namespace: str) -> OrderedDict[str, CacheEntry]:
        return self._ns.setdefault(namespace, OrderedDict())

    def _purge_expired(self, bucket: OrderedDict[str, CacheEntry]) -> None:
        now = self.clock.now()
        dead = [k for k, e in bucket.items() if now - e.created > self.ttl]
        for k in dead:
            del bucket[k]
        self.stats["expirations"] += len(dead)

    def __len__(self) -> int:
        return sum(len(b) for b in self._ns.values())

    def _nearest(self, namespace: str, prompt: str) -> list[tuple[float, str, CacheEntry]]:
        bucket = self._bucket(namespace)
        self._purge_expired(bucket)
        if not bucket:
            return []
        keys = list(bucket)
        mat = np.stack([bucket[k].vector for k in keys])
        sims = mat @ self.embedder.embed_one(prompt)
        order = np.argsort(-sims)
        return [(float(sims[i]), keys[i], bucket[keys[i]]) for i in order]

    def get(self, namespace: str, prompt: str) -> CacheHit | None:
        bucket = self._bucket(namespace)
        self._purge_expired(bucket)
        key = normalise(prompt)
        if key in bucket:
            e = bucket[key]
            bucket.move_to_end(key)
            e.hits += 1
            self.stats["hits"] += 1
            return CacheHit(e, 1.0, True)
        for sim, k, e in self._nearest(namespace, prompt)[:5]:
            if sim < self.threshold:
                break
            if guard_reason(prompt, e.prompt, self.min_overlap) is not None:
                self.stats["guard_rejections"] += 1
                continue
            bucket.move_to_end(k)
            e.hits += 1
            self.stats["hits"] += 1
            return CacheHit(e, sim, False)
        self.stats["misses"] += 1
        return None

    def nearest(self, namespace: str, prompt: str, floor: float = 0.6) -> CacheHit | None:
        """Best guarded neighbour above a looser floor (degradation fallback only)."""
        for sim, _, e in self._nearest(namespace, prompt)[:5]:
            if sim < floor:
                break
            if guard_reason(prompt, e.prompt, self.min_overlap) is None:
                return CacheHit(e, sim, False)
        return None

    def put(self, namespace: str, prompt: str, response: str, model: str,
            meta: dict | None = None) -> None:
        bucket = self._bucket(namespace)
        key = normalise(prompt)
        bucket[key] = CacheEntry(prompt, response, model, self.clock.now(),
                                 self.embedder.embed_one(prompt), meta=meta or {})
        bucket.move_to_end(key)
        while len(bucket) > self.max_entries:
            bucket.popitem(last=False)
            self.stats["evictions"] += 1


class NullCache(SemanticCache):
    """Cache that never stores or returns anything (for ablations / opt-out tenants)."""

    def get(self, namespace: str, prompt: str) -> CacheHit | None:
        self.stats["misses"] += 1
        return None

    def nearest(self, namespace: str, prompt: str, floor: float = 0.6) -> CacheHit | None:
        return None

    def put(self, namespace: str, prompt: str, response: str, model: str,
            meta: dict | None = None) -> None:
        return None
