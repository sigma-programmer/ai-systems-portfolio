"""Entity resolution: collapse surface mentions into canonical entities.

Merging rules, applied with a union-find over normalised name keys:

1. **Normalisation** — case, punctuation, a leading "the" and corporate
   suffixes ("Inc.", "Corp", "Ltd") are ignored: "Zentrix Labs Inc." ==
   "zentrix labs".
2. **Explicit aliases** — "Nimbus Freight (NF)", "X, also known as Y".
3. **Acronyms** — an all-caps token equal to the initials of exactly one
   multi-word name.
4. **Partial names** — a single token that is the surname of exactly one
   person, or the first word of exactly one organisation ("Okafor",
   "Zentrix"). Ambiguous stems are left alone.
5. **Fuzzy match** — difflib ratio >= ``fuzzy_threshold`` between
   type-compatible names (catches typos like "Quilon Systems").

Types are then decided by a mention-weighted vote, with relation signatures
(e.g. the subject of CEO_OF is a PERSON) breaking ties for untyped names.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .models import RELATION_SIGNATURES, Entity, EntityType, ExtractionResult
from .text import normalize_name, slugify


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # keep the longer key as root so roots stay human-meaningful
            if (len(ra.split()), len(ra)) < (len(rb.split()), len(rb)):
                ra, rb = rb, ra
            self.parent[rb] = ra


@dataclass
class _KeyStats:
    surfaces: Counter = field(default_factory=Counter)
    types: Counter = field(default_factory=Counter)
    chunks: set[str] = field(default_factory=set)
    count: int = 0

    def type(self) -> EntityType:
        typed = [(n, t) for t, n in self.types.items() if t != EntityType.OTHER]
        return max(typed, key=lambda x: (x[0], x[1].value))[1] if typed else EntityType.OTHER


def _compatible(a: EntityType, b: EntityType) -> bool:
    return a == b or EntityType.OTHER in (a, b)


@dataclass
class Resolution:
    """Output of :class:`EntityResolver`: canonical entities plus an alias index."""

    entities: dict[str, Entity]
    alias_index: dict[str, str]  # normalised surface -> entity id

    def lookup(self, surface: str) -> str | None:
        return self.alias_index.get(normalize_name(surface))


class EntityResolver:
    """Deduplicate mentions into entities (see module docstring for the rules)."""

    def __init__(self, fuzzy_threshold: float = 0.9, min_fuzzy_len: int = 6):
        self.fuzzy_threshold = fuzzy_threshold
        self.min_fuzzy_len = min_fuzzy_len

    def resolve(self, results: list[ExtractionResult]) -> Resolution:
        stats: dict[str, _KeyStats] = defaultdict(_KeyStats)
        for res in results:
            for m in res.mentions:
                key = normalize_name(m.text)
                if not key:
                    continue
                st = stats[key]
                st.surfaces[m.text] += 1
                st.types[m.type] += 1
                st.chunks.add(m.chunk_id)
                st.count += 1
            for rel in res.relations:
                for name in (rel.subject, rel.object):
                    key = normalize_name(name)
                    if key and key not in stats:
                        stats[key].surfaces[name] += 1
                        stats[key].types[EntityType.OTHER] += 1
                        stats[key].chunks.add(rel.chunk_id)

        self._infer_types_from_relations(stats, results)

        uf = _UnionFind()
        for key in stats:
            uf.add(key)
        # 2. explicit aliases
        for res in results:
            for alias, full in res.aliases:
                a, f = normalize_name(alias), normalize_name(full)
                if a in stats and f in stats:
                    uf.union(f, a)

        multi = [k for k in stats if len(k.split()) > 1]
        for key in [k for k in stats if len(k.split()) == 1]:
            ktype = stats[key].type()
            # 3. acronyms
            surf = stats[key].surfaces.most_common(1)[0][0]
            if surf.isupper() and 2 <= len(surf) <= 5:
                cands = {uf.find(k) for k in multi if "".join(w[0] for w in k.split()) == key and _compatible(ktype, stats[k].type())}
                if len(cands) == 1:
                    uf.union(cands.pop(), key)
                    continue
            # 4. partial names: surname of one person / first word of one org
            cands = set()
            for k in multi:
                ktype2 = stats[k].type()
                toks = k.split()
                if not _compatible(ktype, ktype2):
                    continue
                if (ktype2 == EntityType.PERSON and toks[-1] == key) or (ktype2 == EntityType.ORG and toks[0] == key):
                    cands.add(uf.find(k))
            if len(cands) == 1:
                uf.union(cands.pop(), key)

        # 5. fuzzy
        keys = sorted(stats)
        for i, a in enumerate(keys):
            if len(a) < self.min_fuzzy_len:
                continue
            for b in keys[i + 1 :]:
                if len(b) < self.min_fuzzy_len or abs(len(a) - len(b)) > 3 or uf.find(a) == uf.find(b):
                    continue
                if not _compatible(stats[a].type(), stats[b].type()):
                    continue
                if SequenceMatcher(None, a, b).ratio() >= self.fuzzy_threshold:
                    uf.union(a, b)

        return self._build(stats, uf)

    @staticmethod
    def _infer_types_from_relations(stats: dict[str, _KeyStats], results: list[ExtractionResult]) -> None:
        """Vote a type for untyped names from the relation slots they fill."""
        votes: dict[str, Counter] = defaultdict(Counter)
        for res in results:
            for rel in res.relations:
                subj_t, obj_t = RELATION_SIGNATURES[rel.relation]
                for name, allowed in ((rel.subject, subj_t), (rel.object, obj_t)):
                    if allowed is not None and len(allowed) == 1:
                        votes[normalize_name(name)][next(iter(allowed))] += 1
        for key, counter in votes.items():
            st = stats.get(key)
            if st is not None and st.type() == EntityType.OTHER and counter:
                st.types[counter.most_common(1)[0][0]] += 1

    @staticmethod
    def _build(stats: dict[str, _KeyStats], uf: _UnionFind) -> Resolution:
        groups: dict[str, list[str]] = defaultdict(list)
        for key in stats:
            groups[uf.find(key)].append(key)
        entities: dict[str, Entity] = {}
        alias_index: dict[str, str] = {}
        for root in sorted(groups):
            keys = groups[root]
            surfaces: Counter = Counter()
            types: Counter = Counter()
            chunks: set[str] = set()
            count = 0
            for k in keys:
                surfaces.update(stats[k].surfaces)
                types.update(stats[k].types)
                chunks |= stats[k].chunks
                count += stats[k].count
            name = max(surfaces.items(), key=lambda kv: (len(kv[0].split()) > 1, kv[1], len(kv[0]), kv[0]))[0]
            typed = [(n, t) for t, n in types.items() if t != EntityType.OTHER]
            etype = max(typed, key=lambda x: (x[0], x[1].value))[1] if typed else EntityType.OTHER
            eid = base = slugify(name)
            n = 2
            while eid in entities:
                eid, n = f"{base}_{n}", n + 1
            entities[eid] = Entity(
                id=eid,
                name=name,
                type=etype,
                aliases=sorted(s for s in surfaces if s != name),
                mention_count=count,
                source_chunks=sorted(chunks),
            )
            for k in keys:
                alias_index[k] = eid
        return Resolution(entities=entities, alias_index=alias_index)
