"""A small metadata filter DSL, compiled to boolean masks *before* retrieval.

Build filters with Python objects and operators::

    (Eq("domain", "automotive") | In("source", ["docs", "support"])) & Range("year", gte=2021)

or parse a Mongo-style JSON spec::

    {"$and": [{"domain": "automotive"}, {"year": {"$gte": 2021, "$lt": 2025}}]}

Equality / membership use a per-field value -> row-bitmap index; ranges scan
the column once. Masks are cached per canonical filter so repeated queries
with the same filter cost nothing.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

_MISSING = object()


class Filter:
    """Base class. Subclasses implement `matches` (row-wise) and `mask` (vectorized)."""

    def matches(self, meta: dict[str, Any]) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def mask(self, store: "MetadataStore") -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def __and__(self, other: "Filter") -> "And":
        return And([self, other])

    def __or__(self, other: "Filter") -> "Or":
        return Or([self, other])

    def __invert__(self) -> "Not":
        return Not(self)

    def key(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)


@dataclass(frozen=True)
class Eq(Filter):
    field: str
    value: Any

    def matches(self, meta):
        return meta.get(self.field, _MISSING) == self.value

    def mask(self, store):
        return store.eq_mask(self.field, self.value)

    def to_dict(self):
        return {self.field: {"$eq": self.value}}


@dataclass(frozen=True)
class In(Filter):
    field: str
    values: tuple

    def __init__(self, field: str, values: Iterable[Any]):
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "values", tuple(values))

    def matches(self, meta):
        return meta.get(self.field, _MISSING) in self.values

    def mask(self, store):
        out = np.zeros(store.n, dtype=bool)
        for v in self.values:
            out |= store.eq_mask(self.field, v)
        return out

    def to_dict(self):
        return {self.field: {"$in": list(self.values)}}


@dataclass(frozen=True)
class Range(Filter):
    """Inclusive (gte/lte) and exclusive (gt/lt) bounds; any subset may be given.

    Rows whose value is missing or not comparable to the bounds never match.
    """

    field: str
    gte: Any = None
    lte: Any = None
    gt: Any = None
    lt: Any = None

    def __post_init__(self):
        if all(v is None for v in (self.gte, self.lte, self.gt, self.lt)):
            raise ValueError(f"Range on {self.field!r} needs at least one bound")

    def matches(self, meta):
        v = meta.get(self.field, _MISSING)
        if v is _MISSING or v is None:
            return False
        try:
            if self.gte is not None and not v >= self.gte:
                return False
            if self.gt is not None and not v > self.gt:
                return False
            if self.lte is not None and not v <= self.lte:
                return False
            if self.lt is not None and not v < self.lt:
                return False
        except TypeError:
            return False
        return True

    def mask(self, store):
        return np.fromiter((self.matches(m) for m in store.rows), dtype=bool, count=store.n)

    def to_dict(self):
        ops = {"$gte": self.gte, "$lte": self.lte, "$gt": self.gt, "$lt": self.lt}
        return {self.field: {k: v for k, v in ops.items() if v is not None}}


@dataclass(frozen=True)
class And(Filter):
    clauses: tuple

    def __init__(self, clauses: Sequence[Filter]):
        flat: list[Filter] = []
        for c in clauses:
            flat.extend(c.clauses if isinstance(c, And) else [c])
        object.__setattr__(self, "clauses", tuple(flat))

    def matches(self, meta):
        return all(c.matches(meta) for c in self.clauses)

    def mask(self, store):
        out = np.ones(store.n, dtype=bool)
        for c in self.clauses:
            out &= store.mask(c)
        return out

    def to_dict(self):
        return {"$and": [c.to_dict() for c in self.clauses]}


@dataclass(frozen=True)
class Or(Filter):
    clauses: tuple

    def __init__(self, clauses: Sequence[Filter]):
        flat: list[Filter] = []
        for c in clauses:
            flat.extend(c.clauses if isinstance(c, Or) else [c])
        object.__setattr__(self, "clauses", tuple(flat))

    def matches(self, meta):
        return any(c.matches(meta) for c in self.clauses)

    def mask(self, store):
        out = np.zeros(store.n, dtype=bool)
        for c in self.clauses:
            out |= store.mask(c)
        return out

    def to_dict(self):
        return {"$or": [c.to_dict() for c in self.clauses]}


@dataclass(frozen=True)
class Not(Filter):
    clause: Filter

    def matches(self, meta):
        return not self.clause.matches(meta)

    def mask(self, store):
        return ~store.mask(self.clause)

    def to_dict(self):
        return {"$not": self.clause.to_dict()}


_FIELD_OPS = {"$eq", "$ne", "$in", "$nin", "$gte", "$lte", "$gt", "$lt"}


def parse_filter(spec: dict[str, Any] | Filter | None) -> Filter | None:
    """Parse a Mongo-style dict into a Filter tree (Filters pass through)."""
    if spec is None or isinstance(spec, Filter):
        return spec
    if not isinstance(spec, dict) or not spec:
        raise ValueError(f"filter spec must be a non-empty dict, got {spec!r}")
    clauses: list[Filter] = []
    for key, val in spec.items():
        if key in ("$and", "$or"):
            if not isinstance(val, list) or not val:
                raise ValueError(f"{key} expects a non-empty list")
            subs = [parse_filter(v) for v in val]
            clauses.append(And(subs) if key == "$and" else Or(subs))
        elif key == "$not":
            clauses.append(Not(parse_filter(val)))
        elif key.startswith("$"):
            raise ValueError(f"unknown top-level operator {key!r}")
        elif isinstance(val, dict) and val and all(k.startswith("$") for k in val):
            unknown = set(val) - _FIELD_OPS
            if unknown:
                raise ValueError(f"unknown operator(s) {sorted(unknown)} on field {key!r}")
            bounds = {op[1:]: val[op] for op in ("$gte", "$lte", "$gt", "$lt") if op in val}
            if bounds:
                clauses.append(Range(key, **bounds))
            if "$eq" in val:
                clauses.append(Eq(key, val["$eq"]))
            if "$ne" in val:
                clauses.append(Not(Eq(key, val["$ne"])))
            if "$in" in val:
                clauses.append(In(key, val["$in"]))
            if "$nin" in val:
                clauses.append(Not(In(key, val["$nin"])))
        else:
            clauses.append(Eq(key, val))
    return clauses[0] if len(clauses) == 1 else And(clauses)


class MetadataStore:
    """Columnar view of document metadata with an equality bitmap index."""

    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self.rows = list(rows)
        self.n = len(self.rows)
        self._eq: dict[str, dict[Any, np.ndarray]] = defaultdict(dict)
        index: dict[str, dict[Any, list[int]]] = defaultdict(lambda: defaultdict(list))
        for i, meta in enumerate(self.rows):
            for f, v in meta.items():
                try:
                    hash(v)
                except TypeError:
                    continue
                index[f][v].append(i)
        for f, vals in index.items():
            for v, idx in vals.items():
                m = np.zeros(self.n, dtype=bool)
                m[idx] = True
                self._eq[f][v] = m
        self._cache: dict[str, np.ndarray] = {}

    def eq_mask(self, field: str, value: Any) -> np.ndarray:
        m = self._eq.get(field, {}).get(value) if _hashable(value) else None
        return m.copy() if m is not None else np.zeros(self.n, dtype=bool)

    def mask(self, flt: Filter | dict | None) -> np.ndarray | None:
        """Boolean row mask for a filter (None means 'no filter')."""
        flt = parse_filter(flt)
        if flt is None:
            return None
        key = flt.key()
        if key not in self._cache:
            if len(self._cache) > 1024:
                self._cache.clear()
            self._cache[key] = flt.mask(self)
        return self._cache[key]


def _hashable(v: Any) -> bool:
    try:
        hash(v)
    except TypeError:
        return False
    return True
