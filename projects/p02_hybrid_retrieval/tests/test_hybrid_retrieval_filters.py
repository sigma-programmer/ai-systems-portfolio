"""Metadata filter DSL: objects, operators, JSON parsing, masks."""
import numpy as np
import pytest

from hybrid_retrieval.filters import And, Eq, In, MetadataStore, Not, Or, Range, parse_filter

ROWS = [
    {"domain": "a", "year": 2019, "source": "docs", "tags": ["x"]},
    {"domain": "a", "year": 2022, "source": "forum"},
    {"domain": "b", "year": 2023, "source": "docs"},
    {"domain": "b", "year": None, "source": "blog"},
    {"domain": "c", "source": "docs"},
]


@pytest.fixture
def store():
    return MetadataStore(ROWS)


def rows_of(mask):
    return np.flatnonzero(mask).tolist()


def test_eq_in_range(store):
    assert rows_of(store.mask(Eq("domain", "a"))) == [0, 1]
    assert rows_of(store.mask(In("source", ["forum", "blog"]))) == [1, 3]
    assert rows_of(store.mask(Range("year", gte=2022))) == [1, 2]
    assert rows_of(store.mask(Range("year", gt=2019, lt=2023))) == [1]
    assert rows_of(store.mask(Eq("nope", 1))) == []


def test_boolean_operators(store):
    f = (Eq("domain", "a") | Eq("domain", "b")) & ~Eq("source", "docs")
    assert rows_of(store.mask(f)) == [1, 3]
    assert isinstance(Eq("a", 1) & Eq("b", 2) & Eq("c", 3), And)
    assert len((Eq("a", 1) & Eq("b", 2) & Eq("c", 3)).clauses) == 3  # flattened
    assert len((Eq("a", 1) | Eq("b", 2) | Eq("c", 3)).clauses) == 3


def test_mask_agrees_with_rowwise_matches(store):
    filters = [
        Eq("domain", "b"),
        In("domain", ["a", "c"]),
        Range("year", lte=2022),
        Not(Range("year", gte=2020)),
        Or([Eq("source", "blog"), And([Eq("domain", "a"), Range("year", lt=2020)])]),
    ]
    for f in filters:
        assert rows_of(store.mask(f)) == [i for i, r in enumerate(ROWS) if f.matches(r)]


def test_missing_and_incomparable_values_never_match_range(store):
    assert 3 not in rows_of(store.mask(Range("year", gte=0)))
    assert 4 not in rows_of(store.mask(Range("year", gte=0)))
    assert not Range("year", gte=2000).matches({"year": "abc"})


def test_range_requires_bound():
    with pytest.raises(ValueError):
        Range("year")


def test_parse_mongo_style(store):
    spec = {"$and": [{"domain": "a"}, {"year": {"$gte": 2020}}]}
    assert rows_of(store.mask(spec)) == [1]
    assert rows_of(store.mask({"source": {"$in": ["blog"]}})) == [3]
    assert rows_of(store.mask({"source": {"$nin": ["docs"]}})) == [1, 3]
    assert rows_of(store.mask({"domain": {"$ne": "a"}})) == [2, 3, 4]
    assert rows_of(store.mask({"$or": [{"domain": "c"}, {"year": {"$lt": 2020}}]})) == [0, 4]
    assert rows_of(store.mask({"$not": {"domain": "a"}})) == [2, 3, 4]
    assert rows_of(store.mask({"domain": "b", "source": "docs"})) == [2]  # implicit AND


def test_parse_errors():
    for bad in ({}, {"$xor": []}, {"year": {"$foo": 1}}, {"$and": []}, "domain"):
        with pytest.raises(ValueError):
            parse_filter(bad)
    assert parse_filter(None) is None
    f = Eq("a", 1)
    assert parse_filter(f) is f


def test_roundtrip_to_dict(store):
    f = Or([And([Eq("domain", "a"), Range("year", gte=2020)]), In("source", ["blog"])])
    assert rows_of(store.mask(parse_filter(f.to_dict()))) == rows_of(store.mask(f))


def test_cache_returns_same_mask_and_none(store):
    m1 = store.mask({"domain": "a"})
    m2 = store.mask(Eq("domain", "a"))
    assert m1 is m2
    assert store.mask(None) is None


def test_unhashable_values_are_skipped(store):
    assert rows_of(store.mask(Eq("tags", ["x"]))) == []
