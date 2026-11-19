"""BM25, dense indexes, fusion, and logistic regression."""
import math

import numpy as np
import pytest

from hybrid_retrieval.bm25 import BM25Index, top_k_indices
from hybrid_retrieval.dense import FlatIndex, IVFIndex, spherical_kmeans
from hybrid_retrieval.fusion import normalize_scores, reciprocal_rank_fusion, weighted_score_fusion
from hybrid_retrieval.logreg import LogisticRegression
from hybrid_retrieval.text import analyze, char_ngrams, jaccard

DOCS = [
    "the quick brown fox jumps over the lazy dog",
    "a fast brown fox",
    "lazy dogs sleep all day",
    "error code 0xdeadbe appears on boot",
    "",
]


def test_analyze_drops_stopwords_and_lowercases():
    assert analyze("The Quick, brown FOX!") == ["quick", "brown", "fox"]


def test_char_ngrams_and_jaccard():
    assert "<ve" in char_ngrams("veltra")
    assert jaccard(char_ngrams("veltra"), char_ngrams("veltra")) == 1.0
    assert jaccard(char_ngrams("veltra"), char_ngrams("veltrra")) > 0.5
    assert jaccard(frozenset(), char_ngrams("x")) == 0.0


def test_bm25_matches_reference_formula():
    idx = BM25Index(k1=1.2, b=0.75).fit(DOCS)
    toks = [analyze(d) for d in DOCS]
    avgdl = sum(map(len, toks)) / len(toks)
    n = len(DOCS)

    def ref(query, i):
        s = 0.0
        for t in analyze(query):
            df = sum(t in d for d in toks)
            tf = toks[i].count(t)
            if tf == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            s += idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * len(toks[i]) / avgdl))
        return s

    scores = idx.score_all("brown fox lazy")
    for i in range(n):
        assert scores[i] == pytest.approx(ref("brown fox lazy", i), rel=1e-5)


def test_bm25_rare_term_dominates_and_search_filters_zero():
    idx = BM25Index().fit(DOCS)
    hits = idx.search("0xdeadbe fox", k=10)
    assert hits[0][0] == 3
    assert all(s > 0 for _, s in hits)
    assert {i for i, _ in hits} == {0, 1, 3}
    assert idx.idf("0xdeadbe") > idx.idf("fox") > 0


def test_bm25_mask_and_unknown_terms_and_empty():
    idx = BM25Index().fit(DOCS)
    mask = np.array([False, True, True, True, True])
    assert [i for i, _ in idx.search("brown fox", 5, mask)] == [1]
    assert idx.search("zzzz", 5) == []
    assert idx.search("", 5) == []
    empty = BM25Index().fit([])
    assert empty.search("anything") == []


def test_bm25_length_normalization():
    idx = BM25Index().fit(["fox", "fox " + "filler " * 30])
    s = idx.score_all("fox")
    assert s[0] > s[1]


def test_top_k_indices_deterministic_ties():
    scores = np.array([1.0, 3.0, 3.0, 2.0])
    assert top_k_indices(scores, 3).tolist() == [1, 2, 3]
    assert top_k_indices(scores, 10, np.array([True, False, True, False])).tolist() == [2, 0]
    assert top_k_indices(scores, 0).size == 0


def _unit(rng, n, d):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_flat_index_exact_and_mask():
    rng = np.random.default_rng(0)
    x = _unit(rng, 100, 16)
    flat = FlatIndex(x)
    hits = flat.search(x[7], 3)
    assert hits[0][0] == 7 and hits[0][1] == pytest.approx(1.0, abs=1e-5)
    mask = np.ones(100, bool)
    mask[7] = False
    assert 7 not in [i for i, _ in flat.search(x[7], 5, mask)]
    assert FlatIndex(np.zeros((0, 16), np.float32)).search(x[0]) == []


def test_spherical_kmeans_recovers_separated_clusters():
    rng = np.random.default_rng(1)
    centers = _unit(rng, 4, 32)
    x = np.repeat(centers, 25, axis=0) + 0.05 * rng.normal(size=(100, 32))
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    _, labels = spherical_kmeans(x, 4, seed=0)
    for g in range(4):
        assert len(set(labels[g * 25 : (g + 1) * 25])) == 1
    assert len(set(labels)) == 4


def test_ivf_recall_increases_with_nprobe_and_full_probe_is_exact():
    rng = np.random.default_rng(2)
    x = _unit(rng, 2000, 32)
    q = _unit(rng, 50, 32)
    flat, ivf = FlatIndex(x), IVFIndex(x, n_lists=32, seed=0)
    truth = [{i for i, _ in flat.search(v, 10)} for v in q]

    def recall(nprobe):
        return np.mean([len({i for i, _ in ivf.search(v, 10, nprobe=nprobe)} & t) / 10 for v, t in zip(q, truth)])

    r1, r8, rall = recall(1), recall(8), recall(32)
    assert r1 <= r8 <= rall
    assert rall == pytest.approx(1.0)
    assert sum(len(l) for l in ivf.lists) == 2000


def test_ivf_mask_and_tiny_inputs():
    rng = np.random.default_rng(3)
    x = _unit(rng, 5, 8)
    ivf = IVFIndex(x, n_lists=16)  # more lists than points is clamped
    assert ivf.n_lists == 5
    mask = np.zeros(5, bool)
    mask[2] = True
    assert [i for i, _ in ivf.search(x[0], 5, mask, nprobe=5)] == [2]
    assert ivf.search(x[0], 5, np.zeros(5, bool), nprobe=5) == []


def test_rrf_scores_and_order():
    fused = reciprocal_rank_fusion({"a": [(1, 9.0), (2, 5.0)], "b": [(2, 0.9), (3, 0.8)]}, k=60)
    ids = [i for i, _, _ in fused]
    assert ids[0] == 2
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert set(ids) == {1, 2, 3}
    assert fused[0][2] == {"a": pytest.approx(1 / 62), "b": pytest.approx(1 / 61)}


def test_rrf_weights_and_empty():
    fused = reciprocal_rank_fusion({"a": [(1, 1.0)], "b": [(2, 1.0)]}, weights={"a": 2.0, "b": 1.0})
    assert fused[0][0] == 1
    assert reciprocal_rank_fusion({}) == []


def test_normalize_scores_methods():
    r = [(0, 10.0), (1, 5.0), (2, 0.0)]
    assert normalize_scores(r) == {0: 1.0, 1: 0.5, 2: 0.0}
    assert normalize_scores(r, "max")[1] == 0.5
    z = normalize_scores(r, "zscore")
    assert z[1] == pytest.approx(0.0)
    assert normalize_scores([(4, 3.0), (5, 3.0)]) == {4: 1.0, 5: 1.0}
    assert normalize_scores([]) == {}


def test_weighted_fusion_alpha_extremes():
    lists = {"bm25": [(1, 10.0), (2, 1.0)], "dense": [(2, 0.9), (1, 0.1)]}
    assert weighted_score_fusion(lists, {"bm25": 1.0, "dense": 0.0})[0][0] == 1
    assert weighted_score_fusion(lists, {"bm25": 0.0, "dense": 1.0})[0][0] == 2
    fused = weighted_score_fusion({"bm25": [(1, 1.0)], "dense": [(2, 1.0), (3, 0.5)]}, {"bm25": 0.5, "dense": 0.5})
    assert all(set(parts) == {"bm25", "dense"} for _, _, parts in fused)


def test_logreg_learns_separable_and_matches_direction():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (2 * X[:, 0] - X[:, 1] > 0).astype(int)
    m = LogisticRegression(l2=0.1).fit(X, y)
    assert (m.predict(X) == y).mean() > 0.97
    assert m.coef_[0] > 0 > m.coef_[1]
    assert abs(m.coef_[2]) < abs(m.coef_[1])
    w, b = m.raw_weights()
    assert np.allclose(X @ w + b, m.decision_function(X))
    assert m.n_iter_ < 50


def test_logreg_regularization_shrinks_and_errors():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    small = LogisticRegression(l2=0.01).fit(X, y)
    big = LogisticRegression(l2=100.0).fit(X, y)
    assert np.linalg.norm(big.coef_) < np.linalg.norm(small.coef_)
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, np.ones(200))
    with pytest.raises(RuntimeError):
        LogisticRegression().decision_function(X)
    const = np.hstack([X, np.ones((200, 1))])  # zero-variance column must not blow up
    assert np.isfinite(LogisticRegression().fit(const, y).coef_).all()
