"""End-to-end pipeline, rerankers, evaluation metrics, corpus and tuning."""
import json

import numpy as np
import pytest

from aikit import MockLLM
from hybrid_retrieval import (
    Document,
    Eq,
    FEATURE_NAMES,
    HybridSearcher,
    LLMReranker,
    Query,
    SearchConfig,
    build_corpus,
    build_queries,
    default_sweep_configs,
    evaluate,
    load_benchmark,
    ndcg_at_k,
    pareto_frontier,
    recall_at_k,
    reciprocal_rank,
    split_queries,
    sweep,
    train_reranker,
)
from hybrid_retrieval.corpus import misspell
from hybrid_retrieval.features import min_window
from hybrid_retrieval.tuning import build_training_set


@pytest.fixture(scope="module")
def bench():
    docs, queries = load_benchmark()
    searcher = HybridSearcher(docs)
    train, test = split_queries(queries)
    train_reranker(searcher, train)
    return searcher, train, test


# ---------------------------------------------------------------- corpus

def test_corpus_shape_and_determinism():
    docs = build_corpus()
    assert len(docs) == 240
    assert len({d.id for d in docs}) == 240
    assert [d.text for d in docs] == [d.text for d in build_corpus()]
    assert {d.metadata["domain"] for d in docs} >= {"automotive", "telehealth"}
    codes = [d.metadata["code"] for d in docs if d.metadata["code"]]
    assert len(codes) == len(set(codes)) == 30


def test_queries_have_valid_graded_qrels():
    docs = build_corpus()
    ids = {d.id for d in docs}
    qs = build_queries(docs)
    assert len(qs) == 240
    assert {q.qtype for q in qs} == {"keyword", "lexical", "mixed", "paraphrase"}
    for q in qs:
        assert set(q.qrels) <= ids
        assert sum(g == 2 for g in q.qrels.values()) == 1


def test_split_is_stratified_and_disjoint():
    _, qs = load_benchmark()
    train, test = split_queries(qs)
    assert not {q.id for q in train} & {q.id for q in test}
    assert len(train) + len(test) == len(qs)
    for t in ("keyword", "paraphrase"):
        assert sum(q.qtype == t for q in train) == sum(q.qtype == t for q in test)


def test_misspell_changes_word_deterministically():
    import random

    out = misspell("Veltra", random.Random(1))
    assert out.lower() != "veltra" and abs(len(out) - 6) <= 1
    assert misspell("Veltra", random.Random(1)) == out
    assert misspell("ab", random.Random(0)) == "abb"


# ---------------------------------------------------------------- metrics

def test_metrics_hand_computed():
    qrels = {"a": 2, "b": 1, "c": 2}
    ranked = ["x", "a", "b", "c"]
    assert recall_at_k(ranked, qrels, 2) == 0.5
    assert recall_at_k(ranked, qrels, 4) == 1.0
    assert recall_at_k(ranked, qrels, 4, threshold=1) == 1.0
    assert reciprocal_rank(ranked, qrels) == 0.5
    assert reciprocal_rank(ranked, qrels, k=1) == 0.0
    assert reciprocal_rank(["b"], qrels, threshold=1) == 1.0
    ideal = 3 + 3 / np.log2(3) + 1 / 2
    got = 3 / np.log2(3) + 1 / 2 + 3 / np.log2(5)
    assert ndcg_at_k(ranked, qrels, 4) == pytest.approx(got / ideal)
    assert ndcg_at_k(["a", "c", "b"], qrels, 3) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, {}, 4) == 0.0
    assert recall_at_k(ranked, {"a": 1}, 4) == 0.0  # nothing reaches threshold 2


def test_min_window():
    assert min_window([[0, 10], [3], [4, 20]]) == 5
    assert min_window([[5], [5]]) == 1
    assert min_window([[1, 2, 3]]) == 1


# ---------------------------------------------------------------- pipeline

def test_search_config_validation_and_labels():
    assert SearchConfig(mode="sparse").label == "sparse/c50"
    assert SearchConfig(dense_index="ivf", nprobe=2, rerank=True).label == "hybrid/ivf@2/rrf/c50/rr20"
    assert SearchConfig(name="custom").label == "custom"
    with pytest.raises(ValueError):
        SearchConfig(alpha=1.5)
    with pytest.raises(ValueError):
        SearchConfig(mode="bogus")


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        HybridSearcher([Document("a", "x"), Document("a", "y")])


def test_small_corpus_modes_and_components():
    docs = [
        Document("1", "refund policy for orders within thirty days", "Refunds", {"t": "billing"}),
        Document("2", "reset your password from the settings page", "Login", {"t": "auth"}),
        Document("3", "error code 0xbeef123 appears when the cache is corrupt", "Errors", {"t": "ops"}),
    ]
    s = HybridSearcher(docs, ivf_lists=2)
    assert s.search("0xbeef123", 1, SearchConfig(mode="sparse")).doc_ids == ["3"]
    assert s.search("password reset", 1, SearchConfig(mode="dense")).doc_ids == ["2"]
    r = s.search("reimbursement for my order", 3, SearchConfig(mode="hybrid", fusion="weighted"))
    assert r.doc_ids[0] == "1"
    assert {"bm25", "dense", "fused"} <= set(r.hits[0].components)
    assert {"filter", "sparse", "dense", "fusion", "total"} <= set(r.timings_ms)
    rr = s.search("reimbursement for my order", 3, SearchConfig(rerank=True))  # untrained default weights
    assert rr.doc_ids[0] == "1" and "rerank" in rr.hits[0].components
    assert s.search("password", 3, filter={"t": "ops"}).doc_ids in ([], ["3"])
    assert s.search("zzzz qqqq", 3, SearchConfig(mode="sparse")).hits == []


def test_filter_is_respected_in_every_mode(bench):
    searcher, _, test = bench
    flt = Eq("domain", "telehealth") & Eq("source", "docs")
    allowed = {d.id for d in searcher.docs if d.metadata["domain"] == "telehealth" and d.metadata["source"] == "docs"}
    for cfg in (SearchConfig(mode="sparse"), SearchConfig(mode="dense"),
                SearchConfig(dense_index="ivf", nprobe=1), SearchConfig(rerank=True)):
        for q in test[:15]:
            assert set(searcher.search(q.text, 10, cfg, flt).doc_ids) <= allowed


def test_rerank_never_drops_or_duplicates(bench):
    searcher, _, test = bench
    for q in test[:20]:
        base = searcher.search(q.text, 30, SearchConfig(candidate_k=15))
        rr = searcher.search(q.text, 30, SearchConfig(candidate_k=15, rerank=True, rerank_depth=10))
        assert sorted(base.doc_ids) == sorted(rr.doc_ids)
        assert len(set(rr.doc_ids)) == len(rr.doc_ids)


def test_sparse_wins_keyword_dense_wins_paraphrase_hybrid_rerank_wins_overall(bench):
    searcher, _, test = bench
    res = {m: evaluate(searcher, test, SearchConfig(mode=m, fusion="weighted")) for m in ("sparse", "dense", "hybrid")}
    best = evaluate(searcher, test, SearchConfig(fusion="weighted", rerank=True, rerank_depth=40))
    bt = {m: r["by_type"] for m, r in res.items()}
    assert bt["sparse"]["keyword"]["mrr"] > bt["dense"]["keyword"]["mrr"] + 0.5
    assert bt["dense"]["paraphrase"]["mrr"] > bt["sparse"]["paraphrase"]["mrr"] + 0.15
    assert bt["dense"]["mixed"]["mrr"] > bt["sparse"]["mixed"]["mrr"]
    assert res["hybrid"]["mrr"] > max(res["sparse"]["mrr"], res["dense"]["mrr"])
    assert best["mrr"] > res["hybrid"]["mrr"] + 0.1
    assert best["mrr"] > 0.9


def test_evaluation_is_deterministic(bench):
    searcher, _, test = bench
    a = evaluate(searcher, test, SearchConfig(rerank=True))
    b = evaluate(searcher, test, SearchConfig(rerank=True))
    for key in ("mrr", "ndcg@10", "recall@10", "by_type"):
        assert a[key] == b[key]


def test_training_set_and_learned_weights(bench):
    searcher, train, _ = bench
    X, y = build_training_set(searcher, train[:20], depth=10)
    assert X.shape[1] == len(FEATURE_NAMES) and X.shape[0] == len(y)
    assert 0 < y.sum() < len(y)
    weights = searcher.reranker.feature_weights()
    assert set(weights) == set(FEATURE_NAMES)
    assert searcher.reranker.trained


def test_learned_reranker_beats_untrained_default(bench):
    from hybrid_retrieval.rerank import CrossEncoderReranker

    searcher, _, test = bench
    cfg = SearchConfig(rerank=True, rerank_depth=40)
    learned = evaluate(searcher, test, cfg)["mrr"]
    trained = searcher.reranker
    searcher.set_reranker(CrossEncoderReranker(searcher.extractor))
    try:
        default = evaluate(searcher, test, cfg)["mrr"]
    finally:
        searcher.set_reranker(trained)
    assert learned >= default


# ---------------------------------------------------------------- tuning

def test_pareto_frontier_logic():
    pts = [
        {"config": "a", "lat": 1.0, "acc": 0.5},
        {"config": "b", "lat": 2.0, "acc": 0.4},  # dominated by a
        {"config": "c", "lat": 3.0, "acc": 0.9},
        {"config": "d", "lat": 3.0, "acc": 0.8},  # dominated by c
        {"config": "e", "lat": 0.5, "acc": 0.1},
    ]
    assert [p["config"] for p in pareto_frontier(pts, "lat", "acc")] == ["e", "a", "c"]
    assert pareto_frontier([], "lat", "acc") == []


def test_sweep_runs_and_frontier_is_monotone(bench):
    searcher, _, test = bench
    configs = default_sweep_configs()
    assert len({c.label for c in configs}) == len(configs)
    results = sweep(searcher, test[:40], configs[:6])
    assert len(results) == 6
    front = pareto_frontier(results)
    gains = [p["ndcg@10"] for p in front]
    lats = [p["latency_ms_p50"] for p in front]
    assert gains == sorted(gains) and lats == sorted(lats)
    json.dumps(results)


# ---------------------------------------------------------------- LLM reranker

def _llm_hits(searcher, text):
    return searcher.candidates(text, SearchConfig())[:5]


def test_llm_reranker_reorders_by_json_scores(bench):
    searcher, _, _ = bench
    hits = _llm_hits(searcher, "Veltra X2 refund")
    llm = MockLLM().on(r"Passages:", json.dumps({"scores": [{"id": 5, "score": 9}, {"id": 1, "score": 2}]}))
    rr = LLMReranker(llm, [d.full_text for d in searcher.docs])
    out = rr.rerank("Veltra X2 refund", None, hits)
    assert out[0].doc_id == hits[4].doc_id
    assert out[1].doc_id == hits[0].doc_id
    assert [h.doc_id for h in out[2:]] == [hits[1].doc_id, hits[2].doc_id, hits[3].doc_id]
    assert sorted(h.doc_id for h in out) == sorted(h.doc_id for h in hits)
    assert llm.calls[0]["json_mode"] is True
    assert "[5]" in llm.calls[0]["prompt"] and "Veltra X2 refund" in llm.calls[0]["prompt"]
    assert rr.last_prompt_tokens > 0


def test_llm_reranker_accepts_bare_list_and_ignores_bad_ids(bench):
    searcher, _, _ = bench
    hits = _llm_hits(searcher, "pricing")
    rr = LLMReranker(MockLLM(default="[1, 2, 3, 4, 10]"), [d.full_text for d in searcher.docs])
    assert rr.rerank("pricing", None, hits)[0].doc_id == hits[4].doc_id
    rr2 = LLMReranker(MockLLM(default='{"scores": [{"id": 99, "score": 10}, {"id": 2, "score": 5}]}'),
                      [d.full_text for d in searcher.docs])
    assert rr2.rerank("pricing", None, hits)[0].doc_id == hits[1].doc_id


def test_llm_reranker_falls_back_on_garbage(bench):
    searcher, _, _ = bench
    hits = _llm_hits(searcher, "pricing")
    rr = LLMReranker(MockLLM(default="I cannot do that."), [d.full_text for d in searcher.docs])
    assert [h.doc_id for h in rr.rerank("pricing", None, hits)] == [h.doc_id for h in hits]
    assert rr.last_error
    assert rr.rerank("pricing", None, []) == []


def test_llm_reranker_plugs_into_searcher(bench):
    searcher, _, _ = bench
    target = "auto-veltra-refund"
    texts = [d.full_text for d in searcher.docs]

    def judge(prompt, system):
        # score passages that mention "refund" and "Veltra" highest
        scores = []
        for line in prompt.splitlines():
            if line.startswith("["):
                n = int(line[1 : line.index("]")])
                scores.append({"id": n, "score": 10 if ("Veltra" in line and "refund" in line) else 1})
        return json.dumps({"scores": scores})

    trained = searcher.reranker
    searcher.set_reranker(LLMReranker(MockLLM(default=judge), texts))
    try:
        r = searcher.search("Veltra money back", 5, SearchConfig(rerank=True, rerank_depth=20))
    finally:
        searcher.set_reranker(trained)
    assert r.doc_ids[0] == target
    assert "llm" in r.hits[0].components
