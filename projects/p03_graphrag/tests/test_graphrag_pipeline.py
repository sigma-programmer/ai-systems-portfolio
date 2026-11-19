"""Knowledge graph, communities, local/global search, routing, export, baseline."""
from __future__ import annotations

import json

import pytest

from aikit import MockLLM
from graphrag import (
    DOCUMENTS, GLOBAL_QUESTIONS, GOLD_TRIPLES, QUESTIONS, Document, GlobalAnswer, GraphRAG, KnowledgeGraph,
    LocalAnswer, VectorRAG,
)
from graphrag.models import RelationType


@pytest.fixture(scope="module")
def rag() -> GraphRAG:
    return GraphRAG().index(DOCUMENTS)


# ----------------------------------------------------------------- graph
def test_extraction_matches_gold_triples(rag):
    triples = rag.kg.triples()
    precision = len(triples & GOLD_TRIPLES) / len(triples)
    recall = len(triples & GOLD_TRIPLES) / len(GOLD_TRIPLES)
    assert precision >= 0.9 and recall >= 0.9


def test_graph_has_provenance(rag):
    kg = rag.kg
    node = kg.find("Zentrix Labs")
    assert node and kg.g.nodes[node]["source_chunks"]
    for _, _, d in kg.g.edges(data=True):
        assert d["source_chunks"], "every edge must cite at least one chunk"
        assert all(c in kg.chunk_text for c in d["source_chunks"])


def test_edge_weight_accumulates_across_documents(rag):
    kg = rag.kg
    z, alice = kg.find("Zentrix Labs"), kg.find("Alice Moreau")
    edges = list(kg.edges_to(z, RelationType.CEO_OF))
    data = next(d for u, _, d in edges if u == alice)
    assert data["weight"] >= 2 and len(data["source_chunks"]) >= 2


def test_type_signature_violation_downgrades_to_related(rag):
    kg = KnowledgeGraph()
    from graphrag.models import Entity, EntityType
    kg.add_entity(Entity(id="p", name="Pat", type=EntityType.PERSON))
    kg.add_entity(Entity(id="q", name="Quinn", type=EntityType.PERSON))
    from graphrag.resolution import Resolution
    from graphrag.models import ExtractedRelation, ExtractionResult
    res = Resolution(entities={"p": Entity(id="p", name="Pat", type=EntityType.PERSON),
                               "q": Entity(id="q", name="Quinn", type=EntityType.PERSON)},
                     alias_index={"pat": "p", "quinn": "q"})
    ext = [ExtractionResult(chunk_id="c", relations=[
        ExtractedRelation(subject="Pat", relation=RelationType.ACQUIRED, object="Quinn", chunk_id="c")])]
    built = KnowledgeGraph.build(ext, res, {"c": "Pat acquired Quinn."})
    assert built.rejected == [("Pat", "ACQUIRED", "Quinn")]
    assert built.triples(typed_only=True) == set()
    assert ("Pat", "RELATED_TO", "Quinn") in built.triples(typed_only=False)


def test_json_roundtrip(rag, tmp_path):
    path = tmp_path / "kg.json"
    rag.save(path)
    data = json.loads(path.read_text())
    assert {"nodes", "edges", "communities", "stats"} <= set(data)
    kg2 = KnowledgeGraph.load_json(path)
    assert kg2.triples() == rag.kg.triples()
    assert kg2.g.number_of_nodes() == rag.kg.g.number_of_nodes()


# ----------------------------------------------------------------- communities
def test_communities_partition_and_hierarchy(rag):
    lvl0 = [c for c in rag.communities if c.level == 0]
    members = [m for c in lvl0 for m in c.members]
    assert sorted(members) == sorted(rag.kg.g.nodes)  # a partition
    ids = {c.id: c for c in rag.communities}
    for c in rag.communities:
        if c.parent:
            assert c.id in ids[c.parent].children
            assert set(c.members) <= set(ids[c.parent].members)
        assert c.summary and c.title


def test_communities_are_deterministic():
    a = GraphRAG(seed=7).index(DOCUMENTS).communities
    b = GraphRAG(seed=7).index(DOCUMENTS).communities
    assert [c.members for c in a] == [c.members for c in b]


def test_llm_community_summaries_use_llm():
    llm = MockLLM().on(r"Summarize this community", "LLM SUMMARY")
    rag = GraphRAG(llm=llm).index(DOCUMENTS[:6])
    assert all(c.summary == "LLM SUMMARY" for c in rag.communities)


# ----------------------------------------------------------------- local search
@pytest.mark.parametrize("item", QUESTIONS, ids=lambda q: q.question[:50])
def test_local_search_answers_benchmark(rag, item):
    ans = rag.local_search(item.question, max_hops=3)
    assert ans.answer == item.answer
    assert ans.paths and len(ans.paths[0]) >= 1


def test_multi_hop_path_explanation(rag):
    ans = rag.local_search("Who is the CEO of the company that acquired the company that acquired Quillon Systems?",
                           max_hops=3)
    assert ans.answer == "Ingrid Halvorsen"
    rels = [s.relation for s in ans.paths[0]]
    assert rels.count("ACQUIRED") == 2 and "CEO_OF" in rels
    assert "=> Ingrid Halvorsen" in ans.explanation
    assert set(ans.supporting_chunks) >= {"d04#0", "d06#0", "d05#0"}


def test_neighborhood_fallback_respects_max_hops(rag):
    q = "Tell me about the people around Solace Grid and their energy ties."
    one = rag.local_search(q, max_hops=1)
    two = rag.local_search(q, max_hops=2)
    assert one.method == two.method == "neighborhood"
    assert max(len(p) for p in one.paths) == 1
    assert two.paths and max(len(p) for p in two.paths) <= 2


def test_unknown_entity_returns_none(rag):
    ans = rag.local_search("Who is the CEO of Nonexistent Widgets?")
    assert ans.answer is None or ans.method == "neighborhood"


def test_local_search_llm_phrasing():
    llm = MockLLM().on(r"Knowledge-graph paths", "Alice Moreau (via graph)")
    rag = GraphRAG(llm=llm).index(DOCUMENTS)
    ans = rag.local_search("Who is the CEO of the company that acquired Quillon Systems?")
    assert ans.answer == "Alice Moreau (via graph)" and ans.method.endswith("+llm")
    assert "Zentrix Labs -[ACQUIRED]-> Quillon Systems" in llm.calls[-1]["prompt"]


# ----------------------------------------------------------------- global search / routing
def test_global_search_finds_acquisitions(rag):
    ans = rag.global_search("Summarize the major acquisitions in this industry.")
    assert isinstance(ans, GlobalAnswer)
    text = ans.answer
    hits = sum(name in text for name in ("Quillon Systems", "Zentrix Labs", "Tessel Analytics", "Solace Grid"))
    assert hits >= 3
    assert len(ans.communities_used) >= 2


@pytest.mark.parametrize("q", GLOBAL_QUESTIONS)
def test_global_questions_produce_points(rag, q):
    ans = rag.global_search(q)
    assert ans.points and ans.answer.startswith("Drawing on")


def test_global_search_map_reduce_with_llm():
    llm = (MockLLM()
           .on(r"Notes from community reports", "FINAL ANSWER")
           .on(r"Community report:", json.dumps({"points": [{"text": "Energy firms consolidate", "score": 80}]})))
    rag = GraphRAG().index(DOCUMENTS)
    rag.llm = llm
    ans = rag.global_search("What are the trends?")
    assert ans.answer == "FINAL ANSWER"
    n_map = sum("Community report:" in c["prompt"] for c in llm.calls)
    assert n_map == len([c for c in rag.communities if c.level == 0])


def test_router(rag):
    assert isinstance(rag.query("Who founded Quillon Systems?"), LocalAnswer)
    assert isinstance(rag.query("Give me an overview of the main themes."), GlobalAnswer)


def test_empty_corpus():
    rag = GraphRAG().index([Document(id="x", text="")])
    assert rag.kg.g.number_of_nodes() == 0 and rag.communities == []
    assert rag.global_search("anything").points == []


# ----------------------------------------------------------------- baseline
def test_graphrag_beats_vector_rag_on_multihop(rag):
    vec = VectorRAG(rag.chunks, k=3)
    multi = [q for q in QUESTIONS if q.hops >= 2]
    g = sum(rag.local_search(q.question, max_hops=3).answer == q.answer for q in multi)
    v = sum(vec.answer(q.question).answer == q.answer for q in multi)
    assert g > v


def test_vector_rag_llm_path(rag):
    llm = MockLLM().on(r"Passages:", "Dmitri Vasquez")
    vec = VectorRAG(rag.chunks, k=2, llm=llm)
    ans = vec.answer("Who founded Quillon Systems?")
    assert ans.answer == "Dmitri Vasquez" and len(ans.retrieved) == 2


def test_benchmark_is_json_serializable_and_shows_graph_advantage():
    from graphrag.bench import benchmark

    result = benchmark()
    json.dumps(result)
    s = result["systems"]
    assert s["graphrag_local"]["accuracy_multihop"] > s["vector_rag_top5"]["accuracy_multihop"]
    assert result["extraction_vs_gold"]["recall"] >= 0.9
