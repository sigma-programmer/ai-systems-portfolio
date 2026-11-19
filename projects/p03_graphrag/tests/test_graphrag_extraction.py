"""Chunking, entity/relation extraction, LLM extractor plumbing, resolution."""
from __future__ import annotations

import json

import pytest

from aikit import MockLLM
from graphrag import (
    Document, EntityResolver, EntityType, HeuristicExtractor, LLMExtractor, RelationType, SentenceChunker,
)
from graphrag.extraction import find_entity_spans, parse_relation_type
from graphrag.text import normalize_name, split_sentences


def _extract(text: str, doc_id: str = "t"):
    chunks = SentenceChunker().chunk(Document(id=doc_id, text=text))
    return chunks, HeuristicExtractor().extract_document(chunks)


def _triples(results) -> set[tuple[str, str, str]]:
    return {(r.subject, r.relation.value, r.object) for res in results for r in res.relations
            if r.relation != RelationType.RELATED_TO}


# ----------------------------------------------------------------- text / chunking
def test_split_sentences_respects_abbreviations():
    s = split_sentences("Acme Inc. was founded in 1999. It grew fast! Did it? Yes.")
    assert s[0] == "Acme Inc. was founded in 1999."
    assert len(s) == 4


def test_normalize_name_strips_suffixes_and_articles():
    assert normalize_name("The Zentrix Labs, Inc.") == "zentrix labs"
    assert normalize_name("Halvorsen Group's") == "halvorsen group"


def test_chunker_respects_budget_and_ids():
    text = " ".join(f"Sentence number {i} talks about Zentrix Labs and Port Alder." for i in range(30))
    chunks = SentenceChunker(max_tokens=40).chunk(Document(id="d", text=text))
    assert len(chunks) > 3
    assert all(c.n_tokens <= 40 for c in chunks)
    assert [c.id for c in chunks] == [f"d#{i}" for i in range(len(chunks))]


def test_chunker_overlap_repeats_sentences():
    text = " ".join(f"Fact {i} is here today." for i in range(12))
    chunks = SentenceChunker(max_tokens=20, overlap_sentences=1).chunk(Document(id="d", text=text))
    for a, b in zip(chunks, chunks[1:]):
        assert split_sentences(a.text)[-1] == split_sentences(b.text)[0]


def test_chunker_handles_empty_and_invalid():
    assert SentenceChunker().chunk(Document(id="e", text="   ")) == []
    with pytest.raises(ValueError):
        SentenceChunker(max_tokens=0)


# ----------------------------------------------------------------- entities
def test_entity_types_by_heuristics():
    spans = {s.text: s.type for s in find_entity_spans(
        "Alice Moreau founded Zentrix Labs in Port Alder and launched the Atlas Chip.")}
    assert spans["Alice Moreau"] == EntityType.PERSON
    assert spans["Zentrix Labs"] == EntityType.ORG
    assert spans["Port Alder"] == EntityType.LOCATION
    assert spans["Atlas Chip"] == EntityType.PRODUCT


def test_sentence_initial_common_words_are_not_entities():
    spans = [s.text for s in find_entity_spans("Analysts praised the chip. Following the purchase, it grew.")]
    assert "Analysts" not in spans and "Following" not in spans


# ----------------------------------------------------------------- relations
@pytest.mark.parametrize("sentence, triple", [
    ("Zentrix Labs acquired Quillon Systems in 2021.", ("Zentrix Labs", "ACQUIRED", "Quillon Systems")),
    ("Quillon Systems was founded by Dmitri Vasquez.", ("Dmitri Vasquez", "FOUNDED", "Quillon Systems")),
    ("Priya Raman is the CEO of Orinoco Robotics.", ("Priya Raman", "CEO_OF", "Orinoco Robotics")),
    ("Caldera Energy is headquartered in Redmoor.", ("Caldera Energy", "LOCATED_IN", "Redmoor")),
    ("Brightwater Capital invested in Verdant Bio.", ("Brightwater Capital", "INVESTED_IN", "Verdant Bio")),
    ("Nimbus Freight is a subsidiary of Halvorsen Group.", ("Nimbus Freight", "SUBSIDIARY_OF", "Halvorsen Group")),
])
def test_pattern_relations(sentence, triple):
    _, results = _extract(sentence)
    assert triple in _triples(results)


def test_coordinated_subjects_and_coreference():
    _, results = _extract("Zentrix Labs was founded by Alice Moreau and Ben Okafor. "
                          "The company is headquartered in Port Alder.")
    t = _triples(results)
    assert ("Alice Moreau", "FOUNDED", "Zentrix Labs") in t
    assert ("Ben Okafor", "FOUNDED", "Zentrix Labs") in t
    assert ("Zentrix Labs", "LOCATED_IN", "Port Alder") in t


def test_cooccurrence_fallback_and_toggle():
    text = "Verdant Bio and Tessel Analytics both grew out of research programs."
    _, results = _extract(text)
    rel = [r for res in results for r in res.relations]
    assert any(r.relation == RelationType.RELATED_TO and r.confidence < 1 for r in rel)
    chunks = SentenceChunker().chunk(Document(id="x", text=text))
    off = HeuristicExtractor(cooccurrence=False).extract_document(chunks)
    assert not [r for res in off for r in res.relations]


def test_alias_pattern_detected():
    _, results = _extract("Nimbus Freight (NF) is a subsidiary of Halvorsen Group.")
    aliases = [a for res in results for a in res.aliases]
    assert ("NF", "Nimbus Freight") in aliases


# ----------------------------------------------------------------- LLM extractor
def test_parse_relation_type_synonyms_and_passive():
    assert parse_relation_type("founded by") == (RelationType.FOUNDED, True)
    assert parse_relation_type("headquartered_in") == (RelationType.LOCATED_IN, False)
    assert parse_relation_type("ceo") == (RelationType.CEO_OF, False)
    assert parse_relation_type("likes") == (RelationType.RELATED_TO, False)


def test_llm_extractor_parses_json_and_swaps_passive():
    payload = {
        "entities": [{"name": "Acme Corp", "type": "company"}, {"name": "Jane Roe", "type": "PER"}],
        "relations": [{"source": "Acme Corp", "relation": "FOUNDED_BY", "target": "Jane Roe", "evidence": "e"}],
    }
    llm = MockLLM().on(r"Extract entities", "```json\n" + json.dumps(payload) + "\n```")
    chunks = SentenceChunker().chunk(Document(id="a", text="Acme Corp was founded by Jane Roe."))
    res = LLMExtractor(llm).extract_document(chunks)[0]
    types = {m.text: m.type for m in res.mentions}
    assert types == {"Acme Corp": EntityType.ORG, "Jane Roe": EntityType.PERSON}
    assert [(r.subject, r.relation, r.object) for r in res.relations] == [("Jane Roe", RelationType.FOUNDED, "Acme Corp")]
    assert llm.calls[0]["json_mode"] is True


def test_llm_extractor_falls_back_on_garbage():
    llm = MockLLM(default="sorry, I cannot do that")
    ex = LLMExtractor(llm)
    chunks = SentenceChunker().chunk(Document(id="a", text="Zentrix Labs acquired Quillon Systems."))
    res = ex.extract_document(chunks)[0]
    assert ex.failures == 1
    assert any(r.relation == RelationType.ACQUIRED for r in res.relations)
    strict = LLMExtractor(llm, strict=True)
    with pytest.raises(ValueError):
        strict.extract(chunks[0])


# ----------------------------------------------------------------- resolution
def test_resolution_merges_suffix_surname_acronym_and_typo():
    text = ("Zentrix Labs Inc. was founded by Ben Okafor. Okafor is the CTO of Zentrix Labs. "
            "Nimbus Freight (NF) is a subsidiary of Halvorsen Group. NF is headquartered in Eastmere. "
            "Quillon Systems is based in Northvale. Analysts like Quilon Systems.")
    _, results = _extract(text)
    res = EntityResolver().resolve(results)
    assert res.lookup("Zentrix Labs") == res.lookup("Zentrix Labs Inc.") is not None
    assert res.lookup("Okafor") == res.lookup("Ben Okafor")
    assert res.lookup("NF") == res.lookup("Nimbus Freight")
    assert res.lookup("Quilon Systems") == res.lookup("Quillon Systems")
    assert res.lookup("Halvorsen Group") != res.lookup("Nimbus Freight")


def test_resolution_keeps_distinct_people_apart():
    _, results = _extract("Erik Halvorsen founded Halvorsen Group. Ingrid Halvorsen is the CEO of Halvorsen Group.")
    res = EntityResolver().resolve(results)
    ids = {res.lookup("Erik Halvorsen"), res.lookup("Ingrid Halvorsen"), res.lookup("Halvorsen Group")}
    assert len(ids) == 3
