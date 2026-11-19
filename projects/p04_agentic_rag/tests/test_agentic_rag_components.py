"""Unit tests for text utils, retriever, grader, rewriter, verifier, and generators."""
from __future__ import annotations

import pytest

from agentic_rag import (BM25, HallucinatingGenerator, HeuristicEntailment, HeuristicGrader,
                         HybridRetriever, TermStats, decompose, load_corpus,
                         parse_cited_sentences)
from agentic_rag.adversarial import _flip_negation, _swap_unit, paraphrase
from agentic_rag.corpus import Document
from agentic_rag.generator import Claim, ExtractiveGenerator
from agentic_rag.rewriter import HeuristicRewriter, expand_synonyms, keywords
from agentic_rag.text import (content_terms, entities_in, expects_number, is_negated,
                              numbers_in, split_sentences, stem)
from agentic_rag.verifier import (CONTRADICTED, ENTAILED, UNSUPPORTED, HallucinationDetector,
                                  unit_mismatches)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def stats(corpus):
    return TermStats([d.title + " " + d.text for d in corpus])


@pytest.fixture(scope="module")
def retriever(corpus):
    return HybridRetriever(corpus)


# ---------------------------------------------------------------- text
def test_stem_is_consistent_across_inflections():
    assert stem("batteries") == stem("battery")
    assert stem("raised") == stem("raise")
    assert stem("costs") == stem("cost")
    assert stem("class") == "class"


def test_content_terms_drop_stopwords():
    assert content_terms("How many moons does Mars have?") == ["moon", "mar"]


def test_numbers_and_negation_and_entities():
    assert numbers_in("costs 1,200 dollars and 2.5 kg") == {"1200", "2.5"}
    assert is_negated("The Beacon is not waterproof")
    assert is_negated("It isn't waterproof")
    assert not is_negated("It is waterproof")
    assert "oslo" in entities_in("The company is in Oslo.")
    assert expects_number("How many days?") and not expects_number("Who made it?")


def test_split_sentences():
    assert split_sentences("A is 1. B is 2! C?") == ["A is 1.", "B is 2!", "C?"]
    assert split_sentences("") == []


# ---------------------------------------------------------------- retrieval
def test_bm25_ranks_matching_doc_first():
    bm = BM25([["cat", "sat"], ["dog", "ran"], ["cat", "cat", "purr"]])
    s = bm.scores(["cat"])
    assert s[1] == 0 and s[2] > 0 and s[0] > 0
    assert bm.term_idf("unseen") == bm.max_idf


def test_hybrid_retriever_finds_gold_doc(retriever):
    assert retriever.search("How long does the Beacon battery last?", 3)[0].doc_id == "beacon-specs"
    assert retriever.search("Guido van Rossum language", 3)[0].doc_id == "python"


def test_retriever_edge_cases(corpus):
    assert HybridRetriever([]).search("anything") == []
    assert HybridRetriever(corpus).search("   ") == []
    with pytest.raises(ValueError):
        HybridRetriever(corpus, alpha=1.5)


# ---------------------------------------------------------------- grading
def test_rare_missing_term_kills_relevance(stats, retriever):
    grader = HeuristicGrader(stats)
    ceo_doc = next(p for p in retriever.search("Lumora CEO", 4) if p.doc_id == "lumora-ceo")
    g = grader.grade("What is the salary of the Lumora CEO?", ceo_doc)
    assert not g.relevant and "salary" in g.missing_terms
    g2 = grader.grade("Who is the CEO of Lumora?", ceo_doc)
    assert g2.relevant and g2.coverage > 0.8


def test_synonym_match_gets_partial_credit(stats):
    full, _ = stats.coverage("vacation", "paid vacation")
    syn, _ = stats.coverage("vacation", "paid time off")
    assert full == 1.0 and 0 < syn < 1.0
    assert stats.coverage("", "anything") == (0.0, [])


# ---------------------------------------------------------------- rewriting
def test_decompose_resolves_pronouns():
    parts = decompose("When was Lumora founded and who is its CEO?")
    assert parts == ["When was Lumora founded?", "Who is Lumora's CEO?"]
    assert decompose("How tall is Mount Everest?") == ["How tall is Mount Everest?"]


def test_decompose_appends_entity_when_missing():
    parts = decompose("How many PTO days do employees get and how many public holidays?")
    assert len(parts) == 2 and "PTO" in parts[1]


def test_rewriter_escalates_strategies(stats):
    rw = HeuristicRewriter(stats)
    p1 = rw.rewrite("q", ["When was Lumora founded and who is its CEO?"], 1)
    assert p1.strategy == "decompose" and len(p1.needs) == 2
    p2 = rw.rewrite("q", ["How much does the Beacon cost?"], 1)
    assert p2.strategy == "synonym_expansion" and "pric" in p2.queries[p2.needs[0]][0]
    p3 = rw.rewrite("q", ["How much does the Beacon cost?"], 2)
    assert p3.strategy == "keyword_extraction"
    assert expand_synonyms("zzz") == "zzz"
    assert keywords("the salary of a ceo", stats).split()[0] == "salary"


# ---------------------------------------------------------------- verification
@pytest.fixture(scope="module")
def checker(stats):
    return HeuristicEntailment(stats)


SRC = ("The Beacon battery lasts 18 hours on a full charge. The Beacon is not waterproof. "
       "The company is headquartered in Oslo.")


@pytest.mark.parametrize("claim,label", [
    ("The Beacon battery lasts 18 hours on a full charge.", ENTAILED),
    ("The Beacon battery lasts 36 hours on a full charge.", CONTRADICTED),
    ("The Beacon battery lasts 18 minutes on a full charge.", CONTRADICTED),
    ("The Beacon is waterproof.", CONTRADICTED),
    ("The company is headquartered in Berlin.", CONTRADICTED),
    ("Lumora won a design award for quantum toasters.", UNSUPPORTED),
    ("In 2017 Lumora received a 12 million euro grant.", UNSUPPORTED),
])
def test_entailment_labels(checker, claim, label):
    assert checker.check(claim, SRC).label == label


def test_unit_mismatch_helper():
    assert unit_mismatches("lasts 18 minutes", "lasts 18 hours")
    assert not unit_mismatches("lasts 18 hours", "lasts 18 hours")


def test_detector_handles_missing_and_unknown_citations(checker):
    det = HallucinationDetector(checker)
    passages = {"d1": SRC}
    assert det.verify_claim(Claim("The Beacon is not waterproof.", []), passages).label == UNSUPPORTED
    v = det.verify_claim(Claim("The Beacon is not waterproof.", ["nope"]), passages)
    assert v.label == UNSUPPORTED and "unknown" in v.reason
    v = det.verify_claim(Claim("The Beacon is not waterproof.", ["nope", "d1"]), passages)
    assert v.label == ENTAILED and v.doc_id == "d1"


# ---------------------------------------------------------------- generators
def test_parse_cited_sentences():
    claims = parse_cited_sentences("Mars has two moons [mars]. It is red [mars][moon].\nNo cite.")
    assert [c.citations for c in claims] == [["mars"], ["mars", "moon"], []]
    assert claims[0].text == "Mars has two moons."


def test_extractive_generator_prefers_numeric_sentence(stats, retriever):
    gen = ExtractiveGenerator(stats)
    grader = HeuristicGrader(stats)
    need = "How much does the Beacon weigh?"
    hits = retriever.search(need, 2)
    claims = gen.generate(need, {need: [(h, grader.grade(need, h)) for h in hits]})
    assert claims[0].text == "The Beacon weighs 640 grams." and claims[0].citations == ["beacon-specs"]


def test_hallucinating_generator_marks_injections(stats, retriever, corpus):
    base = ExtractiveGenerator(stats)
    gen = HallucinatingGenerator(base, [d.doc_id for d in corpus], seed=1)
    grader = HeuristicGrader(stats)
    need = "How much does the Beacon weigh?"
    hits = retriever.search(need, 2)
    claims = gen.generate(need, {need: [(h, grader.grade(need, h)) for h in hits]})
    assert all(c.injected for c in claims)
    assert claims[-1].mutation == "fabricated"


def test_mutation_helpers():
    assert _flip_negation("It is not waterproof.") == "It is waterproof."
    assert "not" in _flip_negation("It is waterproof.")
    assert _swap_unit("lasts 18 hours") == "lasts 18 minutes"
    assert _swap_unit("no numbers") is None
    assert paraphrase("The lamp weighs 3 grams") == "The lamp has a weight of 3 grams"


def test_custom_corpus_document():
    d = Document("x", "X", "Text.")
    assert HybridRetriever([d]).search("text", 1)[0].doc_id == "x"
