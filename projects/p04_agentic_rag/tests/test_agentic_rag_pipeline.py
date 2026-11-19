"""End-to-end tests of the controller loop, abstention, repair, LLM plumbing, and bench."""
from __future__ import annotations

import json

import pytest

from aikit import MockLLM
from agentic_rag import (AgenticRAG, Confidence, ExtractiveGenerator, HallucinatingGenerator,
                         LLMEntailment, LLMGenerator, LLMGrader, LLMRewriter)
from agentic_rag.bench import benchmark
from agentic_rag.dataset import ANSWERABLE, UNANSWERABLE
from agentic_rag.generator import Claim


@pytest.fixture(scope="module")
def rag():
    return AgenticRAG()


@pytest.mark.parametrize("q", ANSWERABLE, ids=lambda q: q.question[:40])
def test_answerable_questions_are_answered_with_gold_facts(rag, q):
    r = rag.run(q.question)
    assert not r.abstained
    for a in q.answers:
        assert a.lower() in r.answer.lower()
    assert set(r.citations) <= set(q.gold_docs)
    assert all(not c.flagged for c in r.checks)


@pytest.mark.parametrize("q", UNANSWERABLE, ids=lambda q: q.question[:40])
def test_unanswerable_questions_abstain(rag, q):
    r = rag.run(q.question)
    assert r.abstained and r.answer.startswith("I don't know")
    assert r.claims == [] and r.confidence < rag.abstain_threshold


def test_abstention_explains_missing_terms(rag):
    r = rag.run("What is the salary of the Lumora CEO?")
    assert "salary" in r.answer and r.missing_terms[0] == "salary"


def test_rewrite_loop_bounded_by_max_steps():
    rag = AgenticRAG(max_steps=2)
    r = rag.run("What is the dental insurance plan at Lumora?")
    assert r.trace.actions().count("sufficiency") == 2
    assert r.trace.actions().count("rewrite") == 1
    assert AgenticRAG(max_steps=0).max_steps == 1


def test_multi_part_trace_has_plan_and_two_needs(rag):
    r = rag.run("When was Lumora founded and who is its CEO?")
    plan = r.trace.steps[0]
    assert plan.action == "plan" and len(plan.data["needs"]) == 2
    assert json.dumps(r.to_dict())  # trace is JSON-serialisable
    assert "plan" in r.trace.pretty()


def test_decomposition_happens_via_rewrite_when_not_upfront():
    rag = AgenticRAG(decompose_upfront=False)
    r = rag.run("What is the Beacon battery life and how much does it weigh?")
    assert "18 hours" in r.answer
    assert r.trace.steps[0].data["needs"] == [r.question]


def test_empty_question_abstains(rag):
    assert rag.run("   ").abstained


def test_adversarial_generator_is_caught_and_repaired(rag):
    liar = AgenticRAG(generator=HallucinatingGenerator(
        ExtractiveGenerator(rag.stats), [d.doc_id for d in rag.docs], seed=3))
    r = liar.run("How long is parental leave at Lumora?")
    injected = [c for c in r.checks if c.claim.injected]
    assert injected and all(c.flagged for c in injected)
    assert "repair" in r.trace.actions()
    assert "16 weeks" in r.answer
    final_texts = {c.text for c in r.claims}
    assert not any(c.claim.text in final_texts for c in injected
                   if c.claim.mutation != "miscite")
    assert r.signals["grounding"] < 1.0


def test_miscited_claim_is_recited():
    class Miscite:
        def generate(self, question, evidence):
            return [Claim("The Beacon weighs 640 grams.", ["mars"], next(iter(evidence)))]

    r = AgenticRAG(generator=Miscite()).run("How much does the Beacon weigh?")
    assert r.claims[0].citations == ["beacon-specs"]
    repair = next(s for s in r.trace.steps if s.action == "repair")
    assert repair.data["actions"][0]["action"] == "recited"


def test_low_confidence_abstains_even_with_claims():
    strict = AgenticRAG(abstain_threshold=0.999)
    r = strict.run("How many vacation days do Lumora employees get?")
    assert r.abstained and r.confidence < 0.999


def test_confidence_is_monotone():
    c = Confidence()
    assert c.score(0.9, 1.0, 1.0) > c.score(0.9, 1.0, 0.0) > c.score(0.3, 0.0, 0.0)
    assert 0.0 < c.score(0, 0, 0) < c.score(1, 1, 1) < 1.0


# ---------------------------------------------------------------- LLM plumbing (MockLLM)
def test_llm_components_plumb_through_protocol():
    llm = MockLLM()
    llm.on(r"REWRITE QUERY", json.dumps({"sub_questions": ["How much does the Beacon weigh?"],
                                         "queries": {"How much does the Beacon weigh?":
                                                     ["Beacon weight grams"]}}))
    llm.on(r"GRADE RELEVANCE.*beacon-specs", '{"score": 0.9, "missing": []}')
    llm.on(r"GRADE RELEVANCE", '{"score": 0.1, "missing": ["weight"]}')
    llm.on(r"GENERATE ANSWER", "The Beacon weighs 640 grams [beacon-specs].")
    llm.on(r"VERIFY CLAIM", '{"label": "entailed", "reason": "stated"}')
    rag = AgenticRAG(grader=LLMGrader(llm), rewriter=LLMRewriter(llm),
                     generator=LLMGenerator(llm), checker=LLMEntailment(llm))
    r = rag.run("Beacon heft?")
    assert not r.abstained
    assert r.claims[0].citations == ["beacon-specs"]
    prompts = " ".join(c["prompt"] for c in llm.calls)
    for tag in ("GRADE RELEVANCE", "GENERATE ANSWER", "VERIFY CLAIM"):
        assert tag in prompts


def test_llm_rewriter_used_when_grader_rejects():
    llm = MockLLM()
    llm.on(r"REWRITE QUERY", json.dumps({"sub_questions": ["Beacon weight"],
                                         "queries": {"Beacon weight": ["Beacon weighs grams"]}}))
    rag = AgenticRAG(rewriter=LLMRewriter(llm))
    r = rag.run("How heavy is that lamp thing, the Beacon?")
    assert "rewrite" in r.trace.actions() or not r.abstained
    if "rewrite" in r.trace.actions():
        assert any(s.data.get("strategy") == "llm" for s in r.trace.steps)


def test_llm_entailment_rejects_unknown_label():
    llm = MockLLM(default='{"label": "maybe"}')
    assert LLMEntailment(llm).check("a", "b").label == "unsupported"


# ---------------------------------------------------------------- benchmark
def test_benchmark_metrics():
    m = benchmark(seeds=(0, 1))
    json.dumps(m)
    assert m["answer_accuracy"] >= 0.9
    assert m["abstention_precision"] >= 0.9 and m["abstention_recall"] >= 0.9
    assert m["hallucination_catch_rate"] >= 0.9
    assert m["citation_precision"] >= 0.9
    assert m["hallucinations_leaked_to_final_answer"] == 0
