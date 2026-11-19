"""Agents: researcher retrieval, CEGIS coder, reviewers, LLM repair/fallback."""
from __future__ import annotations

import json

import pytest

from aikit import MockLLM
from agent_swarm import (
    TASKS, TASKS_BY_NAME, CodeArtifact, Coder, EvidenceKind, ExecutionReviewer, FaultyReviewer, HandoffError,
    LLMReviewer, Researcher, ResearchFindings, RevisionRequest, SamplingReviewer, StaticReviewer,
    SynthesisExhausted, TestCase, parse_edge_line,
)
from agent_swarm.library import CANDIDATES


def test_parse_edge_line():
    assert parse_edge_line("EDGE: f([1, 2], 'a') -> {'x': 1}") == ("f", [[1, 2], "a"], {"x": 1}, None)
    assert parse_edge_line("EDGE: f([]) raises ValueError") == ("f", [[]], None, "ValueError")
    assert parse_edge_line("EDGE: f(oops) -> 1") is None
    assert parse_edge_line("REQ: nothing") is None


@pytest.mark.parametrize("td", TASKS, ids=lambda t: t.spec.name)
def test_researcher_finds_the_right_snippet(td, make_ctx):
    r = Researcher()
    f = r.handle([td.spec], make_ctx(td.spec.name))
    assert isinstance(f, ResearchFindings)
    assert f.edge_cases, "should find at least one edge case"
    assert all(e.source == "research" for e in f.edge_cases)
    assert len(f.sources) == 1


def test_researcher_filters_distractors_and_dedups(make_ctx):
    f = Researcher().handle([TASKS_BY_NAME["palindrome"].spec], make_ctx())
    assert f.sources == ["kb-palindrome"]  # kb-palnum is about is_palindrome_number
    assert any("case insensitive" in r for r in f.requirements)
    assert len({json.dumps(e.args) for e in f.edge_cases}) == len(f.edge_cases)


def test_researcher_with_empty_knowledge(make_ctx):
    f = Researcher(knowledge=[]).handle([TASKS_BY_NAME["palindrome"].spec], make_ctx())
    assert f.edge_cases == [] and f.requirements == [TASKS_BY_NAME["palindrome"].spec.description]


def test_coder_rejects_unaccepted_input(make_ctx):
    art = CodeArtifact(task_name="t", func_name="f", source="def f():\n    pass\n", revision=0)
    with pytest.raises(HandoffError, match="does not accept CodeArtifact"):
        Coder().handle([art], make_ctx())


def test_coder_cegis_uses_counterexamples(make_ctx):
    spec = TASKS_BY_NAME["roman"].spec
    ctx = make_ctx("roman")
    first = Coder().handle([spec], ctx)
    assert first.origin == "library[0]"  # naive sum is consistent with III and VIII
    cex = TestCase(id="r0", args=["IV"], expected=4, source="research")
    rr = RevisionRequest(task_name="roman", revision=1, counterexamples=[cex], rejected_hashes=[first.hash])
    second = Coder().handle([spec, rr], ctx)
    assert second.revision == 1 and second.hash != first.hash
    assert ctx.sandbox.run(second.source, spec.func_name, TASKS_BY_NAME["roman"].hidden).passed


def test_coder_repairs_binary_search_by_mutation(make_ctx):
    spec = TASKS_BY_NAME["binary_search"].spec
    ctx = make_ctx("binary_search")
    findings = Researcher().handle([spec], ctx)  # requirements steer ranking towards the binary-search template
    art = Coder().handle([findings], ctx)
    assert art.origin.startswith("mutant(library[1]")
    assert "lo <= hi" in art.source


def test_coder_exhaustion(make_ctx):
    spec = TASKS_BY_NAME["roman"].spec
    hashes = []
    from agent_swarm.synthesis import build_pool
    hashes = [c.hash for c in build_pool("roman_to_int")]
    rr = RevisionRequest(task_name="roman", revision=1, rejected_hashes=hashes)
    with pytest.raises(SynthesisExhausted):
        Coder().handle([spec, rr], make_ctx("roman"))


def _findings(ctx, name):
    f = Researcher().handle([TASKS_BY_NAME[name].spec], ctx)
    ctx.state.findings = f
    return f


def _art(name, idx):
    spec = TASKS_BY_NAME[name].spec
    return CodeArtifact(task_name=name, func_name=spec.func_name, source=CANDIDATES[spec.func_name][idx].source,
                        revision=0)


def test_execution_reviewer(make_ctx):
    ctx = make_ctx("median")
    _findings(ctx, "median")
    bad = ExecutionReviewer().handle([_art("median", 0)], ctx)
    assert not bad.approved and bad.strength == EvidenceKind.TEST_EXECUTION and bad.counterexamples
    assert all(f.startswith("FAIL median(") for f in bad.feedback)
    good = ExecutionReviewer().handle([_art("median", 2)], ctx)
    assert good.approved and good.counterexamples == []


def test_reviewer_rejects_unsafe_code_statically(make_ctx):
    ctx = make_ctx("median")
    art = CodeArtifact(task_name="median", func_name="median", revision=0,
                       source="import subprocess\ndef median(xs):\n    return 0\n")
    for rv in (ExecutionReviewer(), StaticReviewer()):
        v = rv.handle([art], ctx)
        assert not v.approved and v.strength == EvidenceKind.STATIC_ANALYSIS


def test_sampling_reviewer_is_deterministic_subset(make_ctx):
    ctx = make_ctx("median")
    f = _findings(ctx, "median")
    rv = SamplingReviewer(fraction=0.5, seed=3)
    art = _art("median", 1)
    picked = rv.select_tests(f.test_suite(), art)
    assert picked == rv.select_tests(f.test_suite(), art)
    assert len(picked) == round(len(f.test_suite()) * 0.5)
    v = rv.handle([art], ctx)
    assert v.evidence[0].tests_run == len(picked) and v.confidence < 0.95


def test_static_reviewer_approves_buggy_but_safe_code(make_ctx):
    v = StaticReviewer().handle([_art("median", 0)], make_ctx("median"))
    assert v.approved and v.strength == EvidenceKind.STATIC_ANALYSIS


def test_faulty_reviewer_modes(make_ctx):
    ctx = make_ctx("median")
    _findings(ctx, "median")
    good, bad = _art("median", 2), _art("median", 0)
    adv = FaultyReviewer(mode="adversarial")
    assert adv.handle([good], ctx).approved is False
    assert adv.handle([bad], ctx).approved is True
    assert FaultyReviewer(mode="always_approve").handle([bad], ctx).approved
    with pytest.raises(ValueError):
        FaultyReviewer(mode="nope")


def test_llm_coder_repairs_invalid_output(make_ctx):
    replies = iter(["not json at all", json.dumps({"source": "def wrong():\n    pass\n"}),
                    json.dumps({"source": "def median(xs):\n    return sorted(xs)[len(xs) // 2]\n"})])
    llm = MockLLM(default=lambda p, s: next(replies))
    ctx = make_ctx("median")
    art = Coder(llm=llm).handle([TASKS_BY_NAME["median"].spec], ctx)
    assert art.origin == "llm:mock-llm"
    assert ctx.state.repairs == 2 and len(llm.calls) == 3
    assert "VALIDATION ERRORS" in llm.calls[1]["prompt"]
    assert "must define top-level function" in llm.calls[2]["prompt"]
    assert ctx.state.tokens > 0


def test_llm_coder_falls_back_to_synthesis(make_ctx):
    events = []
    llm = MockLLM(default="I refuse to output JSON")
    ctx = make_ctx("median")
    ctx.emit = lambda kind, node="", detail="", tokens=0: events.append(kind)
    art = Coder(llm=llm, max_repairs=1).handle([TASKS_BY_NAME["median"].spec], ctx)
    assert art.origin.startswith("library") and "fallback" in events
    assert len(llm.calls) == 2


def test_llm_coder_prompt_includes_counterexamples(make_ctx):
    llm = MockLLM(default=json.dumps({"source": "def median(xs):\n    return 0\n"}))
    cex = TestCase(id="r0", args=[[1, 2, 3, 4]], expected=2.5, source="research")
    rr = RevisionRequest(task_name="median", revision=1, counterexamples=[cex], feedback=["FAIL even length"])
    Coder(llm=llm).handle([TASKS_BY_NAME["median"].spec, rr], make_ctx("median"))
    assert "REJECTED" in llm.calls[0]["prompt"] and "median([1, 2, 3, 4]) == 2.5" in llm.calls[0]["prompt"]


def test_llm_researcher_and_reviewer(make_ctx):
    llm = MockLLM().on("Researcher", json.dumps({
        "requirements": ["even length averages"], "edge_cases": [{"args": [[1, 2]], "expected": 1.5}]}))
    llm.on("code reviewer", json.dumps({"approved": False, "confidence": 0.7, "feedback": ["no empty check"]}))
    ctx = make_ctx("median")
    f = Researcher(llm=llm).handle([TASKS_BY_NAME["median"].spec], ctx)
    assert f.edge_cases[0].id == "r0" and f.edge_cases[0].expected == 1.5
    v = LLMReviewer(llm=llm).handle([_art("median", 0)], ctx)
    assert not v.approved and v.strength == EvidenceKind.OPINION and v.feedback == ["no empty check"]


def test_llm_reviewer_default_is_opinion(make_ctx):
    v = LLMReviewer().handle([_art("median", 0)], make_ctx("median"))
    assert v.strength == EvidenceKind.OPINION


def test_handoff_error_when_default_output_invalid(make_ctx):
    class BrokenReviewer(StaticReviewer):
        def act_default(self, inputs, ctx):
            return {"reviewer": "x"}
    with pytest.raises(HandoffError, match="invalid ReviewVerdict"):
        BrokenReviewer().handle([_art("median", 0)], make_ctx("median"))
