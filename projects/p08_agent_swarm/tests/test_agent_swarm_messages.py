"""Validation contracts on typed messages."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_swarm import (
    CodeArtifact, ConsensusDecision, Evidence, EvidenceKind, ResearchFindings, ReviewVerdict, TASKS_BY_NAME,
    TaskSpec, TestCase,
)
from agent_swarm.messages import source_hash


def test_code_artifact_requires_parseable_source():
    with pytest.raises(ValidationError, match="does not parse"):
        CodeArtifact(task_name="t", func_name="f", source="def f(:\n", revision=0)


def test_code_artifact_requires_named_function():
    with pytest.raises(ValidationError, match="must define top-level function 'f'"):
        CodeArtifact(task_name="t", func_name="f", source="def g():\n    return 1\n", revision=0)
    art = CodeArtifact(task_name="t", func_name="f", source="def f():\n    return 1\n", revision=0)
    assert len(art.hash) == 12


def test_source_hash_ignores_formatting_and_comments():
    assert source_hash("def f(x):\n    return x  # c\n") == source_hash("def f( x ):\n  return x\n")
    assert source_hash("def f(x):\n    return x\n") != source_hash("def f(x):\n    return -x\n")


def test_verdict_cannot_approve_with_failing_test_evidence():
    ev = Evidence(kind=EvidenceKind.TEST_EXECUTION, tests_run=3, failed_test_ids=["ex0"])
    with pytest.raises(ValidationError, match="approves code while"):
        ReviewVerdict(reviewer="r", strategy="s", artifact_hash="h", approved=True, confidence=0.9, evidence=[ev])
    v = ReviewVerdict(reviewer="r", strategy="s", artifact_hash="h", approved=False, confidence=0.9, evidence=[ev])
    assert v.strength == EvidenceKind.TEST_EXECUTION


def test_evidence_consistency():
    with pytest.raises(ValidationError):
        Evidence(kind=EvidenceKind.TEST_EXECUTION, tests_run=0)
    with pytest.raises(ValidationError):
        Evidence(kind=EvidenceKind.TEST_EXECUTION, tests_run=1, failed_test_ids=["a", "b"])


def test_verdict_requires_evidence_and_bounded_confidence():
    ev = Evidence(kind=EvidenceKind.OPINION)
    with pytest.raises(ValidationError):
        ReviewVerdict(reviewer="r", strategy="s", artifact_hash="h", approved=True, confidence=0.5, evidence=[])
    with pytest.raises(ValidationError):
        ReviewVerdict(reviewer="r", strategy="s", artifact_hash="h", approved=True, confidence=1.5, evidence=[ev])


def test_task_spec_and_testcase_validation():
    with pytest.raises(ValidationError):
        TaskSpec(name="x", func_name="not valid", params=[], description="a long enough description",
                 examples=[TestCase(id="e", args=[])])
    with pytest.raises(ValidationError):
        TaskSpec(name="x", func_name="f", params=[], description="a long enough description", examples=[])
    with pytest.raises(ValidationError):
        TestCase(id="e", args=[], raises="not an exception")
    assert TestCase(id="e", args=[1, "a"], expected=2).describe("f") == "f(1, 'a') == 2"
    assert TestCase(id="e", args=[], raises="ValueError").describe("g") == "g() raises ValueError"


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        TestCase(id="e", args=[], bogus=1)


def test_findings_unique_ids_and_suite():
    spec = TASKS_BY_NAME["palindrome"].spec
    e = TestCase(id="r0", args=[""], expected=True, source="research")
    with pytest.raises(ValidationError, match="unique"):
        ResearchFindings(task=spec, requirements=[], edge_cases=[e, e])
    f = ResearchFindings(task=spec, requirements=[], edge_cases=[e])
    assert [t.id for t in f.test_suite()] == ["ex0", "ex1", "r0"]


def test_consensus_decision_share_bounds():
    with pytest.raises(ValidationError):
        ConsensusDecision(artifact_hash="h", approved=True, rule="m", resolution="vote", approve_share=1.2, votes=[])
