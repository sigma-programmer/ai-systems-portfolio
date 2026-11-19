"""Workflow graph definitions and retry policy."""
from __future__ import annotations

import pytest

from durable_agent import END, FatalError, RetryableError, RetryPolicy, StepTimeout, Workflow, WorkflowDefinitionError, branch


def noop(ctx):
    return None


def test_start_defaults_to_first_step_and_validates() -> None:
    wf = Workflow("w")
    wf.step("a", next="b")(noop)
    wf.step("b")(noop)
    assert wf.start == "a"
    assert wf.validate() is wf
    assert "[task] a -> b" in wf.describe()


def test_unknown_target_rejected() -> None:
    wf = Workflow("w")
    wf.step("a", next="nope")(noop)
    with pytest.raises(WorkflowDefinitionError, match="unknown step 'nope'"):
        wf.validate()


def test_branch_targets_checked_statically() -> None:
    wf = Workflow("w")
    wf.step("a", next=branch(lambda o, c: o, {1: "b", 2: "zzz"}))(noop)
    wf.step("b")(noop)
    with pytest.raises(WorkflowDefinitionError, match="zzz"):
        wf.validate()


def test_gate_targets_checked() -> None:
    wf = Workflow("w")
    wf.step("a", next="g")(noop)
    wf.approval("g", on_approve="missing")
    with pytest.raises(WorkflowDefinitionError):
        wf.validate()


def test_duplicate_and_empty_and_bad_start() -> None:
    wf = Workflow("w")
    wf.step("a")(noop)
    with pytest.raises(WorkflowDefinitionError, match="duplicate"):
        wf.step("a")(noop)
    with pytest.raises(WorkflowDefinitionError, match="no steps"):
        Workflow("e").validate()
    bad = Workflow("w", start="x")
    bad.step("a")(noop)
    with pytest.raises(WorkflowDefinitionError, match="start"):
        bad.validate()


def test_bad_timeout_rejected() -> None:
    wf = Workflow("w")
    wf.step("a", timeout=0)(noop)
    with pytest.raises(WorkflowDefinitionError, match="timeout"):
        wf.validate()


def test_branch_semantics() -> None:
    b = branch(lambda o, c: o["k"], {"x": "s1", "y": END})
    assert b({"k": "x"}, None) == "s1"
    assert b({"k": "y"}, None) is None
    with pytest.raises(KeyError):
        b({"k": "z"}, None)
    d = branch(lambda o, c: o, {1: "a"}, default="b")
    assert d(9, None) == "b"
    assert d.targets() == {"a", "b"}


def test_fingerprint_changes_with_shape() -> None:
    def build(nxt):
        wf = Workflow("w")
        wf.step("a", next=nxt)(noop)
        wf.step("b")(noop)
        wf.step("c")(noop)
        return wf

    assert build("b").fingerprint() == build("b").fingerprint()
    assert build("b").fingerprint() != build("c").fingerprint()


def test_reject_fails_default() -> None:
    wf = Workflow("w")
    wf.step("a", next="g")(noop)
    g1 = wf.approval("g", on_reject=None)
    assert g1.reject_fails
    wf2 = Workflow("w2")
    wf2.step("a", next="g")(noop)
    assert not wf2.approval("g", on_reject="a").reject_fails


# ------------------------------------------------------------------ retry policy


def test_full_jitter_bounds_and_determinism() -> None:
    p = RetryPolicy(max_attempts=10, base_delay=1.0, multiplier=2.0, max_delay=8.0, seed=7)
    assert [p.cap(n) for n in range(1, 6)] == [1.0, 2.0, 4.0, 8.0, 8.0]
    for n in range(1, 8):
        d = p.delay(n, "run", "step")
        assert 0.0 <= d <= p.cap(n)
        assert d == p.delay(n, "run", "step")  # reproducible
    assert p.delay(3, "run-a", "s") != p.delay(3, "run-b", "s")  # decorrelated across runs
    assert RetryPolicy(seed=1).delay(2, "r", "s") != RetryPolicy(seed=2).delay(2, "r", "s")


def test_full_jitter_spreads_uniformly() -> None:
    p = RetryPolicy(base_delay=10.0, max_delay=10.0)
    ds = [p.delay(1, f"r{i}", "s") for i in range(2000)]
    mean = sum(ds) / len(ds)
    assert 4.5 < mean < 5.5
    assert min(ds) < 0.5 and max(ds) > 9.5


def test_no_jitter_is_exponential() -> None:
    p = RetryPolicy(base_delay=0.5, jitter=False, max_delay=100)
    assert [p.delay(n) for n in (1, 2, 3)] == [0.5, 1.0, 2.0]


def test_classification() -> None:
    p = RetryPolicy()
    assert p.is_retryable(RetryableError())
    assert p.is_retryable(StepTimeout())
    assert p.is_retryable(ConnectionError())
    assert p.is_retryable(TimeoutError())
    assert not p.is_retryable(FatalError())
    assert not p.is_retryable(ValueError())
    assert not p.is_retryable(RuntimeError())  # unknown -> fatal by default
    assert RetryPolicy(retry_unknown=True).is_retryable(RuntimeError())
    custom = RetryPolicy(retry_on=(RuntimeError,), fatal_on=(ConnectionResetError,))
    assert custom.is_retryable(RuntimeError())
    assert not custom.is_retryable(ConnectionResetError())


def test_policy_validation() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(multiplier=0.5)
