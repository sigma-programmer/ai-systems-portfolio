import pytest
from pydantic import ValidationError

from trajectory_eval.matchers import ArgMatcher, match_args, semantic_similarity, validate_schema
from trajectory_eval.schema import Step, StepType, Trajectory, TrajectorySuite


def test_tool_call_requires_tool():
    with pytest.raises(ValidationError):
        Step(type=StepType.TOOL_CALL)


def test_negative_tokens_rejected():
    with pytest.raises(ValidationError):
        Step(type=StepType.LLM, tokens=-1)


def test_signature_is_arg_order_independent():
    a = Step(type="tool_call", tool="t", args={"x": 1, "y": 2})
    b = Step(type="tool_call", tool="t", args={"y": 2, "x": 1})
    assert a.signature() == b.signature()


def test_trajectory_totals_and_tool_calls():
    t = Trajectory(run_id="r", task_id="k", steps=[
        Step(type="tool_call", tool="a", tokens=10, latency_ms=5),
        Step(type="observation", tokens=20, latency_ms=1),
        Step(type="llm", content="x", tokens=30)])
    assert t.total_tokens == 60 and t.total_latency_ms == 6
    assert [i for i, _ in t.tool_calls()] == [0]


def test_suite_unique_run_ids_and_roundtrip(tmp_path):
    t = Trajectory(run_id="r", task_id="k")
    with pytest.raises(ValidationError):
        TrajectorySuite(name="s", trajectories=[t, t])
    s = TrajectorySuite(name="s", trajectories=[t])
    s.save(tmp_path / "s.json")
    assert TrajectorySuite.load(tmp_path / "s.json") == s
    assert s.task_ids() == ["k"]


def test_exact_matcher_normalizes_json():
    m = ArgMatcher(key="a", kind="exact", value={"b": 1, "c": [1, 2]})
    assert m.matches({"a": {"c": [1, 2], "b": 1}})
    assert not m.matches({"a": {"c": [2, 1], "b": 1}})


def test_regex_matcher_fullmatch():
    m = ArgMatcher(key="id", kind="regex", pattern=r"ORD-\d{5}")
    assert m.matches({"id": "ORD-12345"})
    assert not m.matches({"id": "xORD-12345"})
    assert not m.matches({})


def test_regex_matcher_requires_valid_pattern():
    with pytest.raises(ValidationError):
        ArgMatcher(key="x", kind="regex")
    with pytest.raises(Exception):
        ArgMatcher(key="x", kind="regex", pattern="(")


def test_schema_matcher_and_validator():
    m = ArgMatcher(key="amount", kind="schema", schema={"type": "number", "minimum": 0, "maximum": 500})
    assert m.matches({"amount": 10})
    assert not m.matches({"amount": 501})
    assert not m.matches({"amount": "10"})
    assert not m.matches({"amount": True})
    errs = validate_schema({"a": [1, "x"]}, {"type": "object", "required": ["a", "b"],
                                             "properties": {"a": {"type": "array", "items": {"type": "integer"}}}})
    assert any("missing required key 'b'" in e for e in errs)
    assert any("$.a[1]" in e for e in errs)
    assert validate_schema("abc", {"type": "string", "pattern": "^a", "minLength": 2, "maxLength": 3}) == []
    assert validate_schema("q", {"enum": ["a", "b"]})


def test_semantic_matcher():
    m = ArgMatcher(key="q", kind="semantic", value="renewable energy adoption statistics", threshold=0.35)
    assert m.matches({"q": "global renewable energy adoption stats"})
    assert not m.matches({"q": "best pizza near me"})
    assert semantic_similarity("a b c", "a b c") == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValidationError):
        ArgMatcher(key="q", kind="semantic", value=3)


def test_optional_arg_and_any():
    assert ArgMatcher(key="x", kind="any").matches({"x": None})
    assert ArgMatcher(key="x", kind="exact", value=1, required=False).matches({})


def test_match_args_aggregates():
    ok, score, notes = match_args([ArgMatcher(key="a", kind="exact", value=1),
                                   ArgMatcher(key="b", kind="exact", value=2)], {"a": 1, "b": 3})
    assert not ok and score == 0.5 and "'b'" in notes[0]
    assert match_args([], {}) == (True, 1.0, [])
