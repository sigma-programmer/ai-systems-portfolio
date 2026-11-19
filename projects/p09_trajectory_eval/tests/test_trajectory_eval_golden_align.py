import pytest
from pydantic import ValidationError

from trajectory_eval.align import StepStatus, align, detect_loops
from trajectory_eval.datagen import DATA_DIR
from trajectory_eval.golden import GoldenDAG, GoldenNode, load_goldens
from trajectory_eval.matchers import ArgMatcher
from trajectory_eval.schema import Step, Trajectory


def tc(tool, **args):
    return Step(type="tool_call", tool=tool, args=args)


def obs(error=False):
    return Step(type="observation", error=error)


@pytest.fixture
def diamond():
    """a -> (b || c) -> d, e optional after d, f forbidden."""
    return GoldenDAG(
        task_id="t",
        nodes=[GoldenNode(id=x, tool=x) for x in "abcd"]
        + [GoldenNode(id="e", tool="e", optional=True), GoldenNode(id="f", tool="f", forbidden=True)],
        edges=[("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", "e")],
        answer=ArgMatcher(key="answer", kind="regex", pattern=r"(?i).*done.*"),
    )


def run(dag, tools, answer="done", **kw):
    return align(Trajectory(run_id="r", task_id="t", steps=[tc(t) for t in tools], final_answer=answer), dag, **kw)


def test_golden_rejects_cycles_unknown_nodes_and_forbidden_edges():
    with pytest.raises(ValidationError, match="cycle"):
        GoldenDAG(task_id="x", nodes=[GoldenNode(id="a", tool="a"), GoldenNode(id="b", tool="b")],
                  edges=[("a", "b"), ("b", "a")])
    with pytest.raises(ValidationError, match="unknown"):
        GoldenDAG(task_id="x", nodes=[GoldenNode(id="a", tool="a")], edges=[("a", "z")])
    with pytest.raises(ValidationError, match="forbidden"):
        GoldenDAG(task_id="x", nodes=[GoldenNode(id="a", tool="a"), GoldenNode(id="f", tool="f", forbidden=True)],
                  edges=[("a", "f")])
    with pytest.raises(ValidationError, match="duplicate"):
        GoldenDAG(task_id="x", nodes=[GoldenNode(id="a", tool="a"), GoldenNode(id="a", tool="b")])


def test_golden_roundtrip_and_closure(tmp_path, diamond):
    diamond.save(tmp_path / "g.json")
    g2 = GoldenDAG.load(tmp_path / "g.json")
    assert g2.edges == diamond.edges and g2.answer is not None
    assert ("a", "d") in diamond.ordering_pairs()
    assert diamond.required_ids == ["a", "b", "c", "d"]


@pytest.mark.parametrize("order", [["a", "b", "c", "d"], ["a", "c", "b", "d"], ["a", "c", "b", "d", "e"]])
def test_any_order_parallel_branches_pass(diamond, order):
    r = run(diamond, order)
    assert r.passed, r.failure_reasons
    assert r.metrics.required_recall == 1.0 and r.metrics.precision == 1.0 and r.metrics.order_score == 1.0
    assert r.metrics.score == pytest.approx(1.0)


def test_missing_required_and_optional(diamond):
    r = run(diamond, ["a", "b", "d"])
    assert r.missing_required == ["c"] and r.missing_optional == ["e"]
    assert r.metrics.required_recall == 0.75 and not r.passed


def test_order_violation_detected(diamond):
    r = run(diamond, ["a", "d", "b", "c"])
    assert not r.passed
    assert {(v.before, v.after) for v in r.order_violations} == {("b", "d"), ("c", "d")}
    assert r.step_grades[1].status == StepStatus.OUT_OF_ORDER
    assert r.metrics.order_score < 1.0


def test_forbidden_and_extra(diamond):
    r = run(diamond, ["a", "f", "b", "zzz", "c", "d"])
    statuses = [g.status for g in r.step_grades]
    assert statuses[1] == StepStatus.FORBIDDEN and statuses[3] == StepStatus.EXTRA
    assert r.metrics.forbidden == 1 and r.metrics.extra == 1
    assert r.metrics.precision == pytest.approx(4 / 6)
    assert r.metrics.score < 0.6  # forbidden halves the score
    assert any("forbidden" in x for x in r.failure_reasons)


def test_redundant_and_loop(diamond):
    r = run(diamond, ["a", "b", "b", "b", "c", "d"])
    assert r.metrics.redundant == 2 and r.metrics.loops == 1
    assert all(g.in_loop for g in r.step_grades[1:4])
    assert not r.passed


def test_retry_after_error_is_not_redundant(diamond):
    steps = [tc("a"), obs(error=True), tc("a"), obs(), tc("b"), tc("c"), tc("d")]
    r = align(Trajectory(run_id="r", task_id="t", steps=steps, final_answer="done"), diamond)
    assert r.step_grades[2].status == StepStatus.REPEAT
    assert r.passed


def test_wrong_answer_fails(diamond):
    r = run(diamond, ["a", "b", "c", "d"], answer="nope")
    assert r.metrics.answer_correct is False and not r.passed


def test_arg_mismatch_and_forbidden_by_args():
    g = GoldenDAG(task_id="t", nodes=[
        GoldenNode(id="pay", tool="pay", args=[ArgMatcher(key="amt", kind="schema", schema={"type": "number", "maximum": 100})]),
        GoldenNode(id="big", tool="pay", forbidden=True,
                   args=[ArgMatcher(key="amt", kind="schema", schema={"type": "number", "minimum": 100.01})])])
    r = align(Trajectory(run_id="r", task_id="t", steps=[tc("pay", amt="x"), tc("pay", amt=500), tc("pay", amt=50)]), g)
    assert [s.status for s in r.step_grades] == [StepStatus.ARG_MISMATCH, StepStatus.FORBIDDEN, StepStatus.MATCHED]
    assert any("mismatched" in x for x in r.failure_reasons)


def test_same_tool_two_nodes_assigned_by_readiness():
    g = load_goldens(DATA_DIR / "golden")["code_fix"]
    steps = [tc("run_tests", cmd="pytest -q"), tc("read_file", path="src/a.py"),
             tc("edit_file", path="src/a.py", patch="fix the bug"), tc("run_tests", cmd="pytest -q")]
    r = align(Trajectory(run_id="r", task_id="code_fix", steps=steps, final_answer="all tests pass"), g)
    assert [s.node_id for s in r.step_grades] == ["run_tests_before", "read_file", "edit_file", "run_tests_after"]
    assert r.passed


def test_llm_node_semantic_match():
    g = load_goldens(DATA_DIR / "golden")["research_summary"]
    steps = [tc("web_search", query="renewable energy adoption statistics 2025"),
             tc("fetch_url", url="https://x.org/a"),
             Step(type="llm", content="Summary of renewable energy adoption findings from the sources.")]
    r = align(Trajectory(run_id="r", task_id="research_summary", steps=steps,
                         final_answer="Renewable energy adoption grew; solar and wind capacity share rose."), g)
    assert r.passed, r.failure_reasons
    steps[2] = Step(type="llm", content="Let me tell you a joke about cats.")
    r2 = align(Trajectory(run_id="r", task_id="research_summary", steps=steps, final_answer="x"), g)
    assert "summarize" in r2.missing_required


def test_budget():
    g = GoldenDAG(task_id="t", nodes=[GoldenNode(id="a", tool="a")], max_steps=2)
    r = align(Trajectory(run_id="r", task_id="t", steps=[tc("a"), obs(), obs()]), g)
    assert r.metrics.over_budget and not r.passed


def test_detect_loops_periods():
    sig = lambda s: [(i, x) for i, x in enumerate(s)]  # noqa: E731
    assert detect_loops(sig("aaab")) [0].period == 1
    lp = detect_loops(sig("xababab"))
    assert lp[0].period == 2 and lp[0].repeats == 3 and lp[0].start == 1
    assert detect_loops(sig("abcabd")) == []
    assert detect_loops(sig("aab")) == []


def test_bundled_goldens_load():
    gs = load_goldens(DATA_DIR / "golden")
    assert set(gs) == {"refund_request", "trip_planner", "code_fix", "research_summary"}
    assert all(g.forbidden_nodes for g in gs.values())
