"""Human-in-the-loop approval gates, escalation, timeouts, signals and cancellation."""
from __future__ import annotations

from pathlib import Path

import pytest

from durable_agent import END, ApprovalError, Engine, FakeClock, InvalidStateError, Status, Workflow


def gated(on_reject="write", **gate_kwargs) -> Workflow:
    wf = Workflow("gated")
    wf.step("write", next="review")(lambda ctx: {"draft": f"v{ctx.visit}"})
    wf.approval(
        "review",
        on_approve="ship",
        on_reject=on_reject,
        approvers=gate_kwargs.pop("approvers", {"alice"}),
        prepare=lambda ctx: {"draft": ctx.outputs["write"]["draft"]},
        **gate_kwargs,
    )
    wf.step("ship")(lambda ctx: {"shipped": ctx.outputs["write"]["draft"], "by": ctx.approvals["review"]["approver"]})
    return wf


def test_gate_suspends_and_is_approved_from_another_engine(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    clock = FakeClock()
    eng1 = Engine(db, [gated()], clock=clock)
    info = eng1.execute("gated", {})
    assert info.status == Status.WAITING_APPROVAL
    assert info.pending_approval["gate"] == "review"
    assert info.pending_approval["request"] == {"draft": "v1"}
    assert [r.run_id for r in eng1.pending_approvals()] == [info.run_id]
    eng1.store.close()  # "process 1" goes away

    eng2 = Engine(db, [gated()], clock=clock)  # e.g. a web handler in another process
    done = eng2.approve(info.run_id, "alice", "ship it")
    assert done.status == Status.COMPLETED
    assert done.outputs["ship"] == {"shipped": "v1", "by": "alice"}
    assert done.outputs["review"]["decision"] == "approved"
    types = [e.type for e in eng2.history(info.run_id)]
    assert "APPROVAL_REQUESTED" in types and "APPROVAL_DECIDED" in types


def test_reject_routes_to_revision_loop() -> None:
    eng = Engine(":memory:", [gated()], clock=FakeClock())
    rid = eng.execute("gated", {}).run_id
    r = eng.reject(rid, "alice", "too short")
    assert r.status == Status.WAITING_APPROVAL and r.pending_approval["request"] == {"draft": "v2"}
    assert r.approvals["review"]["comment"] == "too short"
    done = eng.approve(rid, "alice")
    assert done.path == ["write", "review", "write", "review", "ship"]
    assert done.outputs["ship"]["shipped"] == "v2"


def test_reject_without_branch_fails_run() -> None:
    eng = Engine(":memory:", [gated(on_reject=None)], clock=FakeClock())
    rid = eng.execute("gated", {}).run_id
    r = eng.reject(rid, "alice", "no")
    assert r.status == Status.FAILED and "rejected by alice" in r.error


def test_approve_without_resume() -> None:
    eng = Engine(":memory:", [gated()], clock=FakeClock())
    rid = eng.execute("gated", {}).run_id
    r = eng.approve(rid, "alice", resume=False)
    assert r.status == Status.RUNNING and r.current_step == "ship"
    assert eng.run(rid).status == Status.COMPLETED


def test_authorisation_and_state_checks() -> None:
    eng = Engine(":memory:", [gated()], clock=FakeClock())
    rid = eng.execute("gated", {}).run_id
    with pytest.raises(ApprovalError, match="not an authorised approver"):
        eng.approve(rid, "mallory")
    with pytest.raises(ApprovalError, match="waiting at gate"):
        eng.approve(rid, "alice", gate="other")
    eng.approve(rid, "alice")
    with pytest.raises(InvalidStateError):
        eng.approve(rid, "alice")


def test_open_gate_allows_anyone() -> None:
    eng = Engine(":memory:", [gated(approvers=set())], clock=FakeClock())
    rid = eng.execute("gated", {}).run_id
    assert eng.approve(rid, "anybody").status == Status.COMPLETED


def test_escalation_then_timeout_failure() -> None:
    clock = FakeClock()
    eng = Engine(":memory:", [gated(timeout=100, escalate_to={"boss"}, escalation_timeout=50)], clock=clock)
    rid = eng.execute("gated", {}).run_id
    clock.advance(99)
    assert eng.check_timeouts() == []
    clock.advance(2)
    assert eng.check_timeouts() == [rid]
    r = eng.get_run(rid)
    assert r.status == Status.WAITING_APPROVAL and r.pending_approval["escalated"]
    assert set(r.pending_approval["approvers"]) == {"alice", "boss"}
    clock.advance(51)
    assert eng.check_timeouts() == [rid]
    r = eng.get_run(rid)
    assert r.status == Status.FAILED and "timed out" in r.error
    types = [e.type for e in eng.history(rid)]
    assert types.count("APPROVAL_ESCALATED") == 1 and "APPROVAL_TIMED_OUT" in types


def test_escalated_approver_can_decide() -> None:
    clock = FakeClock()
    eng = Engine(":memory:", [gated(timeout=10, escalate_to={"boss"})], clock=clock)
    rid = eng.execute("gated", {}).run_id
    with pytest.raises(ApprovalError):
        eng.approve(rid, "boss")
    clock.advance(11)
    eng.run(rid)  # run() lazily processes the overdue gate
    assert eng.approve(rid, "boss").status == Status.COMPLETED


def test_timeout_routes_to_step() -> None:
    clock = FakeClock()
    eng = Engine(":memory:", [gated(timeout=10, on_timeout="ship")], clock=clock)
    rid = eng.execute("gated", {}).run_id
    clock.advance(10)
    with pytest.raises(ApprovalError, match="expired"):
        eng.approve(rid, "alice")
    r = eng.run(rid)
    assert r.status == Status.COMPLETED and r.outputs["review"]["decision"] == "timeout"
    assert eng.replay(rid)["outputs"] == r.outputs


def test_timeout_to_end_completes() -> None:
    clock = FakeClock()
    wf = Workflow("w")
    wf.step("a", next="g")(lambda ctx: 1)
    wf.approval("g", on_approve=END, timeout=5, on_timeout=None)
    eng = Engine(":memory:", [wf], clock=clock)
    rid = eng.execute("w", {}).run_id
    clock.advance(6)
    assert eng.check_timeouts() == [rid]
    assert eng.get_run(rid).status == Status.COMPLETED


def test_prepare_failure_dead_letters() -> None:
    wf = Workflow("w")
    wf.step("a", next="g")(lambda ctx: 1)
    wf.approval("g", prepare=lambda ctx: ctx.outputs["missing"])
    info = Engine(":memory:", [wf], clock=FakeClock()).execute("w", {})
    assert info.status == Status.DEAD_LETTER and "preparation failed" in info.error


# ------------------------------------------------------------------ signals and cancellation


def test_signals_are_durable_and_visible_to_steps(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    wf = Workflow("sig")
    wf.step("a", next="b")(lambda ctx: None)
    wf.step("b")(lambda ctx: {"prio": ctx.signal("priority", "normal"), "n": len(ctx.signals())})
    eng = Engine(db, [wf], clock=FakeClock())
    rid = eng.start("sig", {})
    eng.run(rid, max_steps=1)
    eng.signal(rid, "priority", "low")
    eng.signal(rid, "priority", "urgent")
    info = Engine(db, [wf], clock=FakeClock()).run(rid)
    assert info.outputs["b"] == {"prio": "urgent", "n": 2}
    assert [e.type for e in eng.history(rid)].count("SIGNAL_RECEIVED") == 2


def test_cancel_idle_runs_immediately() -> None:
    eng = Engine(":memory:", [gated()], clock=FakeClock())
    waiting = eng.execute("gated", {}).run_id
    pending = eng.start("gated", {})
    for rid in (waiting, pending):
        r = eng.cancel(rid, "customer withdrew")
        assert r.status == Status.CANCELLED and r.error == "customer withdrew"
        assert r.pending_approval is None
    with pytest.raises(InvalidStateError):
        eng.cancel(waiting)
    assert eng.run(waiting).status == Status.CANCELLED


def test_cancel_running_run_stops_at_step_boundary() -> None:
    holder: dict = {}
    wf = Workflow("long")

    @wf.step(next="two")
    def one(ctx):
        holder["engine"].cancel(ctx.run_id, "operator abort")  # arrives while the step runs
        assert ctx.cancelled()
        return 1

    executed = []
    wf.step("two")(lambda ctx: executed.append(1))
    eng = Engine(":memory:", [wf], clock=FakeClock())
    holder["engine"] = eng
    r = eng.execute("long", {})
    assert r.status == Status.CANCELLED and executed == []
    assert r.outputs == {"one": 1}  # the in-flight step's result was still checkpointed
    types = [e.type for e in eng.history(r.run_id)]
    assert types.index("CANCEL_REQUESTED") < types.index("RUN_CANCELLED")


def test_cancel_during_retry_backoff() -> None:
    holder: dict = {}
    wf = Workflow("w")

    @wf.step()
    def a(ctx):
        holder["e"].cancel(ctx.run_id)
        raise ConnectionError("down")

    eng = Engine(":memory:", [wf], clock=FakeClock())
    holder["e"] = eng
    r = eng.execute("w", {})
    assert r.status == Status.CANCELLED and eng.stats["retries"] == 0
