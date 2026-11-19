"""Core engine behaviour: checkpointing, replay, retries, timeouts, dead letters."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from durable_agent import (
    END,
    EffectAbandoned,
    Engine,
    FakeClock,
    FatalError,
    InvalidStateError,
    RetryableError,
    RetryPolicy,
    RunNotFound,
    SQLiteStore,
    Status,
    Workflow,
    WorkflowDefinitionError,
    WorkflowVersionMismatch,
    branch,
)


def linear(calls: list[str] | None = None) -> Workflow:
    calls = calls if calls is not None else []
    wf = Workflow("linear")

    @wf.step(next="b")
    def a(ctx):
        calls.append("a")
        return {"x": ctx.input["n"] * 2}

    @wf.step(next="c")
    def b(ctx):
        calls.append("b")
        return {"y": ctx.outputs["a"]["x"] + 1}

    @wf.step(next=END)
    def c(ctx):
        calls.append("c")
        return {"z": ctx.outputs["b"]["y"] * 10}

    return wf


def test_linear_run_completes_with_outputs_and_history() -> None:
    eng = Engine(":memory:", [linear()], clock=FakeClock())
    info = eng.execute("linear", {"n": 2})
    assert info.status == Status.COMPLETED and info.done
    assert info.outputs == {"a": {"x": 4}, "b": {"y": 5}, "c": {"z": 50}}
    assert info.path == ["a", "b", "c"]
    types = [e.type for e in eng.history(info.run_id)]
    assert types == ["RUN_STARTED"] + ["STEP_STARTED", "STEP_COMPLETED"] * 3 + ["RUN_COMPLETED"]
    assert [e.seq for e in eng.history(info.run_id)] == list(range(1, 9))


def test_checkpoint_after_every_step_and_resume_skips_completed(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    calls: list[str] = []
    eng = Engine(db, [linear(calls)], clock=FakeClock())
    rid = eng.start("linear", {"n": 1})
    info = eng.run(rid, max_steps=2)
    assert info.status == Status.RUNNING and info.current_step == "c"
    assert calls == ["a", "b"]
    # a different engine instance (fresh connection) continues; a and b are replayed, not re-run
    calls2: list[str] = []
    eng2 = Engine(db, [linear(calls2)], clock=FakeClock())
    assert eng2.get_run(rid).outputs == {"a": {"x": 2}, "b": {"y": 3}}
    done = eng2.run(rid)
    assert calls2 == ["c"]
    assert done.outputs["c"] == {"z": 30}


def test_replay_rebuilds_snapshot_and_detects_divergence(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    eng = Engine(db, [linear()], clock=FakeClock())
    rid = eng.start("linear", {"n": 3})
    eng.run(rid, max_steps=2)
    rebuilt = eng.replay(rid)
    assert rebuilt["outputs"] == eng.get_run(rid).outputs
    # corrupt the snapshot behind the engine's back
    with eng.store.transaction() as conn:
        state = json.loads(eng.store.get_run(rid, conn)["state_json"])
        state["outputs"]["a"] = {"x": 999}
        conn.execute("UPDATE runs SET state_json=? WHERE run_id=?", (json.dumps(state), rid))
    with pytest.raises(InvalidStateError, match="diverges"):
        Engine(db, [linear()], clock=FakeClock()).run(rid)


def test_event_log_is_append_only_and_atomic(tmp_path: Path) -> None:
    eng = Engine(tmp_path / "e.db", [linear()], clock=FakeClock())
    info = eng.execute("linear", {"n": 1})
    rows = eng.store.query("SELECT COUNT(*) AS n, COUNT(DISTINCT seq) AS d FROM events WHERE run_id=?", (info.run_id,))
    assert rows[0]["n"] == rows[0]["d"] == 8
    with pytest.raises(Exception):
        with eng.store.transaction() as conn:
            eng.store.append_event(conn, info.run_id, 0.0, "X")
            raise RuntimeError("abort")
    assert len(eng.history(info.run_id)) == 8  # rolled back


def test_idempotent_start_and_unknowns() -> None:
    eng = Engine(":memory:", [linear()], clock=FakeClock())
    rid = eng.start("linear", {"n": 1}, run_id="order-42")
    assert eng.start("linear", {"n": 1}, run_id="order-42") == "order-42"
    assert len(eng.list_runs()) == 1
    with pytest.raises(WorkflowDefinitionError):
        eng.start("nope", {})
    with pytest.raises(RunNotFound):
        eng.run("missing")
    with pytest.raises(RunNotFound):
        eng.history("missing")
    assert rid == "order-42"


def test_workflow_version_mismatch(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    eng = Engine(db, [linear()], clock=FakeClock())
    rid = eng.start("linear", {"n": 1})
    changed = Workflow("linear")
    changed.step("a", next=END)(lambda ctx: 1)
    with pytest.raises(WorkflowVersionMismatch):
        Engine(db, [changed], clock=FakeClock()).run(rid)


# ------------------------------------------------------------------ retries


def flaky_workflow(fail_times: int, exc: type[BaseException] = ConnectionError, **policy) -> tuple[Workflow, dict]:
    state = {"calls": 0}
    wf = Workflow("flaky")

    @wf.step(retry=RetryPolicy(**{"max_attempts": 4, "base_delay": 1.0, "max_delay": 60.0, **policy}))
    def work(ctx):
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise exc("transient")
        return {"attempt": ctx.attempt}

    return wf, state


def test_retry_with_backoff_uses_injected_sleeper() -> None:
    wf, st = flaky_workflow(2)
    clock = FakeClock()
    eng = Engine(":memory:", [wf], clock=clock)
    t0 = time.perf_counter()
    info = eng.execute("flaky", {})
    assert time.perf_counter() - t0 < 1.0  # never really slept
    assert info.status == Status.COMPLETED and info.outputs["work"] == {"attempt": 3}
    assert st["calls"] == 3 and eng.stats["retries"] == 2
    assert len(clock.sleeps) == 2
    fails = [e for e in eng.history(info.run_id) if e.type == "STEP_FAILED"]
    assert clock.sleeps == pytest.approx([f.payload["retry_in"] for f in fails], abs=1e-5)
    assert 0 <= clock.sleeps[0] <= 1.0 and 0 <= clock.sleeps[1] <= 2.0


def test_retries_exhausted_dead_letters_then_manual_retry() -> None:
    wf, st = flaky_workflow(10)
    eng = Engine(":memory:", [wf], clock=FakeClock())
    info = eng.execute("flaky", {})
    assert info.status == Status.DEAD_LETTER
    assert "retries exhausted after 4 attempts" in info.error
    assert st["calls"] == 4
    assert [r.run_id for r in eng.dead_letters()] == [info.run_id]
    st["calls"] = 100  # the outage is over
    again = eng.retry_dead_letter(info.run_id)
    assert again.status == Status.COMPLETED
    assert "DEAD_LETTER_RETRIED" in [e.type for e in eng.history(info.run_id)]
    with pytest.raises(InvalidStateError):
        eng.retry_dead_letter(info.run_id)


def test_fatal_error_not_retried() -> None:
    wf, st = flaky_workflow(1, exc=ValueError)
    eng = Engine(":memory:", [wf], clock=FakeClock())
    info = eng.execute("flaky", {})
    assert info.status == Status.DEAD_LETTER and st["calls"] == 1
    assert "ValueError" in info.error and eng.stats["retries"] == 0


def test_explicit_error_classes() -> None:
    wf, st = flaky_workflow(1, exc=RetryableError)
    assert Engine(":memory:", [wf], clock=FakeClock()).execute("flaky", {}).status == Status.COMPLETED
    wf2, _ = flaky_workflow(1, exc=FatalError)
    assert Engine(":memory:", [wf2], clock=FakeClock()).execute("flaky", {}).status == Status.DEAD_LETTER


def test_backoff_is_durable_across_restart(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    wf, _ = flaky_workflow(1, base_delay=100.0, max_delay=100.0, jitter=False)
    clock = FakeClock()
    eng = Engine(db, [wf], clock=clock)
    rid = eng.start("flaky", {})
    eng.run(rid, max_steps=1)  # first attempt fails; retry scheduled 100 s out
    row = eng.store.get_run(rid)
    assert row["next_attempt_at"] == pytest.approx(clock.now() + 100.0)
    clock.advance(40)
    clock2 = clock
    eng2 = Engine(db, [wf], clock=clock2)
    clock.sleeps.clear()
    assert eng2.run(rid).status == Status.COMPLETED
    assert clock.sleeps == [pytest.approx(60.0)]  # only the remaining backoff


# ------------------------------------------------------------------ timeouts


def test_step_timeout_is_retryable_and_late_effects_refused() -> None:
    attempts: list[int] = []
    released = threading.Event()
    refused: list[bool] = []
    wf = Workflow("slow")

    @wf.step(timeout=0.2, retry=RetryPolicy(max_attempts=3, base_delay=0.0))
    def slow(ctx):
        attempts.append(ctx.attempt)
        if ctx.attempt == 1:
            released.wait(5)
            try:
                ctx.effect("late", lambda key: "should not happen")
            except EffectAbandoned:
                refused.append(True)
            return "late result"
        return "fast"

    eng = Engine(":memory:", [wf], clock=FakeClock())
    info = eng.execute("slow", {})
    assert info.status == Status.COMPLETED and info.outputs["slow"] == "fast"
    assert eng.stats["timeouts"] == 1
    failed = [e for e in eng.history(info.run_id) if e.type == "STEP_FAILED"]
    assert "StepTimeout" in failed[0].payload["error"]
    released.set()
    for _ in range(100):
        if refused:
            break
        time.sleep(0.02)
    assert refused == [True]
    assert eng.store.idem_get(f"{info.run_id}/slow/1/late") == (False, None)


def test_timeout_exhaustion_dead_letters() -> None:
    wf = Workflow("hang")
    wf.step("hang", timeout=0.05, retry=RetryPolicy(max_attempts=2, base_delay=0))(lambda ctx: time.sleep(1))
    info = Engine(":memory:", [wf], clock=FakeClock()).execute("hang", {})
    assert info.status == Status.DEAD_LETTER and "StepTimeout" in info.error


def test_remaining_time_visible_to_handler() -> None:
    wf = Workflow("t")
    seen = {}

    @wf.step(timeout=5)
    def a(ctx):
        seen["r"] = ctx.remaining()
        return None

    wf.step("b")(lambda ctx: None)
    Engine(":memory:", [wf], clock=FakeClock()).execute("t", {})
    assert 0 < seen["r"] <= 5


# ------------------------------------------------------------------ transitions


def test_conditional_branches() -> None:
    wf = Workflow("route")
    wf.step("classify", next=branch(lambda out, ctx: out["kind"], {"big": "big", "small": "small"}))(
        lambda ctx: {"kind": "big" if ctx.input["n"] > 10 else "small"}
    )
    wf.step("big")(lambda ctx: "B")
    wf.step("small")(lambda ctx: "S")
    eng = Engine(":memory:", [wf], clock=FakeClock())
    assert eng.execute("route", {"n": 50}).path == ["classify", "big"]
    assert eng.execute("route", {"n": 1}).path == ["classify", "small"]


def test_callable_router_and_loop_with_visits() -> None:
    wf = Workflow("loop")

    @wf.step(next=lambda out, ctx: "count" if out < 3 else END)
    def count(ctx):
        return ctx.visit

    info = Engine(":memory:", [wf], clock=FakeClock()).execute("loop", {})
    assert info.path == ["count"] * 3 and info.outputs["count"] == 3


def test_router_to_unknown_step_dead_letters() -> None:
    wf = Workflow("bad")
    wf.step("a", next=lambda out, ctx: "ghost")(lambda ctx: 1)
    info = Engine(":memory:", [wf], clock=FakeClock()).execute("bad", {})
    assert info.status == Status.DEAD_LETTER and "ghost" in info.error


def test_max_visits_guards_infinite_loops() -> None:
    wf = Workflow("inf")
    wf.step("spin", next="spin", max_visits=5)(lambda ctx: None)
    info = Engine(":memory:", [wf], clock=FakeClock()).execute("inf", {})
    assert info.status == Status.DEAD_LETTER and "max_visits" in info.error
    assert info.path == ["spin"] * 5


def test_non_serializable_output_is_fatal() -> None:
    wf = Workflow("ser")
    wf.step("a")(lambda ctx: object())
    info = Engine(":memory:", [wf], clock=FakeClock()).execute("ser", {})
    assert info.status == Status.DEAD_LETTER and "TypeError" in info.error


# ------------------------------------------------------------------ effects


def test_effect_recorded_once_and_replayed_on_retry() -> None:
    performed: list[str] = []
    wf = Workflow("eff")
    state = {"n": 0}

    @wf.step(retry=RetryPolicy(max_attempts=3, base_delay=0))
    def pay(ctx):
        r = ctx.effect("charge", lambda key: performed.append(key) or {"charged": key})
        state["n"] += 1
        if state["n"] == 1:
            raise ConnectionError("lost connection after charging")
        return r

    eng = Engine(":memory:", [wf], clock=FakeClock())
    info = eng.execute("eff", {})
    assert info.status == Status.COMPLETED
    assert len(performed) == 1  # second attempt got the recorded result
    assert info.outputs["pay"] == {"charged": f"{info.run_id}/pay/1/charge"}
    assert eng.stats["effects_applied"] == 1 and eng.stats["effects_replayed"] == 1


def test_transactional_effect_rolls_back_with_failure() -> None:
    wf = Workflow("tx")

    @wf.step(retry=RetryPolicy(max_attempts=2, base_delay=0, retry_on=(RuntimeError,)))
    def write(ctx):
        def fn(conn, key):
            conn.execute("CREATE TABLE IF NOT EXISTS t (k TEXT PRIMARY KEY)")
            conn.execute("INSERT INTO t VALUES (?)", (key,))
            if ctx.attempt == 1:
                raise RuntimeError("boom inside the transaction")
            return key

        return ctx.effect("w", fn, transactional=True)

    eng = Engine(":memory:", [wf], clock=FakeClock())
    info = eng.execute("tx", {})
    assert info.status == Status.COMPLETED
    assert eng.store.query("SELECT COUNT(*) AS n FROM t")[0]["n"] == 1


def test_idempotency_key_changes_per_visit() -> None:
    keys: list[str] = []
    wf = Workflow("k")

    @wf.step(next=lambda out, ctx: "s" if ctx.visit < 2 else END)
    def s(ctx):
        keys.append(ctx.idempotency_key("x"))
        return None

    info = Engine(":memory:", [wf], clock=FakeClock()).execute("k", {})
    assert keys == [f"{info.run_id}/s/1/x", f"{info.run_id}/s/2/x"]


def test_inspection_api() -> None:
    eng = Engine(":memory:", [linear()], clock=FakeClock())
    a = eng.execute("linear", {"n": 1})
    b = eng.start("linear", {"n": 2})
    assert {r.run_id for r in eng.list_runs()} == {a.run_id, b}
    assert [r.run_id for r in eng.list_runs(status=Status.PENDING)] == [b]
    assert [r.run_id for r in eng.list_runs(workflow="linear", status=Status.COMPLETED)] == [a.run_id]
    assert eng.get_run(b).status == Status.PENDING


def test_store_rejects_bad_sync_mode() -> None:
    with pytest.raises(ValueError):
        SQLiteStore(":memory:", synchronous="WHENEVER")
