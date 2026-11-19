"""Crash recovery: simulated crashes at every crash point, real process kills,
and leases. The invariant everywhere: each side effect happens exactly once."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from durable_agent import (
    CRASH_POINTS,
    CrashAt,
    Engine,
    FakeClock,
    LeaseHeld,
    RandomFaults,
    RetryPolicy,
    SimulatedCrash,
    Status,
)
from durable_agent.samples import ORDER_EFFECTS, Ledger, order_workflow

PROJECT = Path(__file__).resolve().parents[1]
COMMON = PROJECT.parents[1] / "common"


def assert_exactly_once(ledger: Ledger, run_id: str) -> None:
    assert sorted(n for _, n in ledger.effects(run_id)) == sorted(ORDER_EFFECTS)
    assert ledger.duplicates() == 0


@pytest.mark.parametrize("point", CRASH_POINTS)
@pytest.mark.parametrize("step", ORDER_EFFECTS)
def test_simulated_crash_at_every_point_then_resume(tmp_path: Path, point: str, step: str) -> None:
    db, ledger = tmp_path / "e.db", Ledger(tmp_path / "ledger.db")
    clock = FakeClock()
    crashing = Engine(db, [order_workflow(ledger)], clock=clock, faults=CrashAt(point, step), lease_ttl=30)
    rid = crashing.start("order", {"sku": "X"})
    with pytest.raises(SimulatedCrash):
        crashing.run(rid)
    finished_before_crash = point == "after_commit" and step == "ship"
    if not finished_before_crash:
        # the crashed engine never cleaned up: its lease is still held
        with pytest.raises(LeaseHeld):
            Engine(db, [order_workflow(ledger)], clock=clock).run(rid)
    clock.advance(31)  # lease expires
    fresh = Engine(db, [order_workflow(ledger)], clock=clock)
    info = fresh.run(rid)
    assert info.status == Status.COMPLETED
    assert_exactly_once(ledger, rid)
    types = [e.type for e in fresh.history(rid)]
    assert ("LEASE_TAKEN_OVER" in types) is not finished_before_crash
    if point != "after_commit":
        assert "RECOVERED" in types  # the in-flight attempt was detected
    if point == "after_effect":
        assert ledger.calls() == 4  # effect re-invoked once with the same key, deduplicated downstream
    else:
        assert ledger.calls() == 3
    assert fresh.replay(rid)["outputs"] == info.outputs


def test_repeated_crashes_consume_attempt_budget(tmp_path: Path) -> None:
    db, ledger = tmp_path / "e.db", Ledger()
    clock = FakeClock()
    wf = order_workflow(ledger, retry=RetryPolicy(max_attempts=2, base_delay=0))
    faults = CrashAt("before_handler", "charge", times=99)
    for _ in range(2):
        eng = Engine(db, [wf], clock=clock, faults=faults)
        rid = eng.list_runs()[0].run_id if eng.list_runs() else eng.start("order", {})
        with pytest.raises(SimulatedCrash):
            eng.run(rid)
        clock.advance(60)
    info = Engine(db, [wf], clock=clock, faults=faults).run(rid)
    assert info.status == Status.DEAD_LETTER and "attempt budget exhausted" in info.error  # poison step quarantined


def test_random_faults_many_runs_exactly_once(tmp_path: Path) -> None:
    db, ledger = tmp_path / "e.db", Ledger(tmp_path / "l.db")
    clock = FakeClock()
    faults = RandomFaults(crash_rate=0.15, seed=3)
    wf = order_workflow(ledger, retry=RetryPolicy(max_attempts=10, base_delay=0.01))
    ids = [Engine(db, [wf], clock=clock).start("order", {"i": i}) for i in range(25)]
    for _ in range(500):
        eng = Engine(db, [wf], clock=clock, faults=faults)
        try:
            eng.recover()
        except SimulatedCrash:
            clock.advance(31)
            continue
        if all(eng.get_run(r).status == Status.COMPLETED for r in ids):
            break
    assert faults.crashes > 5
    assert {eng.get_run(r).status for r in ids} == {Status.COMPLETED}
    assert all(eng.get_run(r).status == Status.COMPLETED for r in ids)
    assert len(ledger.effects()) == 25 * 3 and ledger.duplicates() == 0


def test_live_lease_blocks_second_engine(tmp_path: Path) -> None:
    db, ledger = tmp_path / "e.db", Ledger()
    clock = FakeClock()
    holder: dict = {}
    wf = order_workflow(ledger, flaky=lambda name, ctx: holder.setdefault("err", _try_run(holder, ctx)))
    e1 = Engine(db, [wf], clock=clock)
    holder["other"] = Engine(db, [wf], clock=clock)
    rid = e1.execute("order", {}).run_id
    assert isinstance(holder["err"], LeaseHeld)
    assert e1.get_run(rid).status == Status.COMPLETED
    assert e1.get_run(rid).run_id == rid


def _try_run(holder: dict, ctx) -> BaseException | None:
    try:
        holder["other"].run(ctx.run_id)
    except LeaseHeld as exc:
        return exc
    return None


# ------------------------------------------------------------------ real processes


def _worker(tmp: Path, run_id: str, *extra: str, **kw) -> subprocess.CompletedProcess | subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": f"{PROJECT}{os.pathsep}{COMMON}", "AIKIT_OFFLINE": "1"}
    cmd = [sys.executable, "-m", "durable_agent.worker", "--db", str(tmp / "e.db"), "--ledger", str(tmp / "l.db"), "--run-id", run_id, *extra]
    if kw.get("popen"):
        return subprocess.Popen(cmd, env=env, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, env=env, cwd=PROJECT, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("point", ["before_handler", "after_effect", "after_handler", "after_commit"])
def test_process_exit_mid_workflow_then_resume_in_fresh_process(tmp_path: Path, point: str) -> None:
    crashed = _worker(tmp_path, "r1", "--start", "--crash-at", point, "--crash-step", "charge")
    assert crashed.returncode == 137, crashed.stderr  # really died, no status line
    assert crashed.stdout == ""
    ledger = Ledger(tmp_path / "l.db")
    done_before = {n for _, n in ledger.effects("r1")}
    assert "reserve" in done_before and "ship" not in done_before
    resumed = _worker(tmp_path, "r1")  # dead owner pid -> lease taken over immediately
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout.strip().splitlines()[-1]) == {"status": "COMPLETED", "path": ["reserve", "charge", "ship"]}
    assert_exactly_once(ledger, "r1")
    eng = Engine(tmp_path / "e.db", [order_workflow(ledger)])
    hist = [e.type for e in eng.history("r1")]
    assert hist.count("STEP_COMPLETED") == 3 and hist[-1] == "RUN_COMPLETED"


def test_sigkill_after_side_effect_then_resume(tmp_path: Path) -> None:
    proc = _worker(tmp_path, "r2", "--start", "--hang-after", "charge", popen=True)
    try:
        lines = []
        for line in proc.stdout:
            lines.append(line.strip())
            if line.strip() == "EFFECT_DONE charge":
                break
        assert lines == ["EFFECT_DONE reserve", "EFFECT_DONE charge"]
        os.kill(proc.pid, signal.SIGKILL)
    finally:
        proc.wait(timeout=30)
    assert proc.returncode == -signal.SIGKILL
    ledger = Ledger(tmp_path / "l.db")
    assert sorted(n for _, n in ledger.effects("r2")) == ["charge", "reserve"]
    resumed = _worker(tmp_path, "r2")
    assert resumed.returncode == 0, resumed.stderr
    assert '"COMPLETED"' in resumed.stdout
    assert_exactly_once(ledger, "r2")
    eng = Engine(tmp_path / "e.db", [order_workflow(ledger)])
    types = [e.type for e in eng.history("r2")]
    assert "RECOVERED" in types and "LEASE_TAKEN_OVER" in types
