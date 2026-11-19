"""End-to-end demo of the durable agent engine (fully offline).

    AIKIT_OFFLINE=1 .venv/bin/python projects/p05_durable_agent/demo.py

1. A research-report agent (plan → search → draft → human review → publish)
   runs with a flaky search tool, suspends for approval, gets a rejection and
   a revision, and is approved from a *different engine instance*.
2. A worker process is SIGKILLed right after charging a customer; a fresh
   process resumes the run, and every side effect happens exactly once.
3. Chaos benchmark summary.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "common"))

from durable_agent import Engine, FakeClock  # noqa: E402
from durable_agent.bench import benchmark  # noqa: E402
from durable_agent.samples import FlakySearch, Ledger, default_llm, order_workflow, research_workflow  # noqa: E402


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def show_history(engine: Engine, run_id: str) -> None:
    for e in engine.history(run_id):
        extra = ""
        if e.type == "STEP_FAILED":
            extra = f"  error={e.payload['error']!r} retry_in={e.payload.get('retry_in')}"
        elif e.type == "APPROVAL_DECIDED":
            rec = e.payload["decision_record"]
            extra = f"  {rec['decision']} by {rec['approver']}: {rec['comment']!r}"
        elif e.type == "EFFECT_RECORDED":
            extra = f"  key={e.payload['key']}"
        print(f"  {e.seq:>3} {e.type:<20} {e.step or '':<8} {('#' + str(e.attempt)) if e.attempt else '':<3}{extra}")


def research_demo(tmp: Path) -> None:
    section("1. Research agent: retries, approval gate, revision loop, cross-instance approval")
    cms = Ledger(tmp / "cms.db")
    clock = FakeClock()
    search = FlakySearch(fail_rate=0.35, seed=2)
    mk = lambda: Engine(tmp / "agent.db", [research_workflow(cms)], clock=clock, llm=default_llm(), services={"search": search})  # noqa: E731
    agent = mk()
    print(agent.workflows["research_report"].describe())
    info = agent.execute("research_report", {"topic": "durable AI agents", "notify": True})
    print(f"\n  status={info.status} path={info.path}")
    print(f"  search tool: {search.calls} calls, {search.failures} transient failures (backoff slept {sum(clock.sleeps):.2f}s on the fake clock)")
    print(f"  review request -> {info.pending_approval['request']['title']!r}, approvers={info.pending_approval['approvers']}")

    reviewer = mk()  # e.g. a web handler in another process
    r = reviewer.reject(info.run_id, "editor", "please add a revision note")
    print(f"  editor rejected -> status={r.status}, draft revision {r.outputs['draft']['revision']}")
    try:
        reviewer.approve(info.run_id, "intern")
    except Exception as exc:  # noqa: BLE001
        print(f"  intern tries to approve -> {type(exc).__name__}: {exc}")
    done = reviewer.approve(info.run_id, "editor", "ship it")
    print(f"  editor approved -> status={done.status} path={done.path}")
    print(f"  CMS publish count for run: {cms.count('publish', info.run_id)}; replay == snapshot: {reviewer.replay(info.run_id)['outputs'] == done.outputs}")
    print("\n  event log:")
    show_history(reviewer, info.run_id)


def crash_demo(tmp: Path) -> None:
    section("2. Crash recovery: SIGKILL a worker right after it charged the card")
    env = {**os.environ, "PYTHONPATH": f"{HERE}{os.pathsep}{HERE.parents[1] / 'common'}", "AIKIT_OFFLINE": "1"}
    base = [sys.executable, "-m", "durable_agent.worker", "--db", str(tmp / "orders.db"), "--ledger", str(tmp / "payments.db"), "--run-id", "order-1001"]
    proc = subprocess.Popen(base + ["--start", "--hang-after", "charge"], env=env, cwd=HERE, stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:  # type: ignore[union-attr]
        print(f"  worker[{proc.pid}]: {line.strip()}")
        if line.strip() == "EFFECT_DONE charge":
            break
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    print(f"  worker killed with SIGKILL (returncode {proc.returncode}) before checkpointing 'charge'")
    ledger = Ledger(tmp / "payments.db")
    print(f"  external effects so far: {[n for _, n in ledger.effects('order-1001')]}")
    res = subprocess.run(base, env=env, cwd=HERE, capture_output=True, text=True, timeout=60)
    print(f"  fresh worker: {res.stdout.strip()}")
    print(f"  external effects now:   {[n for _, n in ledger.effects('order-1001')]}  duplicates={ledger.duplicates()}")
    print(
        f"  invocations of the external API: {ledger.calls()}. The charge step re-ran, but its effect result was already "
        "in the idempotency table, so it was replayed instead of re-sent. Had the kill landed before that record "
        "committed, the re-sent call would have carried the same key and been deduplicated downstream."
    )
    eng = Engine(tmp / "orders.db", [order_workflow(ledger)])
    show_history(eng, "order-1001")


def main() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        research_demo(tmp)
        crash_demo(tmp)
    section("3. Benchmark")
    print(json.dumps(benchmark(), indent=2))


if __name__ == "__main__":
    main()
