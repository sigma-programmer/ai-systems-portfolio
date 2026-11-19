"""Benchmark: chaos run of many workflows with injected crashes and transient
failures, plus checkpoint overhead and real process-kill recovery.

    cd projects/p05_durable_agent && ../../.venv/bin/python -m durable_agent.bench

All chaos metrics are deterministic for a given seed (a FakeClock means backoff
never really sleeps); timings are wall-clock and machine-dependent.
"""
from __future__ import annotations

import json
import os
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .clock import FakeClock
from .engine import Engine, Status, StepContext
from .faults import RandomFaults, SimulatedCrash
from .retry import RetryPolicy
from .samples import Ledger, order_workflow
from .store import SQLiteStore
from .workflow import END, Workflow

__all__ = ["benchmark"]


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def chaos(n_workflows: int = 200, crash_rate: float = 0.04, transient_rate: float = 0.15, seed: int = 0) -> dict[str, Any]:
    """Run ``n_workflows`` order pipelines (3 side-effecting steps each) through
    random crashes at every crash point and random transient step failures."""
    with tempfile.TemporaryDirectory() as tmp:
        db, ledger = Path(tmp) / "engine.db", Ledger(Path(tmp) / "ledger.db")
        clock = FakeClock()
        transient = {"n": 0}

        def flaky(name: str, ctx: StepContext) -> None:
            rng = random.Random(f"{seed}:{ctx.run_id}:{name}:{ctx.attempt}")
            if rng.random() < transient_rate:
                transient["n"] += 1
                raise ConnectionError(f"transient failure in {name}")

        wf = order_workflow(ledger, flaky=flaky, retry=RetryPolicy(max_attempts=8, base_delay=0.2, max_delay=5.0, seed=seed))
        faults = RandomFaults(crash_rate=crash_rate, seed=seed)
        setup = Engine(db, [wf], clock=clock)
        ids = [setup.start("order", {"order": i}, run_id=f"run-{i:04d}") for i in range(n_workflows)]
        setup.store.close()

        recovery_ms: list[float] = []
        txn_seconds = 0.0
        txn_count = 0
        rounds = 0
        t_start = time.perf_counter()
        pending_recovery: str | None = None
        while rounds < 50 * n_workflows:
            rounds += 1
            eng = Engine(db, [wf], clock=clock, faults=faults)
            try:
                if pending_recovery is not None:
                    t0 = time.perf_counter()
                    eng.run(pending_recovery)  # fresh engine resumes the interrupted run
                    recovery_ms.append((time.perf_counter() - t0) * 1000)
                    pending_recovery = None
                eng.recover()
                txn_seconds += eng.store.txn_seconds
                txn_count += eng.store.txn_count
                break
            except SimulatedCrash:
                txn_seconds += eng.store.txn_seconds
                txn_count += eng.store.txn_count
                eng.store.close()  # the "process" is gone; nothing was cleaned up
                pending_recovery = faults.last_crash[0] if faults.last_crash else None
                clock.advance(31)  # wait out the dead owner's lease
        wall = time.perf_counter() - t_start

        final = Engine(db, [wf], clock=clock)
        runs = [final.get_run(r) for r in ids]
        completed = sum(r.status == Status.COMPLETED for r in runs)
        retries = sum(
            1 for r in ids for e in final.store.events(r) if e.type == "STEP_FAILED" and e.payload.get("retryable")
        )
        recovered = sum(1 for r in ids for e in final.store.events(r) if e.type == "RECOVERED")
        unique_effects = len(ledger.effects())
        return {
            "workflows": n_workflows,
            "steps_per_workflow": 3,
            "completion_rate": round(completed / n_workflows, 4),
            "dead_lettered": sum(r.status == Status.DEAD_LETTER for r in runs),
            "crashes_injected": faults.crashes,
            "transient_failures_injected": transient["n"],
            "interrupted_attempts_recovered": recovered,
            "side_effects_expected": 3 * completed,
            "side_effects_applied": unique_effects,
            "duplicate_side_effects": ledger.duplicates(),
            "effect_reinvocations_deduped_by_key": ledger.calls() - unique_effects,
            "mean_retries_per_workflow": round(retries / n_workflows, 3),
            "recovery_ms_mean": round(statistics.fmean(recovery_ms), 3) if recovery_ms else 0.0,
            "recovery_ms_p95": round(_pct(recovery_ms, 0.95), 3),
            "checkpoint_txn_ms_mean": round(txn_seconds / max(txn_count, 1) * 1000, 4),
            "chaos_wall_s": round(wall, 3),
        }


def checkpoint_overhead(n_runs: int = 100, n_steps: int = 5) -> dict[str, Any]:
    """Per-step cost of durability: SQLite file (WAL, FULL / NORMAL) vs in-memory."""
    wf = Workflow("noop")
    for i in range(n_steps):
        wf.step(f"s{i}", next=f"s{i + 1}" if i + 1 < n_steps else END)(lambda ctx: {"v": ctx.visit})
    out: dict[str, Any] = {"steps_measured": n_runs * n_steps}
    baseline_ms = None
    for label, sync, mem in (("memory", "FULL", True), ("file_sync_normal", "NORMAL", False), ("file_sync_full", "FULL", False)):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(":memory:" if mem else Path(tmp) / "e.db", synchronous=sync)
            eng = Engine(store, [wf], clock=FakeClock(), verify_replay=False)
            for _ in range(10):  # warm-up (page cache, WAL file creation)
                eng.execute("noop", {})
            t0 = time.perf_counter()
            for _ in range(n_runs):
                eng.execute("noop", {})
            per_step = (time.perf_counter() - t0) * 1000 / (n_runs * n_steps)
            out[f"{label}_ms_per_step"] = round(per_step, 4)
            if mem:
                baseline_ms = per_step
            store.close()
    out["checkpoint_overhead_ms_per_step_full"] = round(out["file_sync_full_ms_per_step"] - (baseline_ms or 0), 4)
    out["checkpoint_overhead_ms_per_step_normal"] = round(out["file_sync_normal_ms_per_step"] - (baseline_ms or 0), 4)
    return out


def process_kills(n: int = 3) -> dict[str, Any]:
    """SIGKILL a real worker process right after a side effect, then resume in a new process."""
    project = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": f"{project}{os.pathsep}{project.parents[1] / 'common'}", "AIKIT_OFFLINE": "1"}
    resume_ms: list[float] = []
    completed = dupes = 0
    with tempfile.TemporaryDirectory() as tmp:
        base = [sys.executable, "-m", "durable_agent.worker", "--db", f"{tmp}/e.db", "--ledger", f"{tmp}/l.db"]
        for i in range(n):
            rid = f"kill-{i}"
            step = ("reserve", "charge", "ship")[i % 3]
            proc = subprocess.Popen(base + ["--run-id", rid, "--start", "--hang-after", step], env=env, cwd=project,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            for line in proc.stdout:  # type: ignore[union-attr]
                if line.strip() == f"EFFECT_DONE {step}":
                    break
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait()
            t0 = time.perf_counter()
            res = subprocess.run(base + ["--run-id", rid], env=env, cwd=project, capture_output=True, text=True, timeout=60)
            resume_ms.append((time.perf_counter() - t0) * 1000)
            completed += '"COMPLETED"' in res.stdout
        ledger = Ledger(f"{tmp}/l.db")
        dupes = ledger.duplicates()
        applied = len(ledger.effects())
    return {
        "kills": n,
        "completed_after_resume": completed,
        "side_effects_applied": applied,
        "duplicate_side_effects": dupes,
        "resume_process_ms_mean": round(statistics.fmean(resume_ms), 1),
    }


def benchmark() -> dict[str, Any]:
    """Run all benchmarks; returns JSON-serializable metrics."""
    return {
        "chaos": chaos(),
        "checkpoint": checkpoint_overhead(),
        "process_kills": process_kills(),
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
