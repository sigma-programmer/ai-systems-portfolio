# P05: Durable Agent State Machine

> You build agents that survive production failures.

A durable execution engine for long-running agent workflows. It checkpoints to
SQLite after every step, replays deterministically on resume, keeps side effects
idempotent, retries with exponential backoff and full jitter, dead-letters failed
runs, pauses at human approval gates (with escalation), supports signals and
cancellation, and recovers from crashes. Crash recovery is proven by actually
`SIGKILL`ing and `os._exit`ing worker processes and resuming in a fresh process.
Only the standard library is used (plus `aikit` for the sample agent's LLM).

## Problem

An agent that plans, calls tools, waits a day for a human and then publishes will
eventually be killed mid-flight: a deploy, an OOM, a spot instance, a laptop lid.
Restarting from scratch costs money (LLM calls again), loses work (the human's
approval) and, worst of all, repeats side effects (a second charge, a second
email). The engine has to make "kill -9 at any instruction" a non-event.

## Architecture

```
                    ┌───────────────────────── Engine (any process) ─────────────────────────┐
  start(wf, input)  │                                                                        │
  run(run_id) ────► │  acquire lease ─► replay event log == snapshot? ─► loop:               │
  recover()         │     ┌─────────────────────────────────────────────────────────────┐    │
                    │     │ cancel requested? ──► CANCELLED                             │    │
                    │     │ node = workflow[current_step]                               │    │
                    │     │  ├─ ApprovalGate ─► APPROVAL_REQUESTED, WAITING_APPROVAL ───┼──► │ return
                    │     │  └─ Step:                                                   │    │
                    │     │      sleep(next_attempt_at - now)   (durable backoff)       │    │
                    │     │      TXN{ STEP_STARTED #n, lease renew }   ◄── crash point  │    │
                    │     │      handler(ctx) [worker thread if timeout]                │    │
                    │     │        └─ ctx.effect(name, fn) ─ idempotency key            │    │
                    │     │             run/step/visit/name ─► fn ─► TXN{record}        │    │
                    │     │      ok:   TXN{ STEP_COMPLETED(output,next) + snapshot }    │    │
                    │     │      fail: retryable & budget left ─► TXN{STEP_FAILED,      │    │
                    │     │              next_attempt_at = now + U(0, cap)}             │    │
                    │     │            else ─► DEAD_LETTERED                            │    │
                    │     └─────────────────────────────────────────────────────────────┘    │
  approve/reject ─► │  TXN{ APPROVAL_DECIDED, route on_approve | on_reject } ─► run()        │
  check_timeouts ─► │  overdue gate ─► APPROVAL_ESCALATED (once) ─► APPROVAL_TIMED_OUT       │
  signal/cancel ──► │  durable signals table / cancel flag (checked at step boundaries)      │
                    └───────────────────────────────┬────────────────────────────────────────┘
                                                    ▼
          SQLite (WAL, synchronous=FULL):  runs (snapshot + cursor + lease)
                                           events (append-only, UNIQUE(run_id, seq))
                                           idempotency (key → recorded result)
                                           signals
```

| Module | Role |
|---|---|
| `workflow.py` | `Workflow`, `@wf.step(next=..., retry=..., timeout=...)`, `wf.approval(...)`, `branch()`, static validation, graph fingerprint |
| `engine.py` | `Engine` (run / recover / approve / reject / signal / cancel / retry_dead_letter / replay / inspection), `StepContext` (outputs, `effect`, `idempotency_key`, `signal`, `cancelled`, `remaining`) |
| `store.py` | `SQLiteStore`: schema, re-entrant `BEGIN IMMEDIATE` transactions, events, idempotency, signals |
| `retry.py` | `RetryPolicy`: exponential backoff, full jitter seeded per (run, step, attempt), retryable vs fatal classification |
| `clock.py` | `SystemClock`, `FakeClock` (sleep advances virtual time, so tests never sleep) |
| `faults.py` | Crash points, `CrashAt`, `ExitAt` (real `os._exit`), `RandomFaults` |
| `samples.py` | `Ledger` (an external system that dedupes on keys), `order_workflow`, the `research_workflow` agent |
| `worker.py` | `python -m durable_agent.worker`, the process the crash tests kill |
| `bench.py` | Chaos benchmark, checkpoint overhead, real process kills |

Statuses: `PENDING → RUNNING ⇄ WAITING_APPROVAL → COMPLETED | FAILED | CANCELLED | DEAD_LETTER`
(dead-lettered runs can be re-queued with `retry_dead_letter`).

## Key design decisions and tradeoffs

- **Snapshot plus event log, committed together.** Each step completion writes
  the event, the new snapshot and the cursor move in one transaction. Resume reads
  the snapshot (O(1)) *and* replays the log to rebuild the state, refusing to run
  if they disagree, which catches corruption or out-of-band edits. Completed steps
  are never re-executed: later steps see their **recorded** outputs, so LLM calls
  are not repeated or re-sampled into different answers after a crash.
- **At-least-once steps, exactly-once effects.** A step interrupted mid-flight is
  detected (a trailing `STEP_STARTED`), logged as `RECOVERED`, and run again. Side
  effects go through `ctx.effect(name, fn)`, keyed by `run/step/visit/name`. That
  key is stable across retries and crash recovery of the *same visit*, and a
  revision loop gets a fresh key. There are two modes:
  - *External* (`fn(key)`): the result is recorded after the call. If the process
    dies between the call and the record, the call is re-sent **with the same
    key**, and the downstream system deduplicates, as Stripe- or SES-style
    idempotency keys do. The benchmark counts these re-sends
    (`effect_reinvocations_deduped_by_key`).
  - *Transactional outbox* (`fn(conn, key)`, `transactional=True`): for effects in
    the engine's own database, the effect and its record commit atomically, so
    there are no re-sends at all.
- **Durable, reproducible backoff.** Full jitter, `U(0, min(max, base·2^(n-1)))`,
  is seeded from `(seed, run_id, step, attempt)`, so a schedule replays identically
  while different runs stay decorrelated (no thundering herd). `next_attempt_at` is
  persisted, so a restart waits only the *remaining* backoff. Crashes count against
  the attempt budget, so a step that kills the process on every attempt ends in the
  dead letter queue instead of crash-looping.
- **Error classification.** `FatalError` and `ValueError`/`TypeError`/`KeyError`/`PermissionError`
  are fatal. `RetryableError`, `ConnectionError`, `TimeoutError` and `StepTimeout`
  are retryable. Unknown exceptions are fatal by default, because a bug should not
  burn the retry budget. All of this is configurable per step.
- **Timeouts, honestly.** Python threads can't be killed. A timed-out handler keeps
  running in a daemon thread, but its context is marked *abandoned*: any later
  `ctx.effect` raises `EffectAbandoned` and its result is discarded. Handlers can
  also poll `ctx.remaining()` and `ctx.cancelled()`. If you need hard kills,
  run the step in a subprocess.
- **Leases, not locks.** A run is owned by `host:pid:engine-id` with a TTL that is
  renewed every step. A second live engine gets `LeaseHeld`. If the owner's PID is
  gone on the same host, the lease is taken over immediately (`LEASE_TAKEN_OVER`).
  Otherwise the new owner waits for the TTL to expire. Steps longer than the TTL
  should raise it.
- **Approval gates are just states.** A gate commits `WAITING_APPROVAL` and returns.
  Nothing stays in memory, so `approve` and `reject` work from any process, days
  later. Approvers are authorized per gate. On timeout the engine escalates once,
  adding approvers and extending the deadline, then times out (fail, or route to a
  step). Rejection can loop back for a revision, and `max_visits` bounds every loop.
- **Workflow versioning.** Runs store a fingerprint of the graph's shape, and
  resuming against a changed graph raises `WorkflowVersionMismatch` instead of
  silently mis-routing.

## Sample agent: research report

`plan` (LLM, JSON queries; malformed output is retried) → `search` (a flaky tool
that fails transiently, with backoff) → `draft` (LLM, using reviewer feedback when
revising) → `review` (approval gate: `editor`; escalates to `managing_editor` after
24 h; a rejection loops back to `draft`) → `publish` (an external CMS effect,
exactly once) → `notify` (transactional outbox, chosen by a `branch`). It uses the
`aikit` LLM protocol: `MockLLM` offline, real Claude via `get_llm()` when
credentials are present.

## How to run

```bash
.venv/bin/python -m pytest projects/p05_durable_agent            # 82 tests, ~2.5 s
AIKIT_OFFLINE=1 .venv/bin/python projects/p05_durable_agent/demo.py
cd projects/p05_durable_agent && PYTHONPATH=.:../../common ../../.venv/bin/python -m durable_agent.bench
```

The tests cover:

- simulated crashes at all 4 crash points × 3 steps;
- `os._exit` of a real worker at every crash point, then resume in a fresh process;
- `SIGKILL` of a worker blocked right after a side effect;
- 25 runs under random crashes;
- retries and backoff timing on the fake clock, durable backoff across a restart,
  timeouts and abandoned effects, the dead letter queue and manual retry,
  branches, loops and `max_visits`;
- replay divergence detection and version mismatch;
- approvals from another engine, rejection loops, authorization, escalation and
  timeout routing;
- signals, and cancelling idle, running and backing-off runs;
- the research agent end to end.

## Benchmark (measured on an Apple Silicon Mac, Python 3.14; about 2 s total)

**Chaos run:** 200 order workflows with 3 side-effecting steps each, a 4% crash
chance at each of the 4 crash points per attempt, and a 15% transient failure
chance per step attempt (seed 0):

| Metric | Value |
|---|---|
| Completion rate | **1.0** (200/200, 0 dead-lettered) |
| Crashes injected / transient failures injected | 112 / 122 |
| Interrupted in-flight attempts recovered | 90 (the other 22 crashes hit just after a commit) |
| Side effects applied / expected | 600 / 600 |
| **Duplicate side effects** | **0** |
| Effect re-sends deduplicated by idempotency key | 24 (crash between the external call and its record) |
| Mean retries per workflow | 0.61 |
| Recovery time: fresh engine resumes and finishes the interrupted run | mean 1.8–2.3 ms, p95 3.5–4.5 ms |
| Mean checkpoint transaction | ~0.2 ms |

**Checkpoint overhead per step** (500 no-op steps, two transactions per step):
in-memory SQLite 0.08 ms/step; file in WAL mode with `synchronous=NORMAL`
0.20–0.23 ms/step; with `synchronous=FULL` 0.27–0.30 ms/step. So the cost of
durability is **about 0.2 ms per step**, which is negligible next to any LLM or
tool call. Note that on macOS SQLite's `fsync` does not issue `F_FULLFSYNC`
unless `PRAGMA fullfsync` is set, so "FULL" here is durable against a process
crash, not necessarily against power loss.

**Real process kills:** 3 workers `SIGKILL`ed right after a side effect (one each
at reserve, charge and ship). All 3 completed after resume in a new process, with
9/9 effects and 0 duplicates. Resume takes about 110 ms per process, which is
mostly interpreter startup.
