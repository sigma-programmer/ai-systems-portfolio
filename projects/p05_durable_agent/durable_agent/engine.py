"""The durable execution engine.

Execution model
---------------
A run is a cursor (``current_step``) into a :class:`~durable_agent.workflow.Workflow`
plus a JSON state snapshot ``{"input", "outputs", "visits", "approvals", "path"}``.
For each task step the engine:

1. checkpoints ``STEP_STARTED`` (attempt n) in one transaction, which also
   renews the run's lease;
2. runs the handler (in a worker thread when the step has a timeout);
3. on success, commits ``STEP_COMPLETED`` (with the output and the next step),
   the new snapshot and the cursor move **in one transaction**;
4. on failure, classifies the error: a retryable error schedules the next
   attempt (``next_attempt_at`` is persisted, so the backoff survives a crash);
   a fatal error, or an exhausted retry budget, moves the run to
   ``DEAD_LETTER``.

Resume and replay: a fresh engine (in any process) loads the snapshot and
first *replays the event log* to rebuild the state, checking that it equals
the snapshot (to detect corruption). Completed steps are never executed again:
their recorded outputs are what later steps see in ``ctx.outputs``. A step
that was in flight when the process died is detected (a dangling
``STEP_STARTED``), logged as ``RECOVERED``, and executed again. Its side
effects stay exactly-once through idempotency keys (see
:meth:`StepContext.effect`).

Statuses: ``PENDING → RUNNING → (WAITING_APPROVAL ↔ RUNNING) →
COMPLETED | FAILED | CANCELLED | DEAD_LETTER``.
"""
from __future__ import annotations

import copy
import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .clock import Clock, SystemClock
from .errors import (
    ApprovalError,
    EffectAbandoned,
    FatalError,
    InvalidStateError,
    LeaseHeld,
    RunNotFound,
    StepTimeout,
    WorkflowDefinitionError,
    WorkflowVersionMismatch,
)
from .faults import FaultInjector
from .store import TERMINAL, Event, SQLiteStore, dumps
from .workflow import ApprovalGate, Step, Workflow

__all__ = ["Engine", "StepContext", "RunInfo", "Status"]


class Status:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    DEAD_LETTER = "DEAD_LETTER"


# Events that open or close a step attempt; a trailing STEP_STARTED means the attempt was interrupted.
_ATTEMPT_EVENTS = ("STEP_STARTED", "STEP_COMPLETED", "STEP_FAILED", "RECOVERED", "DEAD_LETTER_RETRIED", "APPROVAL_DECIDED", "APPROVAL_TIMED_OUT")


def _empty_state(input_: Any) -> dict[str, Any]:
    return {"input": input_, "outputs": {}, "visits": {}, "approvals": {}, "path": []}


@dataclass
class RunInfo:
    """Read-only view of a run."""

    run_id: str
    workflow: str
    status: str
    current_step: str | None
    input: Any
    outputs: dict[str, Any]
    approvals: dict[str, Any]
    path: list[str]
    attempt: int
    error: str | None
    pending_approval: dict[str, Any] | None
    cancel_requested: bool
    created_at: float
    updated_at: float

    @property
    def done(self) -> bool:
        return self.status in TERMINAL

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "RunInfo":
        state = json.loads(row["state_json"])
        return cls(
            run_id=row["run_id"],
            workflow=row["workflow"],
            status=row["status"],
            current_step=row["current_step"],
            input=state["input"],
            outputs=state["outputs"],
            approvals=state["approvals"],
            path=state["path"],
            attempt=row["attempt"],
            error=row["error"],
            pending_approval=json.loads(row["approval_json"]) if row["approval_json"] else None,
            cancel_requested=bool(row["cancel_requested"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class StepContext:
    """What a step handler sees.

    Attributes:
        run_id, step, attempt, visit: identity of this execution.
        input: the run input.
        outputs: recorded outputs of previously completed steps (and gate
            decisions), replayed from the checkpoint and never recomputed.
        approvals: decisions of approval gates so far.
        llm: the engine's LLM client (any ``aikit.LLM``), if configured.
    """

    def __init__(self, engine: "Engine", run_id: str, step: Step, attempt: int, state: dict[str, Any]) -> None:
        self._engine = engine
        self.run_id = run_id
        self.step = step.name
        self.attempt = attempt
        self.visit = int(state["visits"].get(step.name, 0)) + 1
        self.input = copy.deepcopy(state["input"])
        self.outputs = copy.deepcopy(state["outputs"])
        self.approvals = copy.deepcopy(state["approvals"])
        self.llm = engine.llm
        self.services = engine.services
        self._deadline = (time.monotonic() + step.timeout) if step.timeout else None
        self.abandoned = False

    # ------------------------------------------------------------ time
    def now(self) -> float:
        return self._engine.clock.now()

    def remaining(self) -> float | None:
        """Seconds left before this attempt times out (``None`` = no timeout)."""
        return None if self._deadline is None else max(0.0, self._deadline - time.monotonic())

    # ------------------------------------------------------------ side effects
    def idempotency_key(self, name: str = "effect") -> str:
        """Stable key for a side effect: identical across retries and crash
        recovery of the *same* visit, different for a later visit (a loop)."""
        return f"{self.run_id}/{self.step}/{self.visit}/{name}"

    def effect(self, name: str, fn: Callable[..., Any], *, transactional: bool = False) -> Any:
        """Run a side effect at most once per idempotency key and return its result.

        * Already recorded: the stored result is returned and ``fn`` is not called.
        * ``transactional=False`` (external systems): ``fn(key)`` runs, then its
          result is recorded. A crash between the two re-invokes ``fn`` with the
          **same key**, so the downstream system must deduplicate on it (as
          payment and email APIs do). Combined, the effect happens exactly once.
        * ``transactional=True`` (the transactional-outbox pattern for effects in
          this same database): ``fn(conn, key)`` runs inside the transaction
          that stores the record, so effect and record commit atomically.
        """
        eng = self._engine
        if self.abandoned:
            raise EffectAbandoned(f"step {self.step!r} attempt {self.attempt} timed out; effect {name!r} refused")
        key = self.idempotency_key(name)
        found, result = eng.store.idem_get(key)
        if found:
            eng.stats["effects_replayed"] += 1
            return result
        if transactional:
            with eng.store.transaction() as conn:
                if self.abandoned:
                    raise EffectAbandoned(f"effect {name!r} refused after timeout")
                result = fn(conn, key)
                self._record(conn, key, name, result)
            return result
        result = fn(key)
        eng.faults.hit("after_effect", self.run_id, self.step, self.attempt)
        with eng.store.transaction() as conn:
            self._record(conn, key, name, result)
        return result

    def _record(self, conn: Any, key: str, name: str, result: Any) -> None:
        eng = self._engine
        now = eng.clock.now()
        eng.store.idem_put(conn, key, self.run_id, self.step, result, now)
        eng.store.append_event(conn, self.run_id, now, "EFFECT_RECORDED", self.step, self.attempt, {"name": name, "key": key})
        eng.stats["effects_applied"] += 1

    # ------------------------------------------------------------ signals / cancellation
    def signal(self, name: str, default: Any = None) -> Any:
        """Payload of the latest signal ``name`` sent to this run."""
        sigs = self._engine.store.signals(self.run_id, name)
        return sigs[-1]["payload"] if sigs else default

    def signals(self, name: str | None = None) -> list[dict[str, Any]]:
        return self._engine.store.signals(self.run_id, name)

    def cancelled(self) -> bool:
        """True if cancellation was requested (long steps should poll this)."""
        row = self._engine.store.get_run(self.run_id)
        return bool(row and row["cancel_requested"])


@dataclass
class _Outcome:
    output: Any = None
    error: BaseException | None = None


class Engine:
    """Durable workflow engine over a :class:`SQLiteStore`.

    Args:
        store: a store, or a path to a SQLite file.
        workflows: workflows this engine can execute.
        clock: time source and sleeper (inject :class:`FakeClock` in tests).
        llm: optional LLM client exposed to steps as ``ctx.llm``.
        services: arbitrary dependencies exposed as ``ctx.services``.
        faults: fault injector (crash points) for tests and benchmarks.
        lease_ttl: seconds a run's lease stays valid without renewal.
        verify_replay: on resume, rebuild state from the event log and
            compare it with the snapshot.
    """

    def __init__(
        self,
        store: SQLiteStore | str | Path,
        workflows: Iterable[Workflow] = (),
        clock: Clock | None = None,
        llm: Any = None,
        services: dict[str, Any] | None = None,
        faults: FaultInjector | None = None,
        lease_ttl: float = 30.0,
        verify_replay: bool = True,
    ) -> None:
        self.store = store if isinstance(store, SQLiteStore) else SQLiteStore(store)
        self.clock: Clock = clock or SystemClock()
        self.llm = llm
        self.services = services or {}
        self.faults = faults or FaultInjector()
        self.lease_ttl = lease_ttl
        self.verify_replay = verify_replay
        self.owner_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self.workflows: dict[str, Workflow] = {}
        self.stats: dict[str, Any] = {
            "steps_executed": 0,
            "steps_completed": 0,
            "retries": 0,
            "timeouts": 0,
            "recoveries": 0,
            "effects_applied": 0,
            "effects_replayed": 0,
        }
        for wf in workflows:
            self.register(wf)

    def register(self, wf: Workflow) -> None:
        wf.validate()
        self.workflows[wf.name] = wf

    # ================================================================= starting
    def start(self, workflow: str, input: Any = None, run_id: str | None = None) -> str:
        """Create a run (``PENDING``). Starting an existing ``run_id`` is a no-op,
        so callers can use a business key as the run id for idempotent starts."""
        wf = self._workflow(workflow)
        run_id = run_id or uuid.uuid4().hex
        now = self.clock.now()
        state = _empty_state(input)
        with self.store.transaction() as conn:
            existing = self.store.get_run(run_id, conn)
            if existing is not None:
                if existing["workflow"] != workflow:
                    raise InvalidStateError(f"run {run_id!r} exists for workflow {existing['workflow']!r}")
                return run_id
            self.store.insert_run(
                conn,
                {
                    "run_id": run_id,
                    "workflow": workflow,
                    "fingerprint": wf.fingerprint(),
                    "status": Status.PENDING,
                    "current_step": wf.start,
                    "input_json": dumps(input),
                    "state_json": dumps(state),
                    "created_at": now,
                    "updated_at": now,
                },
            )
            self.store.append_event(conn, run_id, now, "RUN_STARTED", None, None, {"workflow": workflow, "input": input})
        return run_id

    def execute(self, workflow: str, input: Any = None, run_id: str | None = None) -> RunInfo:
        """``start`` + ``run``."""
        return self.run(self.start(workflow, input, run_id))

    # ================================================================= running
    def run(self, run_id: str, max_steps: int | None = None) -> RunInfo:
        """Drive a run until it is terminal, waiting for approval, or ``max_steps``
        steps have executed. Safe to call on any run, in any process."""
        row = self._row(run_id)
        if row["status"] == Status.WAITING_APPROVAL:
            self._check_approval_timeout(run_id)
            row = self._row(run_id)
        if row["status"] in TERMINAL or row["status"] in (Status.WAITING_APPROVAL, Status.DEAD_LETTER):
            return RunInfo.from_row(row)
        wf = self._workflow(row["workflow"])
        if row["fingerprint"] != wf.fingerprint():
            raise WorkflowVersionMismatch(
                f"run {run_id} was started with workflow fingerprint {row['fingerprint']}, current is {wf.fingerprint()}"
            )
        self._acquire(run_id)
        if self.verify_replay:
            self._verify_replay(run_id)
        steps = 0
        while True:
            row = self._row(run_id)
            if row["cancel_requested"] and row["status"] not in TERMINAL:
                self._finish_cancel(run_id, row["cancel_reason"])
                break
            if row["status"] != Status.RUNNING:
                break
            if max_steps is not None and steps >= max_steps:
                self._release(run_id)
                break
            node = wf.node(row["current_step"])
            state = json.loads(row["state_json"])
            if int(state["visits"].get(node.name, 0)) >= node.max_visits:
                self._dead_letter(run_id, node.name, row["attempt"], f"step {node.name!r} exceeded max_visits={node.max_visits}")
                break
            if isinstance(node, ApprovalGate):
                self._enter_gate(run_id, wf, node, state)
                break
            self._run_step(run_id, wf, node, row, state)
            steps += 1
        return self.get_run(run_id)

    def _run_step(self, run_id: str, wf: Workflow, step: Step, row: dict[str, Any], state: dict[str, Any]) -> None:
        # Durable backoff: the next attempt time is persisted, so it is honoured after a crash too.
        wait = row["next_attempt_at"] - self.clock.now()
        if wait > 0:
            self.clock.sleep(wait)
            if self._row(run_id)["cancel_requested"]:
                return
        attempt = row["attempt"] + 1
        policy = step.retry
        now = self.clock.now()
        with self.store.transaction() as conn:
            last = self.store.last_event(run_id, conn, types=_ATTEMPT_EVENTS)
            if last is not None and last.type == "STEP_STARTED":
                # The previous process died mid-step: record it and re-execute.
                self.stats["recoveries"] += 1
                self.store.append_event(conn, run_id, now, "RECOVERED", last.step, last.attempt, {"interrupted_attempt": last.attempt})
            if attempt > policy.max_attempts:
                pass  # handled below, outside the transaction
            else:
                self.store.append_event(conn, run_id, now, "STEP_STARTED", step.name, attempt, {"visit": int(state["visits"].get(step.name, 0)) + 1})
                self.store.update_run(conn, run_id, now, attempt=attempt, lease_expires=now + self.lease_ttl, owner=self.owner_id)
        if attempt > policy.max_attempts:
            self._dead_letter(run_id, step.name, attempt - 1, f"attempt budget exhausted ({policy.max_attempts}) after interrupted attempts")
            return

        ctx = StepContext(self, run_id, step, attempt, state)
        self.faults.hit("before_handler", run_id, step.name, attempt)
        self.stats["steps_executed"] += 1
        outcome = self._invoke(step, ctx)

        if outcome.error is None:
            try:
                dumps(outcome.output)
                nxt = self._resolve_next(wf, step, outcome.output, ctx)
            except Exception as exc:  # noqa: BLE001 - invalid output / routing is a fatal step error
                outcome = _Outcome(error=FatalError(f"{type(exc).__name__}: {exc}"))
        if outcome.error is not None:
            self._handle_failure(run_id, step, attempt, outcome.error)
            return

        self.faults.hit("after_handler", run_id, step.name, attempt)
        now = self.clock.now()
        state["outputs"][step.name] = outcome.output
        state["visits"][step.name] = int(state["visits"].get(step.name, 0)) + 1
        state["path"].append(step.name)
        done = nxt is None
        with self.store.transaction() as conn:
            self.store.append_event(conn, run_id, now, "STEP_COMPLETED", step.name, attempt, {"output": outcome.output, "next": nxt})
            if done:
                self.store.append_event(conn, run_id, now, "RUN_COMPLETED", None, None, {})
            self.store.update_run(
                conn,
                run_id,
                now,
                state_json=dumps(state),
                current_step=nxt,
                status=Status.COMPLETED if done else Status.RUNNING,
                attempt=0,
                next_attempt_at=0,
                error=None,
                lease_expires=now + self.lease_ttl,
                owner=None if done else self.owner_id,
            )
        self.stats["steps_completed"] += 1
        self.faults.hit("after_commit", run_id, step.name, attempt)

    def _invoke(self, step: Step, ctx: StepContext) -> _Outcome:
        """Run the handler, enforcing the per-attempt timeout with a worker thread.

        Python threads cannot be killed, so a timed-out handler keeps running
        in the background. Its context is marked *abandoned*, so any side effect
        it tries afterwards is refused, and its result is discarded.
        """
        if step.timeout is None:
            try:
                return _Outcome(output=step.handler(ctx))
            except Exception as exc:  # noqa: BLE001
                return _Outcome(error=exc)
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["output"] = step.handler(ctx)
            except BaseException as exc:  # noqa: BLE001 - re-raised/classified in the caller
                box["error"] = exc

        t = threading.Thread(target=target, name=f"step-{step.name}", daemon=True)
        t.start()
        t.join(step.timeout)
        if t.is_alive():
            ctx.abandoned = True
            self.stats["timeouts"] += 1
            return _Outcome(error=StepTimeout(f"step {step.name!r} exceeded timeout of {step.timeout}s"))
        err = box.get("error")
        if err is not None and not isinstance(err, Exception):
            raise err  # SimulatedCrash / KeyboardInterrupt: propagate like a real crash
        return _Outcome(output=box.get("output"), error=err)

    def _resolve_next(self, wf: Workflow, step: Step, output: Any, ctx: StepContext) -> str | None:
        nxt = step.next
        if callable(nxt):
            nxt = nxt(output, ctx)
        if nxt is not None and nxt not in wf.nodes:
            raise WorkflowDefinitionError(f"step {step.name!r} routed to unknown step {nxt!r}")
        return nxt

    def _handle_failure(self, run_id: str, step: Step, attempt: int, exc: BaseException) -> None:
        policy = step.retry
        err = f"{type(exc).__name__}: {exc}"
        now = self.clock.now()
        if self._row(run_id)["cancel_requested"]:
            with self.store.transaction() as conn:
                self.store.append_event(conn, run_id, now, "STEP_FAILED", step.name, attempt, {"error": err, "retryable": False})
            return  # the run loop finalises the cancellation
        retryable = policy.is_retryable(exc)
        if retryable and attempt < policy.max_attempts:
            delay = policy.delay(attempt, run_id, step.name)
            self.stats["retries"] += 1
            with self.store.transaction() as conn:
                self.store.append_event(
                    conn, run_id, now, "STEP_FAILED", step.name, attempt,
                    {"error": err, "retryable": True, "retry_in": round(delay, 6)},
                )
                self.store.update_run(conn, run_id, now, next_attempt_at=now + delay, error=err, lease_expires=now + self.lease_ttl)
            return
        reason = err if not retryable else f"retries exhausted after {attempt} attempts: {err}"
        with self.store.transaction() as conn:
            self.store.append_event(conn, run_id, now, "STEP_FAILED", step.name, attempt, {"error": err, "retryable": retryable})
        self._dead_letter(run_id, step.name, attempt, reason)

    def _dead_letter(self, run_id: str, step: str, attempt: int, reason: str) -> None:
        now = self.clock.now()
        with self.store.transaction() as conn:
            self.store.append_event(conn, run_id, now, "DEAD_LETTERED", step, attempt, {"reason": reason})
            self.store.update_run(conn, run_id, now, status=Status.DEAD_LETTER, error=reason, owner=None, lease_expires=0)

    # ================================================================= leases
    def _acquire(self, run_id: str) -> None:
        now = self.clock.now()
        with self.store.transaction() as conn:
            row = self.store.get_run(run_id, conn)
            owner = row["owner"]
            if owner and owner != self.owner_id and row["lease_expires"] > now and not _owner_dead(owner):
                raise LeaseHeld(f"run {run_id} is leased by {owner} until {row['lease_expires']:.3f}")
            fields: dict[str, Any] = {"owner": self.owner_id, "lease_expires": now + self.lease_ttl}
            if row["status"] == Status.PENDING:
                fields["status"] = Status.RUNNING
            elif owner and owner != self.owner_id:
                self.store.append_event(conn, run_id, now, "LEASE_TAKEN_OVER", row["current_step"], None, {"previous_owner": owner})
            self.store.update_run(conn, run_id, now, **fields)

    def _release(self, run_id: str) -> None:
        with self.store.transaction() as conn:
            row = self.store.get_run(run_id, conn)
            if row and row["owner"] == self.owner_id:
                self.store.update_run(conn, run_id, self.clock.now(), owner=None, lease_expires=0)

    # ================================================================= replay
    def replay(self, run_id: str) -> dict[str, Any]:
        """Rebuild a run's state purely from its event log."""
        events = self.store.events(run_id)
        if not events or events[0].type != "RUN_STARTED":
            raise RunNotFound(run_id)
        state = _empty_state(events[0].payload.get("input"))
        for ev in events[1:]:
            if ev.type == "STEP_COMPLETED":
                state["outputs"][ev.step] = ev.payload["output"]
                state["visits"][ev.step] = int(state["visits"].get(ev.step, 0)) + 1
                state["path"].append(ev.step)
            elif ev.type in ("APPROVAL_DECIDED", "APPROVAL_TIMED_OUT"):
                decision = ev.payload["decision_record"]
                state["approvals"][ev.step] = decision
                state["outputs"][ev.step] = decision
                state["visits"][ev.step] = int(state["visits"].get(ev.step, 0)) + 1
                state["path"].append(ev.step)
        return state

    def _verify_replay(self, run_id: str) -> None:
        snapshot = json.loads(self._row(run_id)["state_json"])
        rebuilt = json.loads(dumps(self.replay(run_id)))
        if rebuilt != snapshot:
            raise InvalidStateError(f"run {run_id}: snapshot diverges from event log replay")

    # ================================================================= approvals
    def _enter_gate(self, run_id: str, wf: Workflow, gate: ApprovalGate, state: dict[str, Any]) -> None:
        now = self.clock.now()
        request: dict[str, Any] = {}
        if gate.prepare is not None:
            try:
                fake_step = Step(name=gate.name, handler=lambda c: None)
                request = gate.prepare(StepContext(self, run_id, fake_step, 1, state)) or {}
                dumps(request)
            except Exception as exc:  # noqa: BLE001
                self._dead_letter(run_id, gate.name, 1, f"approval request preparation failed: {type(exc).__name__}: {exc}")
                return
        pending = {
            "gate": gate.name,
            "requested_at": now,
            "deadline": (now + gate.timeout) if gate.timeout else None,
            "approvers": sorted(gate.approvers),
            "escalated": False,
            "request": request,
        }
        with self.store.transaction() as conn:
            self.store.append_event(conn, run_id, now, "APPROVAL_REQUESTED", gate.name, None, pending)
            self.store.update_run(
                conn, run_id, now, status=Status.WAITING_APPROVAL, approval_json=dumps(pending), owner=None, lease_expires=0
            )

    def approve(self, run_id: str, approver: str, comment: str = "", data: Any = None, gate: str | None = None, resume: bool = True) -> RunInfo:
        """Approve the pending gate (from any process); resumes the run by default."""
        return self._decide(run_id, "approved", approver, comment, data, gate, resume)

    def reject(self, run_id: str, approver: str, comment: str = "", data: Any = None, gate: str | None = None, resume: bool = True) -> RunInfo:
        """Reject the pending gate; routes to ``on_reject`` or fails the run."""
        return self._decide(run_id, "rejected", approver, comment, data, gate, resume)

    def _decide(self, run_id: str, decision: str, approver: str, comment: str, data: Any, gate_name: str | None, resume: bool) -> RunInfo:
        now = self.clock.now()
        with self.store.transaction() as conn:
            row = self.store.get_run(run_id, conn)
            if row is None:
                raise RunNotFound(run_id)
            if row["status"] != Status.WAITING_APPROVAL:
                raise InvalidStateError(f"run {run_id} is {row['status']}, not WAITING_APPROVAL")
            pending = json.loads(row["approval_json"])
            if gate_name is not None and gate_name != pending["gate"]:
                raise ApprovalError(f"run {run_id} is waiting at gate {pending['gate']!r}, not {gate_name!r}")
            if pending["deadline"] is not None and now >= pending["deadline"]:
                raise ApprovalError(f"approval window for gate {pending['gate']!r} has expired")
            allowed = set(pending["approvers"])
            if allowed and approver not in allowed:
                raise ApprovalError(f"{approver!r} is not an authorised approver for {pending['gate']!r} (allowed: {sorted(allowed)})")
            wf = self._workflow(row["workflow"])
            gate = wf.node(pending["gate"])
            assert isinstance(gate, ApprovalGate)
            record = {"decision": decision, "approver": approver, "comment": comment, "data": data, "ts": now}
            if decision == "approved":
                nxt, fail = gate.on_approve, False
            else:
                nxt, fail = gate.on_reject, gate.reject_fails
            self._apply_gate_outcome(conn, row, gate, "APPROVAL_DECIDED", record, nxt, fail, f"rejected by {approver}: {comment}".strip(": "))
        if resume and self._row(run_id)["status"] == Status.RUNNING:
            return self.run(run_id)
        return self.get_run(run_id)

    def _apply_gate_outcome(
        self, conn: Any, row: dict[str, Any], gate: ApprovalGate, event: str, record: dict[str, Any],
        nxt: str | None, fail: bool, fail_reason: str,
    ) -> None:
        run_id, now = row["run_id"], self.clock.now()
        state = json.loads(row["state_json"])
        state["approvals"][gate.name] = record
        state["outputs"][gate.name] = record
        state["visits"][gate.name] = int(state["visits"].get(gate.name, 0)) + 1
        state["path"].append(gate.name)
        if fail:
            status, error, cur = Status.FAILED, fail_reason, gate.name
        elif nxt is None:
            status, error, cur = Status.COMPLETED, None, None
        else:
            status, error, cur = Status.RUNNING, None, nxt
        self.store.append_event(conn, run_id, now, event, gate.name, None, {"decision_record": record, "next": cur if status == Status.RUNNING else None})
        if status == Status.COMPLETED:
            self.store.append_event(conn, run_id, now, "RUN_COMPLETED", None, None, {})
        elif status == Status.FAILED:
            self.store.append_event(conn, run_id, now, "RUN_FAILED", gate.name, None, {"reason": error})
        self.store.update_run(
            conn, run_id, now, state_json=dumps(state), status=status, current_step=cur, error=error, approval_json=None, attempt=0
        )

    def check_timeouts(self) -> list[str]:
        """Escalate or time out every overdue approval; returns affected run ids.
        Call periodically (cron / scheduler) or rely on :meth:`run`, which checks lazily."""
        changed = []
        for row in self.store.list_runs(status=Status.WAITING_APPROVAL):
            if self._check_approval_timeout(row["run_id"]):
                changed.append(row["run_id"])
        return changed

    def _check_approval_timeout(self, run_id: str) -> bool:
        now = self.clock.now()
        with self.store.transaction() as conn:
            row = self.store.get_run(run_id, conn)
            if row is None or row["status"] != Status.WAITING_APPROVAL:
                return False
            pending = json.loads(row["approval_json"])
            if pending["deadline"] is None or now < pending["deadline"]:
                return False
            wf = self.workflows.get(row["workflow"])
            if wf is None:
                return False
            gate = wf.node(pending["gate"])
            assert isinstance(gate, ApprovalGate)
            if gate.escalate_to and not pending["escalated"]:
                pending["escalated"] = True
                pending["approvers"] = sorted(set(pending["approvers"]) | gate.escalate_to) if pending["approvers"] else []
                pending["deadline"] = now + (gate.escalation_timeout or gate.timeout or 0)
                self.store.append_event(conn, run_id, now, "APPROVAL_ESCALATED", gate.name, None, {"escalate_to": sorted(gate.escalate_to), "deadline": pending["deadline"]})
                self.store.update_run(conn, run_id, now, approval_json=dumps(pending))
                return True
            record = {"decision": "timeout", "approver": None, "comment": "approval timed out", "data": None, "ts": now}
            fail = gate.on_timeout == "fail"
            nxt = None if fail else gate.on_timeout
            self._apply_gate_outcome(conn, row, gate, "APPROVAL_TIMED_OUT", record, nxt, fail, f"approval at {gate.name!r} timed out")
            return True

    # ================================================================= signals / cancel / DLQ
    def signal(self, run_id: str, name: str, payload: Any = None) -> None:
        """Deliver a durable signal to a run (read by steps via ``ctx.signal``)."""
        now = self.clock.now()
        with self.store.transaction() as conn:
            if self.store.get_run(run_id, conn) is None:
                raise RunNotFound(run_id)
            self.store.add_signal(conn, run_id, name, payload, now)
            self.store.append_event(conn, run_id, now, "SIGNAL_RECEIVED", None, None, {"name": name, "payload": payload})

    def cancel(self, run_id: str, reason: str = "cancelled") -> RunInfo:
        """Cancel a run. Idle runs (pending, waiting, dead-lettered, or whose
        owner is gone) are cancelled at once. A run that a live engine is
        executing is flagged, and that engine stops at the next step boundary."""
        now = self.clock.now()
        with self.store.transaction() as conn:
            row = self.store.get_run(run_id, conn)
            if row is None:
                raise RunNotFound(run_id)
            if row["status"] in TERMINAL:
                raise InvalidStateError(f"run {run_id} is already {row['status']}")
            self.store.append_event(conn, run_id, now, "CANCEL_REQUESTED", row["current_step"], None, {"reason": reason})
            self.store.update_run(conn, run_id, now, cancel_requested=1, cancel_reason=reason)
            active = (
                row["status"] == Status.RUNNING
                and row["owner"]
                and row["lease_expires"] > now
                and not _owner_dead(row["owner"])
            )
            if not active:
                self._finish_cancel(run_id, reason)
        return self.get_run(run_id)

    def _finish_cancel(self, run_id: str, reason: str | None) -> None:
        now = self.clock.now()  # joins the caller's transaction when nested
        with self.store.transaction() as c:
            self.store.append_event(c, run_id, now, "RUN_CANCELLED", None, None, {"reason": reason})
            self.store.update_run(c, run_id, now, status=Status.CANCELLED, error=reason, owner=None, lease_expires=0, approval_json=None)

    def retry_dead_letter(self, run_id: str, resume: bool = True) -> RunInfo:
        """Re-queue a dead-lettered run at its failed step with a fresh attempt budget."""
        now = self.clock.now()
        with self.store.transaction() as conn:
            row = self.store.get_run(run_id, conn)
            if row is None:
                raise RunNotFound(run_id)
            if row["status"] != Status.DEAD_LETTER:
                raise InvalidStateError(f"run {run_id} is {row['status']}, not DEAD_LETTER")
            self.store.append_event(conn, run_id, now, "DEAD_LETTER_RETRIED", row["current_step"], None, {"previous_error": row["error"]})
            self.store.update_run(conn, run_id, now, status=Status.RUNNING, attempt=0, next_attempt_at=0, error=None)
        return self.run(run_id) if resume else self.get_run(run_id)

    def recover(self) -> list[RunInfo]:
        """Resume every run that is pending or was interrupted mid-execution and
        is not leased by a live engine. Also processes approval timeouts."""
        self.check_timeouts()
        out = []
        for status in (Status.RUNNING, Status.PENDING):
            for row in self.store.list_runs(status=status):
                if row["workflow"] not in self.workflows:
                    continue
                try:
                    out.append(self.run(row["run_id"]))
                except LeaseHeld:
                    continue
        return out

    # ================================================================= inspection
    def get_run(self, run_id: str) -> RunInfo:
        return RunInfo.from_row(self._row(run_id))

    def list_runs(self, status: str | None = None, workflow: str | None = None) -> list[RunInfo]:
        return [RunInfo.from_row(r) for r in self.store.list_runs(status, workflow)]

    def history(self, run_id: str) -> list[Event]:
        self._row(run_id)
        return self.store.events(run_id)

    def dead_letters(self) -> list[RunInfo]:
        return self.list_runs(status=Status.DEAD_LETTER)

    def pending_approvals(self) -> list[RunInfo]:
        return self.list_runs(status=Status.WAITING_APPROVAL)

    # ================================================================= helpers
    def _row(self, run_id: str) -> dict[str, Any]:
        row = self.store.get_run(run_id)
        if row is None:
            raise RunNotFound(run_id)
        return row

    def _workflow(self, name: str) -> Workflow:
        try:
            return self.workflows[name]
        except KeyError:
            raise WorkflowDefinitionError(f"workflow {name!r} is not registered with this engine") from None


def _owner_dead(owner: str) -> bool:
    """True if ``owner`` is a process on this host that no longer exists."""
    try:
        host, pid_s, _ = owner.split(":")
        pid = int(pid_s)
    except ValueError:
        return False
    if host != socket.gethostname() or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False
