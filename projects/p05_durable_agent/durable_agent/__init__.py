"""durable_agent: a durable state-machine engine for long-running agent workflows.

Checkpointing to SQLite after every step, deterministic replay on resume,
idempotent side effects, retries with exponential backoff and full jitter,
dead-lettering, human approval gates with escalation, signals, cancellation
and an inspection API.
"""
from .clock import Clock, FakeClock, SystemClock
from .engine import Engine, RunInfo, Status, StepContext
from .errors import (
    ApprovalError,
    DurableError,
    EffectAbandoned,
    FatalError,
    InvalidStateError,
    LeaseHeld,
    RetryableError,
    RunNotFound,
    StepTimeout,
    WorkflowDefinitionError,
    WorkflowVersionMismatch,
)
from .faults import CRASH_POINTS, CrashAt, ExitAt, FaultInjector, RandomFaults, SimulatedCrash
from .retry import NO_RETRY, RetryPolicy
from .store import Event, SQLiteStore
from .workflow import END, ApprovalGate, Branch, Step, Workflow, branch

__all__ = [
    "Clock", "FakeClock", "SystemClock",
    "Engine", "RunInfo", "Status", "StepContext",
    "ApprovalError", "DurableError", "EffectAbandoned", "FatalError", "InvalidStateError", "LeaseHeld",
    "RetryableError", "RunNotFound", "StepTimeout", "WorkflowDefinitionError", "WorkflowVersionMismatch",
    "CRASH_POINTS", "CrashAt", "ExitAt", "FaultInjector", "RandomFaults", "SimulatedCrash",
    "NO_RETRY", "RetryPolicy",
    "Event", "SQLiteStore",
    "END", "ApprovalGate", "Branch", "Step", "Workflow", "branch",
]
