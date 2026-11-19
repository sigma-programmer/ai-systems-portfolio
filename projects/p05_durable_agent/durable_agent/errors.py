"""Exception hierarchy used to classify step failures.

The engine distinguishes *retryable* failures (transient: network blips, rate
limits, timeouts) from *fatal* ones (bad input, bugs, policy violations).
Retryable failures are retried with backoff until ``max_attempts``; fatal ones,
and retryable ones that exhaust their budget, move the run to DEAD_LETTER.
"""
from __future__ import annotations


class DurableError(Exception):
    """Base class for engine errors."""


class RetryableError(DurableError):
    """A transient failure; the step may succeed if retried."""


class FatalError(DurableError):
    """A permanent failure; retrying cannot help."""


class StepTimeout(RetryableError, TimeoutError):
    """A step exceeded its per-attempt timeout."""


class WorkflowDefinitionError(DurableError):
    """The workflow graph is invalid (unknown targets, missing start, ...)."""


class WorkflowVersionMismatch(DurableError):
    """A persisted run was started with a different workflow definition."""


class RunNotFound(DurableError, KeyError):
    """No run with the given id exists."""


class InvalidStateError(DurableError):
    """The requested operation is not valid in the run's current status."""


class ApprovalError(DurableError):
    """Approval/rejection refused (wrong gate, unauthorised approver, ...)."""


class LeaseHeld(DurableError):
    """Another live engine instance currently owns the run."""


class EffectAbandoned(DurableError):
    """A side effect was attempted from a step attempt that already timed out."""
