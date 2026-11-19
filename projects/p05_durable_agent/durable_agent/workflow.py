"""Typed workflow definitions: a graph of steps with (conditional) transitions.

A :class:`Workflow` is a named state machine. Each node is either

* a **task step**: a handler ``fn(ctx) -> JSON-serializable output`` with a
  retry policy, an optional per-attempt timeout and a transition, or
* an **approval gate**: suspends the run in ``WAITING_APPROVAL`` until a human
  approves/rejects it (possibly from another process), with an optional
  timeout and escalation.

Transitions (``next``) may be:

* ``None`` / :data:`END` — the run completes after this step;
* a step name;
* a callable ``(output, ctx) -> step name | None`` for conditional branches
  (see :func:`branch` for the common "switch on a key" case).

Example::

    wf = Workflow("demo", start="fetch")

    @wf.step(next="decide", retry=RetryPolicy(max_attempts=5), timeout=10)
    def fetch(ctx):
        return {"n": 3}

    @wf.step(next=branch(lambda out, ctx: out["big"], {True: "review", False: END}))
    def decide(ctx):
        return {"big": ctx.outputs["fetch"]["n"] > 2}

    wf.approval("review", on_approve=END, on_reject="fetch", approvers={"editor"})
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Union

from .errors import WorkflowDefinitionError
from .retry import RetryPolicy

if TYPE_CHECKING:  # pragma: no cover
    from .engine import StepContext

END: None = None
"""Transition target meaning "the run is complete"."""

Handler = Callable[["StepContext"], Any]
Router = Callable[[Any, "StepContext"], Union[str, None]]
Transition = Union[str, None, Router]


@dataclass(frozen=True)
class Branch:
    """Switch on ``key(output, ctx)``; unmatched keys go to ``default``."""

    key: Callable[[Any, Any], Any]
    cases: Mapping[Any, str | None]
    default: str | None = None
    has_default: bool = False

    def __call__(self, output: Any, ctx: Any) -> str | None:
        k = self.key(output, ctx)
        if k in self.cases:
            return self.cases[k]
        if self.has_default:
            return self.default
        raise KeyError(f"branch has no case for {k!r}")

    def targets(self) -> set[str]:
        out = {t for t in self.cases.values() if t is not None}
        if self.has_default and self.default is not None:
            out.add(self.default)
        return out


_MISSING = object()


def branch(key: Callable[[Any, Any], Any], cases: Mapping[Any, str | None], default: Any = _MISSING) -> Branch:
    """Build a statically checkable conditional transition."""
    if default is _MISSING:
        return Branch(key, dict(cases))
    return Branch(key, dict(cases), default, True)


@dataclass(frozen=True)
class Step:
    """A task step."""

    name: str
    handler: Handler
    next: Transition = END
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    timeout: float | None = None
    max_visits: int = 20
    description: str = ""

    kind = "task"


@dataclass(frozen=True)
class ApprovalGate:
    """A human-in-the-loop gate.

    Args:
        on_approve / on_reject: next step (or END) for each decision. A
            rejection with ``on_reject=None`` fails the run.
        approvers: identities allowed to decide (empty = anyone).
        timeout: seconds to wait before escalation/timeout handling.
        escalate_to: extra approvers added on timeout; the deadline is then
            extended by ``escalation_timeout`` (once).
        on_timeout: where to go when the (escalated) deadline passes;
            ``"fail"`` (default) fails the run, otherwise a step name / END.
        prepare: ``fn(ctx) -> dict`` building the review request payload
            shown to approvers (e.g. the draft to review).
    """

    name: str
    on_approve: str | None = END
    on_reject: str | None = END
    reject_fails: bool = True
    approvers: frozenset[str] = frozenset()
    timeout: float | None = None
    escalate_to: frozenset[str] = frozenset()
    escalation_timeout: float | None = None
    on_timeout: str | None = "fail"
    prepare: Callable[["StepContext"], dict] | None = None
    max_visits: int = 20
    description: str = ""

    kind = "approval"


Node = Union[Step, ApprovalGate]


class Workflow:
    """A named, validated step graph."""

    def __init__(self, name: str, start: str | None = None, version: str = "1") -> None:
        self.name = name
        self.start = start
        self.version = version
        self.nodes: dict[str, Node] = {}

    # ------------------------------------------------------------ building
    def add(self, node: Node) -> Node:
        if node.name in self.nodes:
            raise WorkflowDefinitionError(f"duplicate step {node.name!r}")
        self.nodes[node.name] = node
        if self.start is None:
            self.start = node.name
        return node

    def step(
        self,
        name: str | None = None,
        *,
        next: Transition = END,
        retry: RetryPolicy | None = None,
        timeout: float | None = None,
        max_visits: int = 20,
    ) -> Callable[[Handler], Handler]:
        """Decorator registering a task step (name defaults to the function name)."""

        def deco(fn: Handler) -> Handler:
            self.add(
                Step(
                    name=name or fn.__name__,
                    handler=fn,
                    next=next,
                    retry=retry or RetryPolicy(),
                    timeout=timeout,
                    max_visits=max_visits,
                    description=(inspect.getdoc(fn) or "").split("\n")[0],
                )
            )
            return fn

        return deco

    def approval(
        self,
        name: str,
        *,
        on_approve: str | None = END,
        on_reject: str | None = END,
        reject_fails: bool | None = None,
        approvers: set[str] | frozenset[str] = frozenset(),
        timeout: float | None = None,
        escalate_to: set[str] | frozenset[str] = frozenset(),
        escalation_timeout: float | None = None,
        on_timeout: str | None = "fail",
        prepare: Callable[["StepContext"], dict] | None = None,
        max_visits: int = 20,
    ) -> ApprovalGate:
        """Register an approval gate.

        By default a rejection fails the run unless ``on_reject`` names a step
        (e.g. a revision loop back to ``draft``).
        """
        gate = ApprovalGate(
            name=name,
            on_approve=on_approve,
            on_reject=on_reject,
            reject_fails=(on_reject is None) if reject_fails is None else reject_fails,
            approvers=frozenset(approvers),
            timeout=timeout,
            escalate_to=frozenset(escalate_to),
            escalation_timeout=escalation_timeout,
            on_timeout=on_timeout,
            prepare=prepare,
            max_visits=max_visits,
        )
        self.add(gate)
        return gate

    # ------------------------------------------------------------ queries
    def node(self, name: str) -> Node:
        try:
            return self.nodes[name]
        except KeyError:
            raise WorkflowDefinitionError(f"workflow {self.name!r} has no step {name!r}") from None

    def static_targets(self, node: Node) -> set[str]:
        if isinstance(node, ApprovalGate):
            out = {t for t in (node.on_approve, node.on_reject) if t is not None}
            if node.on_timeout not in (None, "fail"):
                out.add(node.on_timeout)
            return out
        nxt = node.next
        if nxt is None:
            return set()
        if isinstance(nxt, str):
            return {nxt}
        if isinstance(nxt, Branch):
            return nxt.targets()
        return set()  # opaque router: checked at runtime

    def validate(self) -> "Workflow":
        """Check the graph; raises :class:`WorkflowDefinitionError`."""
        if not self.nodes:
            raise WorkflowDefinitionError(f"workflow {self.name!r} has no steps")
        if self.start not in self.nodes:
            raise WorkflowDefinitionError(f"start step {self.start!r} is not defined")
        for node in self.nodes.values():
            for t in self.static_targets(node):
                if t not in self.nodes:
                    raise WorkflowDefinitionError(f"step {node.name!r} transitions to unknown step {t!r}")
            if isinstance(node, Step) and node.timeout is not None and node.timeout <= 0:
                raise WorkflowDefinitionError(f"step {node.name!r}: timeout must be > 0")
        return self

    def fingerprint(self) -> str:
        """Hash of the graph shape; persisted runs refuse to resume on a changed graph."""
        shape: dict[str, Any] = {"name": self.name, "version": self.version, "start": self.start, "nodes": {}}
        for n in self.nodes.values():
            nxt: Any
            if isinstance(n, ApprovalGate):
                nxt = [n.on_approve, n.on_reject, n.on_timeout]
            elif n.next is None or isinstance(n.next, str):
                nxt = n.next
            elif isinstance(n.next, Branch):
                nxt = sorted(map(str, n.next.targets()))
            else:
                nxt = "<router>"
            shape["nodes"][n.name] = [n.kind, nxt]
        return hashlib.sha256(json.dumps(shape, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def describe(self) -> str:
        """Human-readable listing of the graph."""
        lines = [f"workflow {self.name} (v{self.version}, start={self.start})"]
        for n in self.nodes.values():
            if isinstance(n, ApprovalGate):
                lines.append(f"  [gate] {n.name}: approve->{n.on_approve or 'END'} reject->{n.on_reject or ('FAIL' if n.reject_fails else 'END')}")
            else:
                tg = self.static_targets(n)
                arrow = ", ".join(sorted(tg)) if tg else ("END" if n.next is None else "<dynamic>")
                lines.append(f"  [task] {n.name} -> {arrow}")
        return "\n".join(lines)
