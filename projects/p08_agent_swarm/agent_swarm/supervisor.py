"""Supervisor: plans, routes over the typed topology, loops, budgets, terminates.

Execution model
---------------
1. ``plan(task)`` decomposes the task into subgoals (description clauses +
   one "satisfy example" goal per example) and routes each topology node to a
   plan step in topological order.
2. A forward pass runs nodes in topological order; each node receives the
   outputs of its forward predecessors (fan-in = list of messages, e.g. N
   verdicts for the consensus panel). Every hand-off goes through
   ``Agent.handle`` (type check + validation + repair/fallback).
3. If the terminal node emits a rejecting ``ConsensusDecision`` and it has a
   feedback edge, the supervisor builds a typed ``RevisionRequest`` and
   re-runs the sub-graph downstream of the feedback target (coder ↔ reviewer
   revise cycle) until approval, ``max_revisions``, or a budget is exhausted.
4. Conflicting reviewer test evidence is escalated to ``arbitrate``: the
   supervisor re-executes the *trusted* suite itself and its result is final.

Every dispatch, output, repair, escalation and termination is appended to the
event trace (``SwarmState.events``); ``transcript()`` renders it.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from pydantic import BaseModel

from .agents import AgentContext, ArbiterResult, HandoffError, SynthesisExhausted, review_suite
from .consensus import ConsensusPanel
from .messages import (
    CodeArtifact,
    ConsensusDecision,
    Event,
    Plan,
    PlanStep,
    ResearchFindings,
    ReviewVerdict,
    RevisionRequest,
    Role,
    SwarmState,
    TaskSpec,
)
from .sandbox import Sandbox
from .topology import Topology

_ROLE_GOALS = {
    Role.RESEARCHER: "gather requirements and edge cases from the knowledge base",
    Role.CODER: "synthesise an implementation consistent with all known examples",
    Role.REVIEWER: "verify the implementation (static analysis and/or sandboxed tests)",
    Role.CONSENSUS: "aggregate verdicts, resolve conflicts, approve or request revision",
}


@dataclass
class Budget:
    """Hard limits for one run."""

    max_steps: int = 80
    max_tokens: int = 250_000
    max_revisions: int = 4
    max_wall_s: float = 60.0


class BudgetExceeded(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Supervisor:
    """Orchestrates one task at a time over a validated topology."""

    def __init__(self, topology: Topology, budget: Budget | None = None, sandbox: Sandbox | None = None):
        if not topology.validated:
            topology.validate()
        self.topology = topology
        self.budget = budget or Budget()
        self.sandbox = sandbox or Sandbox()
        self._state: SwarmState | None = None
        self._t0 = 0.0

    # -- planning ----------------------------------------------------------------
    def plan(self, task: TaskSpec) -> Plan:
        clauses = [c.strip() for c in re.split(r"[.;]|,\s*(?:and|then)\s+|\s+and\s+then\s+", task.description)
                   if len(c.strip()) > 3]
        subgoals = clauses + [f"satisfy example {t.describe(task.func_name)}" for t in task.examples]
        steps = [PlanStep(index=i, node=n, role=self.topology.agent(n).role,
                          goal=_ROLE_GOALS[self.topology.agent(n).role])
                 for i, n in enumerate(self.topology.order())]
        return Plan(task_name=task.name, subgoals=subgoals, steps=steps)

    # -- tracing ------------------------------------------------------------------
    def _emit(self, kind: str, node: str = "", detail: str = "", tokens: int = 0) -> None:
        st = self._state
        if st is None:
            return
        st.events.append(Event(seq=len(st.events), t_ms=round((time.perf_counter() - self._t0) * 1000, 2),
                               kind=kind, node=node, detail=detail, tokens=tokens))

    # -- arbitration --------------------------------------------------------------
    def arbitrate(self, verdicts: list[ReviewVerdict]) -> ArbiterResult:
        """Re-execute the trusted suite on the current artifact; settle the dispute."""
        st = self._state
        assert st is not None and st.final_artifact is not None
        st.escalations += 1
        art = st.final_artifact
        ctx = AgentContext(state=st, sandbox=self.sandbox)
        suite = review_suite(ctx)
        rep = self.sandbox.run(art.source, st.task.func_name, suite, arity=len(st.task.params))
        failed = set(rep.failed_ids) if rep.compile_error is None else {t.id for t in suite}
        approved = not failed
        contradicted = []
        for v in verdicts:
            claimed = {i for e in v.evidence for i in e.failed_test_ids}
            if v.approved != approved or (claimed and not claimed <= failed):
                contradicted.append(v.reviewer)
        self._emit("escalation", "supervisor",
                   f"re-ran {len(suite)} trusted tests: {'PASS' if approved else f'{len(failed)} failed'}"
                   + (f"; contradicted {contradicted}" if contradicted else ""))
        return ArbiterResult(approved=approved, failing=[t for t in suite if t.id in failed],
                             contradicted=contradicted)

    # -- execution ---------------------------------------------------------------
    def _check_budget(self) -> None:
        st = self._state
        assert st is not None
        if st.steps >= self.budget.max_steps:
            raise BudgetExceeded("steps")
        if st.tokens >= self.budget.max_tokens:
            raise BudgetExceeded("tokens")
        if time.perf_counter() - self._t0 > self.budget.max_wall_s:
            raise BudgetExceeded("wall_time")

    def _record(self, out: BaseModel) -> str:
        st = self._state
        assert st is not None
        if isinstance(out, ResearchFindings):
            st.findings = out
            return f"{len(out.requirements)} requirements, {len(out.edge_cases)} edge cases from {out.sources}"
        if isinstance(out, CodeArtifact):
            st.artifacts.append(out)
            return f"rev {out.revision} [{out.hash}] via {out.origin}"
        if isinstance(out, ReviewVerdict):
            st.verdicts.append(out)
            fails = sum(len(e.failed_test_ids) for e in out.evidence)
            return (f"{'APPROVE' if out.approved else 'REJECT'} ({out.strength.name.lower()}"
                    f"{f', {fails} failing' if fails else ''}, conf {out.confidence:.2f})")
        if isinstance(out, ConsensusDecision):
            st.decisions.append(out)
            seen = {t.id for t in st.counterexamples}
            st.counterexamples += [t for t in out.counterexamples if t.id not in seen]
            return (f"{'APPROVED' if out.approved else 'REJECTED'} by {out.rule} via {out.resolution} "
                    f"(share {out.approve_share:.2f})")
        return type(out).__name__

    def _invoke(self, node: str, inputs: list[BaseModel], ctx: AgentContext) -> BaseModel:
        self._check_budget()
        st = self._state
        assert st is not None
        st.steps += 1
        agent = self.topology.agent(node)
        self._emit("dispatch", node, f"{agent.role.value} <- {[type(m).__name__ for m in inputs]}")
        before = st.tokens
        out = agent.handle(inputs, ctx)
        self._emit("output", node, self._record(out), tokens=st.tokens - before)
        return out

    def run(self, task: TaskSpec) -> SwarmState:
        """Run the swarm on one task and return the final typed state."""
        self._t0 = time.perf_counter()
        st = SwarmState(task=task)
        self._state = st
        st.plan = self.plan(task)
        self._emit("plan", "supervisor", f"{len(st.plan.subgoals)} subgoals; route: "
                   + " -> ".join(s.node for s in st.plan.steps))
        ctx = AgentContext(state=st, sandbox=self.sandbox, emit=self._emit, arbiter=self.arbitrate)
        order = self.topology.order()
        outputs: dict[str, BaseModel | list[BaseModel]] = {}
        pending: dict[str, list[BaseModel]] = {}
        start = 0
        try:
            while True:
                for node in order[start:]:
                    preds = self.topology.preds(node)
                    inputs: list[BaseModel] = [task] if not preds else []
                    for p in preds:
                        inputs.append(outputs[p])  # type: ignore[arg-type]
                    inputs += pending.pop(node, [])
                    outputs[node] = self._invoke(node, inputs, ctx)
                terminal = outputs[order[-1]]
                if not isinstance(terminal, ConsensusDecision):
                    st.status = "completed_unreviewed"
                    break
                if terminal.approved:
                    st.status = "approved"
                    break
                fb = self.topology.feedback_from(order[-1])
                if fb is None:
                    st.status = "rejected"
                    break
                if st.revisions >= self.budget.max_revisions:
                    st.status = "max_revisions"
                    break
                st.revisions += 1
                target, payload = fb
                if payload is not RevisionRequest:
                    raise HandoffError(f"unsupported feedback payload {payload.__name__}")
                rr = ConsensusPanel.to_revision(terminal, st)
                self._emit("revise", target, f"revision {st.revisions}: {len(rr.counterexamples)} counterexamples")
                pending[target] = [rr]
                start = order.index(target)
        except BudgetExceeded as e:
            st.status = f"budget_exhausted:{e.reason}"
        except SynthesisExhausted as e:
            st.status = "coder_exhausted"
            self._emit("error", "", str(e))
        except HandoffError as e:
            st.status = "handoff_failed"
            self._emit("error", "", str(e))
        self._emit("terminate", "supervisor",
                   f"status={st.status} revisions={st.revisions} steps={st.steps} tokens={st.tokens}")
        self._state = None
        return st


def transcript(state: SwarmState, max_detail: int = 110) -> str:
    """Human-readable event trace."""
    lines = [f"== transcript: {state.task.name} ({state.status}) =="]
    for e in state.events:
        tok = f" +{e.tokens}tok" if e.tokens else ""
        lines.append(f"[{e.seq:03d} {e.t_ms:8.1f}ms] {e.kind:<17} {e.node:<12} {e.detail[:max_detail]}{tok}")
    return "\n".join(lines)
