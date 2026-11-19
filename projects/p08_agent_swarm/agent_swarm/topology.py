"""Typed topology graph: which agent talks to which, with build-time type checks.

A topology is a networkx ``DiGraph`` whose nodes are agents and whose edges
carry a payload *type*. Forward edges must form a DAG; feedback edges (e.g.
consensus -> coder for revise cycles) are declared separately and must point
backwards to an ancestor. ``validate()`` rejects, before anything runs:

* an edge whose payload type is not produced by the source agent, or not
  accepted by the target agent (the core "typed handoff" guarantee),
* cycles in the forward graph,
* entry nodes that cannot accept the task message type,
* feedback edges that do not point to an ancestor,
* nodes unreachable from an entry point.
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx
from pydantic import BaseModel

from .agents import Agent
from .messages import TaskSpec


class TopologyError(ValueError):
    """The topology is ill-typed or structurally invalid."""


class Topology:
    """A validated, typed agent graph."""

    def __init__(self, entry_type: type[BaseModel] = TaskSpec):
        self.entry_type = entry_type
        self.g = nx.DiGraph()
        self._validated = False

    # -- construction -------------------------------------------------------
    def add(self, *agents: Agent) -> "Topology":
        for a in agents:
            if a.name in self.g:
                raise TopologyError(f"duplicate agent name {a.name!r}")
            self.g.add_node(a.name, agent=a)
        self._validated = False
        return self

    def agent(self, name: str) -> Agent:
        return self.g.nodes[name]["agent"]

    def connect(self, src: str, dst: str, payload: type[BaseModel] | None = None,
                feedback: bool = False) -> "Topology":
        for n in (src, dst):
            if n not in self.g:
                raise TopologyError(f"unknown agent {n!r}")
        s, d = self.agent(src), self.agent(dst)
        if payload is None:
            if feedback:
                options = [t for t in s.output_types if d.accepts(t)]
                if not options:
                    raise TopologyError(f"feedback edge {src}->{dst}: no output type of {src} is accepted by {dst}")
                payload = options[-1]
            else:
                payload = s.output_type
        self.g.add_edge(src, dst, payload=payload, feedback=feedback)
        self._validated = False
        return self

    # -- queries ---------------------------------------------------------------
    @property
    def forward(self) -> nx.DiGraph:
        return nx.subgraph_view(self.g, filter_edge=lambda u, v: not self.g.edges[u, v]["feedback"])

    def entries(self) -> list[str]:
        f = self.forward
        return [n for n in self.g.nodes if f.in_degree(n) == 0]

    def order(self) -> list[str]:
        return list(nx.lexicographical_topological_sort(self.forward))

    def preds(self, node: str) -> list[str]:
        return sorted(self.forward.predecessors(node))

    def feedback_from(self, node: str) -> tuple[str, type[BaseModel]] | None:
        for _, dst, data in self.g.out_edges(node, data=True):
            if data["feedback"]:
                return dst, data["payload"]
        return None

    def nodes_with_role(self, role: object) -> list[str]:
        return [n for n in self.order() if self.agent(n).role == role]

    # -- validation ------------------------------------------------------------
    def validate(self) -> "Topology":
        errors: list[str] = []
        if self.g.number_of_nodes() == 0:
            raise TopologyError("topology has no agents")
        f = self.forward
        if not nx.is_directed_acyclic_graph(f):
            cycle = nx.find_cycle(f)
            errors.append(f"forward edges contain a cycle: {' -> '.join(u for u, _ in cycle)}")
        for u, v, data in self.g.edges(data=True):
            t: type[BaseModel] = data["payload"]
            src, dst = self.agent(u), self.agent(v)
            if not any(issubclass(t, o) for o in src.output_types):
                errors.append(f"edge {u}->{v}: {u} ({src.role.value}) cannot produce {t.__name__}; "
                              f"it outputs {[o.__name__ for o in src.output_types]}")
            if not dst.accepts(t):
                errors.append(f"edge {u}->{v}: {t.__name__} is not accepted by {v} ({dst.role.value}); "
                              f"it accepts {[i.__name__ for i in dst.input_types]}")
            if data["feedback"] and u not in nx.descendants(f, v):
                errors.append(f"feedback edge {u}->{v} must point to an upstream agent")
        entries = self.entries()
        for e in entries:
            if not self.agent(e).accepts(self.entry_type):
                errors.append(f"entry agent {e} does not accept {self.entry_type.__name__}")
        if not errors:
            reach: set[str] = set(entries)
            for e in entries:
                reach |= nx.descendants(f, e)
            missing = set(self.g.nodes) - reach
            if missing:
                errors.append(f"unreachable agents: {sorted(missing)}")
        if errors:
            raise TopologyError("invalid topology:\n  - " + "\n  - ".join(errors))
        self._validated = True
        return self

    @property
    def validated(self) -> bool:
        return self._validated

    def describe(self) -> str:
        """ASCII rendering of nodes and typed edges."""
        lines = []
        for u, v, d in sorted(self.g.edges(data=True), key=lambda e: (e[2]["feedback"], e[0], e[1])):
            arrow = "~~>" if d["feedback"] else "-->"
            lines.append(f"  {u:<12} {arrow} {v:<12} [{d['payload'].__name__}]")
        return "\n".join(lines) if lines else "  (single node: " + ", ".join(self.g.nodes) + ")"


def single_agent_topology(coder: Agent) -> Topology:
    """Coder alone: TaskSpec -> CodeArtifact, no research, no review."""
    return Topology().add(coder).validate()


def review_topology(researcher: Agent, coder: Agent, reviewers: Iterable[Agent], panel: Agent) -> Topology:
    """researcher -> coder -> reviewers (fan-out) -> panel (fan-in) ~~> coder (revise loop)."""
    reviewers = list(reviewers)
    t = Topology().add(researcher, coder, *reviewers, panel)
    t.connect(researcher.name, coder.name)
    for r in reviewers:
        t.connect(coder.name, r.name)
        t.connect(r.name, panel.name)
    t.connect(panel.name, coder.name, feedback=True)
    return t.validate()
