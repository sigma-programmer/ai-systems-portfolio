"""Golden trajectories expressed as partial-order DAGs.

Nodes are *required step patterns*; an edge ``u -> v`` means "u must happen
before v". Nodes without a path between them may happen in any order (or in
parallel). Nodes may be ``optional`` (not counted against recall when
missing) or ``forbidden`` (any matching step is a hard violation).
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .matchers import ArgMatcher, match_args, semantic_similarity
from .schema import Step, StepType


class GoldenNode(BaseModel):
    """A step pattern in a golden DAG."""

    id: str
    step_type: StepType = StepType.TOOL_CALL
    tool: str | None = None  # "*" matches any tool
    args: list[ArgMatcher] = Field(default_factory=list)
    content: str | None = None  # semantic target for llm steps
    content_threshold: float = 0.3
    optional: bool = False
    forbidden: bool = False
    max_repeats: int = Field(default=1, ge=1)
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> "GoldenNode":
        if self.step_type == StepType.TOOL_CALL and not self.tool:
            raise ValueError(f"node {self.id!r}: tool_call nodes need a tool")
        return self

    def match(self, step: Step) -> tuple[bool, float, list[str]]:
        """Does ``step`` instantiate this pattern? Returns (ok, score, notes)."""
        if step.type != self.step_type:
            return False, 0.0, []
        if self.step_type == StepType.TOOL_CALL:
            if self.tool != "*" and step.tool != self.tool:
                return False, 0.0, []
            return match_args(self.args, step.args)
        if self.step_type == StepType.LLM and self.content:
            s = semantic_similarity(step.content, self.content)
            ok = s >= self.content_threshold
            return ok, s, [] if ok else [f"content similarity {s:.2f} < {self.content_threshold}"]
        if self.tool and step.tool != self.tool:
            return False, 0.0, []
        return True, 1.0, []

    def tool_matches(self, step: Step) -> bool:
        """Same step type and tool name, ignoring arguments."""
        return step.type == self.step_type and (
            self.step_type != StepType.TOOL_CALL or self.tool in ("*", step.tool)
        )


class GoldenDAG(BaseModel):
    """Golden reference trajectory for one task."""

    task_id: str
    description: str = ""
    nodes: list[GoldenNode]
    edges: list[tuple[str, str]] = Field(default_factory=list)
    answer: ArgMatcher | None = None  # applied to {"answer": final_answer}
    max_steps: int | None = None
    max_tokens: int | None = None
    graded_step_types: list[StepType] = Field(default_factory=lambda: [StepType.TOOL_CALL])

    _graph: nx.DiGraph | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_graph(self) -> "GoldenDAG":
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate golden node ids")
        known = set(ids)
        by_id = {n.id: n for n in self.nodes}
        for u, v in self.edges:
            if u not in known or v not in known:
                raise ValueError(f"edge ({u}, {v}) references unknown node")
            if by_id[u].forbidden or by_id[v].forbidden:
                raise ValueError("forbidden nodes cannot participate in ordering edges")
        g = nx.DiGraph()
        g.add_nodes_from(ids)
        g.add_edges_from(self.edges)
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError(f"golden graph for {self.task_id!r} has a cycle")
        self._graph = g
        return self

    @property
    def graph(self) -> nx.DiGraph:
        assert self._graph is not None
        return self._graph

    @cached_property
    def node_map(self) -> dict[str, GoldenNode]:
        return {n.id: n for n in self.nodes}

    @cached_property
    def ancestors(self) -> dict[str, frozenset[str]]:
        return {n: frozenset(nx.ancestors(self.graph, n)) for n in self.graph.nodes}

    @property
    def required_ids(self) -> list[str]:
        return [n.id for n in self.nodes if not n.optional and not n.forbidden]

    @property
    def allowed_nodes(self) -> list[GoldenNode]:
        return [n for n in self.nodes if not n.forbidden]

    @property
    def forbidden_nodes(self) -> list[GoldenNode]:
        return [n for n in self.nodes if n.forbidden]

    def ordering_pairs(self) -> list[tuple[str, str]]:
        """All (ancestor, descendant) pairs in the transitive closure."""
        return [(a, d) for d, anc in self.ancestors.items() for a in sorted(anc)]

    @classmethod
    def load(cls, path: str | Path) -> "GoldenDAG":
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=1, by_alias=True, exclude_none=True))


def load_goldens(directory: str | Path) -> dict[str, GoldenDAG]:
    """Load every ``*.json`` golden DAG in a directory, keyed by task_id."""
    out: dict[str, GoldenDAG] = {}
    for p in sorted(Path(directory).glob("*.json")):
        g = GoldenDAG.load(p)
        out[g.task_id] = g
    return out


def golden_to_dict(g: GoldenDAG) -> dict[str, Any]:
    return g.model_dump(mode="json", by_alias=True, exclude_none=True)
