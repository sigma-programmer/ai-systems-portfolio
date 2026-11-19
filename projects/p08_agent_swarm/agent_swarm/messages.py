"""Typed messages and shared state for the swarm.

Every hand-off between agents is one of these Pydantic v2 models. Validators
encode *semantic* contracts, not just shapes: a `CodeArtifact` must parse and
define the requested function; a `ReviewVerdict` cannot approve code while
its own test evidence reports failures. An agent whose output violates a
contract is sent back for structured repair by `Agent.handle`, it never
crashes the run.
"""
from __future__ import annotations

import ast
import hashlib
from enum import Enum, IntEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Role(str, Enum):
    """The role an agent plays in the topology."""

    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"
    CONSENSUS = "consensus"


class _Msg(BaseModel):
    """Base for all messages: immutable-ish, strict about unknown fields."""

    model_config = ConfigDict(extra="forbid")


class TestCase(_Msg):
    """One executable check: ``func(*args) == expected`` or ``raises`` an exception."""

    __test__ = False  # keep pytest from collecting this class

    id: str = Field(min_length=1)
    args: list[Any]
    expected: Any = None
    raises: str | None = None
    source: Literal["spec", "research", "hidden", "counterexample"] = "spec"
    note: str = ""

    @field_validator("raises")
    @classmethod
    def _exception_name(cls, v: str | None) -> str | None:
        if v is not None and not v.isidentifier():
            raise ValueError(f"raises must be an exception class name, got {v!r}")
        return v

    def describe(self, func_name: str) -> str:
        args = ", ".join(repr(a) for a in self.args)
        outcome = f"raises {self.raises}" if self.raises else f"== {self.expected!r}"
        return f"{func_name}({args}) {outcome}"


class TaskSpec(_Msg):
    """What the user wants: a single Python function described in prose + examples."""

    name: str = Field(min_length=1)
    func_name: str
    params: list[str]
    description: str = Field(min_length=10)
    examples: list[TestCase] = Field(min_length=1)

    @field_validator("func_name")
    @classmethod
    def _ident(cls, v: str) -> str:
        if not v.isidentifier():
            raise ValueError(f"func_name must be a Python identifier, got {v!r}")
        return v


class PlanStep(_Msg):
    """One step of the supervisor's decomposition."""

    index: int
    node: str
    role: Role
    goal: str


class Plan(_Msg):
    """Supervisor decomposition of a task into subgoals and routed steps."""

    task_name: str
    subgoals: list[str]
    steps: list[PlanStep]


class ResearchFindings(_Msg):
    """Requirements and edge cases the researcher extracted from the knowledge base."""

    task: TaskSpec
    requirements: list[str]
    edge_cases: list[TestCase]
    sources: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "ResearchFindings":
        ids = [t.id for t in self.edge_cases]
        if len(ids) != len(set(ids)):
            raise ValueError("edge case ids must be unique")
        return self

    def test_suite(self) -> list[TestCase]:
        """Spec examples + researched edge cases: the shared, trusted review suite."""
        return list(self.task.examples) + list(self.edge_cases)


def source_hash(source: str) -> str:
    """Stable short hash of normalised source (whitespace/comment insensitive)."""
    try:
        norm = ast.unparse(ast.parse(source))
    except SyntaxError:
        norm = source
    return hashlib.sha1(norm.encode()).hexdigest()[:12]


class CodeArtifact(_Msg):
    """Candidate implementation. Must parse and define ``func_name`` at top level."""

    task_name: str
    func_name: str
    source: str = Field(min_length=1)
    revision: int = Field(ge=0)
    origin: str = "unknown"
    rationale: str = ""

    @model_validator(mode="after")
    def _defines_function(self) -> "CodeArtifact":
        try:
            tree = ast.parse(self.source)
        except SyntaxError as e:
            raise ValueError(f"source does not parse: {e.msg} (line {e.lineno})") from e
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        if self.func_name not in names:
            raise ValueError(f"source must define top-level function {self.func_name!r}")
        return self

    @property
    def hash(self) -> str:
        return source_hash(self.source)


class EvidenceKind(IntEnum):
    """Evidence strength tiers; higher beats lower in conflict resolution."""

    OPINION = 1
    STATIC_ANALYSIS = 2
    TEST_EXECUTION = 3


class Evidence(_Msg):
    """Support for a verdict. Test evidence is verifiable by re-execution."""

    kind: EvidenceKind
    detail: str = ""
    tests_run: int = Field(default=0, ge=0)
    failed_test_ids: list[str] = Field(default_factory=list)
    artifact_hash: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> "Evidence":
        if self.kind == EvidenceKind.TEST_EXECUTION:
            if self.tests_run == 0:
                raise ValueError("test-execution evidence must report tests_run > 0")
            if len(self.failed_test_ids) > self.tests_run:
                raise ValueError("more failed tests than tests run")
        return self


class ReviewVerdict(_Msg):
    """One reviewer's vote on one artifact."""

    reviewer: str
    strategy: str
    artifact_hash: str
    approved: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(min_length=1)
    feedback: list[str] = Field(default_factory=list)
    counterexamples: list[TestCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def _approval_matches_evidence(self) -> "ReviewVerdict":
        failing = any(e.failed_test_ids for e in self.evidence if e.kind == EvidenceKind.TEST_EXECUTION)
        if self.approved and failing:
            raise ValueError("verdict approves code while its own test evidence reports failures")
        return self

    @property
    def strength(self) -> EvidenceKind:
        return max(e.kind for e in self.evidence)


class Vote(_Msg):
    """A verdict as counted by the consensus panel."""

    reviewer: str
    approved: bool
    weight: float
    strength: EvidenceKind
    trusted: bool = True


class ConsensusDecision(_Msg):
    """Aggregated decision of the reviewer panel."""

    artifact_hash: str
    approved: bool
    rule: str
    resolution: Literal["unanimous", "vote", "evidence", "escalated"]
    approve_share: float = Field(ge=0.0, le=1.0)
    votes: list[Vote]
    counterexamples: list[TestCase] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)
    rationale: str = ""


class RevisionRequest(_Msg):
    """Feedback-edge message: what the coder must fix on the next revision."""

    task_name: str
    revision: int = Field(ge=1)
    counterexamples: list[TestCase] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)
    rejected_hashes: list[str] = Field(default_factory=list)


class Event(_Msg):
    """One line of the swarm transcript."""

    seq: int
    t_ms: float
    kind: str
    node: str = ""
    detail: str = ""
    tokens: int = 0


class SwarmState(_Msg):
    """Everything the supervisor knows about one run; passed by reference to agents."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task: TaskSpec
    plan: Plan | None = None
    findings: ResearchFindings | None = None
    artifacts: list[CodeArtifact] = Field(default_factory=list)
    verdicts: list[ReviewVerdict] = Field(default_factory=list)
    decisions: list[ConsensusDecision] = Field(default_factory=list)
    counterexamples: list[TestCase] = Field(default_factory=list)
    revisions: int = 0
    steps: int = 0
    tokens: int = 0
    escalations: int = 0
    repairs: int = 0
    status: str = "running"
    events: list[Event] = Field(default_factory=list)

    @property
    def final_artifact(self) -> CodeArtifact | None:
        return self.artifacts[-1] if self.artifacts else None

    @property
    def approved(self) -> bool:
        return bool(self.decisions) and self.decisions[-1].approved
