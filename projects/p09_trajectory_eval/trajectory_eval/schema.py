"""Trajectory schema: the typed record of what an agent actually did.

A trajectory is an ordered list of steps. Each step is a tool call, an LLM
generation, or an observation (tool result fed back to the model). Steps
carry cost/latency telemetry so the grader can reason about efficiency as
well as correctness.
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class StepType(str, Enum):
    """Kind of step recorded in a trajectory."""

    TOOL_CALL = "tool_call"
    LLM = "llm"
    OBSERVATION = "observation"


class Step(BaseModel):
    """A single agent step."""

    type: StepType
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    content: str = ""
    tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: bool = False

    @model_validator(mode="after")
    def _tool_required_for_calls(self) -> "Step":
        if self.type == StepType.TOOL_CALL and not self.tool:
            raise ValueError("tool_call steps must name a tool")
        return self

    def signature(self) -> str:
        """Canonical identity of the step (tool + sorted args) used for loop detection."""
        if self.type == StepType.TOOL_CALL:
            return f"{self.tool}({json.dumps(self.args, sort_keys=True, default=str)})"
        if self.type == StepType.LLM:
            return f"llm:{self.content[:80]}"
        return f"obs:{self.tool or ''}"


class Trajectory(BaseModel):
    """One agent run on one task."""

    run_id: str
    task_id: str
    task: str = ""
    steps: list[Step] = Field(default_factory=list)
    final_answer: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.steps)

    @property
    def total_latency_ms(self) -> float:
        return float(sum(s.latency_ms for s in self.steps))

    def tool_calls(self) -> list[tuple[int, Step]]:
        return [(i, s) for i, s in enumerate(self.steps) if s.type == StepType.TOOL_CALL]


class TrajectorySuite(BaseModel):
    """A named collection of runs (e.g. a nightly baseline or a PR candidate)."""

    name: str
    trajectories: list[Trajectory]

    @field_validator("trajectories")
    @classmethod
    def _unique_run_ids(cls, v: list[Trajectory]) -> list[Trajectory]:
        ids = [t.run_id for t in v]
        if len(ids) != len(set(ids)):
            raise ValueError("run_id values must be unique within a suite")
        return v

    @classmethod
    def load(cls, path: str | Path) -> "TrajectorySuite":
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=1))

    def task_ids(self) -> list[str]:
        return sorted({t.task_id for t in self.trajectories})
