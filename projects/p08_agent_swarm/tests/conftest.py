"""Shared fixtures for agent_swarm tests."""
from __future__ import annotations

import pytest

from agent_swarm import Sandbox, SwarmState, TASKS_BY_NAME
from agent_swarm.agents import AgentContext


@pytest.fixture(scope="session")
def sandbox() -> Sandbox:
    return Sandbox()


@pytest.fixture
def make_ctx(sandbox):
    def _make(task_name: str = "palindrome", findings=None) -> AgentContext:
        st = SwarmState(task=TASKS_BY_NAME[task_name].spec, findings=findings)
        return AgentContext(state=st, sandbox=sandbox)
    return _make
