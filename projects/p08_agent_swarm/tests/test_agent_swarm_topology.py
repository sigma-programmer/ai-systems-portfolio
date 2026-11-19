"""Typed topology validation."""
from __future__ import annotations

import pytest

from agent_swarm import (
    Coder, ConsensusPanel, ExecutionReviewer, Researcher, RevisionRequest, Role, Topology, TopologyError,
    build_topology, review_topology, single_agent_topology,
)
from agent_swarm.messages import CodeArtifact


def test_valid_topologies_build():
    t = review_topology(Researcher(), Coder(), [ExecutionReviewer("e1"), ExecutionReviewer("e2")], ConsensusPanel())
    assert t.validated and t.order()[0] == "researcher" and t.order()[-1] == "panel"
    assert t.feedback_from("panel") == ("coder", RevisionRequest)
    assert t.preds("panel") == ["e1", "e2"]
    assert t.nodes_with_role(Role.REVIEWER) == ["e1", "e2"]
    assert "~~>" in t.describe()
    assert single_agent_topology(Coder()).entries() == ["coder"]
    for mode in ("single_agent", "swarm_review", "swarm_consensus"):
        assert build_topology(mode).validated
    with pytest.raises(ValueError):
        build_topology("nope")


def test_edge_type_mismatch_rejected():
    t = Topology().add(Researcher(), Coder(), ConsensusPanel())
    t.connect("researcher", "coder").connect("coder", "panel")
    with pytest.raises(TopologyError, match="CodeArtifact is not accepted by panel"):
        t.validate()


def test_explicit_payload_not_produced_rejected():
    t = Topology().add(Researcher(), Coder())
    t.connect("researcher", "coder", payload=CodeArtifact)
    with pytest.raises(TopologyError, match="cannot produce CodeArtifact"):
        t.validate()


def test_forward_cycle_rejected():
    t = Topology().add(Coder("c1"), Coder("c2"), ExecutionReviewer("r"))
    t.add(ConsensusPanel("p"))
    t.connect("c1", "r").connect("r", "p")
    t.g.add_edge("p", "c1", payload=RevisionRequest, feedback=False)  # forward loop, not a declared feedback edge
    with pytest.raises(TopologyError, match="cycle"):
        t.validate()


def test_entry_must_accept_task_spec():
    t = Topology().add(ExecutionReviewer("r"))
    with pytest.raises(TopologyError, match="does not accept TaskSpec"):
        t.validate()


def test_feedback_must_point_upstream():
    t = Topology().add(Researcher(), Coder(), ExecutionReviewer("r"), ConsensusPanel("p"), Coder("other"))
    t.connect("researcher", "coder").connect("coder", "r").connect("r", "p")
    t.connect("researcher", "other")
    t.connect("p", "other", feedback=True)
    with pytest.raises(TopologyError, match="upstream"):
        t.validate()


def test_feedback_without_compatible_type_rejected():
    t = Topology().add(Researcher(), Coder())
    with pytest.raises(TopologyError, match="no output type"):
        t.connect("coder", "researcher", feedback=True)


def test_duplicate_and_unknown_names():
    t = Topology().add(Coder())
    with pytest.raises(TopologyError, match="duplicate"):
        t.add(Coder())
    with pytest.raises(TopologyError, match="unknown"):
        t.connect("coder", "ghost")
    with pytest.raises(TopologyError, match="no agents"):
        Topology().validate()
