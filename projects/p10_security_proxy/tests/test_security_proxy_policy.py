"""Least-privilege policy engine and capability attenuation."""
from __future__ import annotations

import pytest

from security_proxy import ArgConstraint, AttenuationError, Capability, Policy


def make_policy() -> Policy:
    return Policy.from_dict(
        {
            "max_delegation_depth": 2,
            "agents": {
                "lead": {
                    "can_message": ["research-*", "writer"],
                    "pii_unmask": ["EMAIL", "PHONE"],
                    "trusted": True,
                    "capabilities": [
                        {"tool": "read_file", "constraints": {"path": {"prefix": "/data/", "required": True}}},
                        {
                            "tool": "web_search",
                            "constraints": {"query": {"max_length": 200}, "k": {"min_value": 1, "max_value": 10}},
                            "max_calls_per_message": 3,
                        },
                        {"tool": "send_email", "constraints": {"to": {"patterns": [r"[^@]+@corp\.io"]}, "body": {}}},
                        {"tool": "set_mode", "constraints": {"mode": {"allowed": ["fast", "safe"]}}},
                    ],
                },
                "research-1": {"can_message": ["lead"]},
                "writer": {"can_message": ["lead"]},
            },
        }
    )


def test_deny_by_default_unknown_agent_and_tool() -> None:
    p = make_policy()
    assert not p.check_tool_call("ghost", "read_file", {"path": "/data/x"}).allowed
    d = p.check_tool_call("lead", "shell", {})
    assert not d.allowed and d.rule == "deny_default"
    assert not p.check_tool_call("writer", "read_file", {"path": "/data/x"}).allowed


def test_messaging_globs() -> None:
    p = make_policy()
    assert p.check_message("lead", "research-1").allowed
    assert p.check_message("lead", "writer").allowed
    assert not p.check_message("writer", "research-1").allowed
    assert not p.check_message("lead", "nobody").allowed
    assert not p.check_message("ghost", "lead").allowed


@pytest.mark.parametrize(
    "tool,args,allowed",
    [
        ("read_file", {"path": "/data/report.txt"}, True),
        ("read_file", {"path": "/etc/passwd"}, False),
        ("read_file", {"path": "/data/../etc/passwd"}, False),
        ("read_file", {}, False),
        ("read_file", {"path": "/data/a", "mode": "w"}, False),  # extra arg
        ("web_search", {"query": "llm security", "k": 5}, True),
        ("web_search", {"query": "x" * 201}, False),
        ("web_search", {"query": "ok", "k": 0}, False),
        ("web_search", {"query": "ok", "k": True}, False),
        ("web_search", {"query": 5}, False),
        ("send_email", {"to": "ann@corp.io", "body": "hi"}, True),
        ("send_email", {"to": "ann@evil.io", "body": "hi"}, False),
        ("set_mode", {"mode": "fast"}, True),
        ("set_mode", {"mode": "god"}, False),
    ],
)
def test_argument_constraints(tool: str, args: dict, allowed: bool) -> None:
    assert make_policy().check_tool_call("lead", tool, args).allowed is allowed


def test_allow_extra_args() -> None:
    p = Policy.from_dict({"agents": {"a": {"capabilities": [{"tool": "t", "allow_extra_args": True}]}}})
    assert p.check_tool_call("a", "t", {"anything": 1}).allowed


def test_bad_regex_rejected() -> None:
    with pytest.raises(ValueError):
        ArgConstraint(patterns=["("])


def test_roundtrip_dict_and_json() -> None:
    import json

    p = make_policy()
    p2 = Policy.from_json(json.dumps(p.to_dict()))
    assert p2.to_dict() == p.to_dict()
    assert p2.unmask_types("lead") == {"EMAIL", "PHONE"}
    assert p2.unmask_types("ghost") == set()


# ------------------------------------------------------------------ delegation


def test_delegation_narrows_and_intersects() -> None:
    p = make_policy()
    child = p.delegate(
        "lead",
        "research-sub",
        capabilities=[{"tool": "web_search", "constraints": {"k": {"min_value": 1, "max_value": 3}}, "max_calls_per_message": 1}],
        can_message=["research-2"],
        pii_unmask=["EMAIL"],
    )
    assert child.parent == "lead" and child.depth == 1 and not child.trusted
    cap = child.capability("web_search")
    assert cap.constraints["k"].max_value == 3
    assert cap.constraints["query"].max_length == 200  # inherited parent constraint
    assert p.check_tool_call("research-sub", "web_search", {"query": "a", "k": 3}).allowed
    assert not p.check_tool_call("research-sub", "web_search", {"query": "a", "k": 5}).allowed
    assert not p.check_tool_call("research-sub", "read_file", {"path": "/data/x"}).allowed
    assert p.unmask_types("research-sub") == {"EMAIL"}


def test_delegation_inherits_nothing_by_default() -> None:
    p = make_policy()
    child = p.delegate("lead", "bare")
    assert child.capabilities == [] and child.can_message == [] and child.pii_unmask == []


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"capabilities": [{"tool": "shell"}]}, "lacks tool"),
        ({"capabilities": [{"tool": "web_search", "constraints": {"k": {"max_value": 50}}, "max_calls_per_message": 1}]}, "max_value"),
        ({"capabilities": [{"tool": "web_search", "max_calls_per_message": 9}]}, "max_calls"),
        ({"capabilities": [{"tool": "web_search"}]}, "max_calls"),
        ({"capabilities": [{"tool": "read_file", "constraints": {"path": {"prefix": "/"}}}]}, "prefix"),
        ({"capabilities": [{"tool": "read_file", "constraints": {"path": {"prefix": "/data/x"}}}]}, "required"),
        ({"capabilities": [{"tool": "read_file", "allow_extra_args": True}]}, "allow_extra_args"),
        ({"capabilities": [{"tool": "read_file", "constraints": {"mode": {}}}]}, "not permitted"),
        ({"capabilities": [{"tool": "set_mode", "constraints": {"mode": {"allowed": ["fast", "god"]}}}]}, "subset"),
        ({"can_message": ["*"]}, "messaging"),
        ({"can_message": ["admin"]}, "messaging"),
        ({"pii_unmask": ["CREDIT_CARD"]}, "PII"),
        ({"pii_unmask": ["*"]}, "PII"),
    ],
)
def test_delegation_cannot_broaden(kwargs: dict, msg: str) -> None:
    p = make_policy()
    with pytest.raises(AttenuationError, match=msg):
        p.delegate("lead", "child", **kwargs)
    assert p.get("child") is None


def test_delegation_depth_and_duplicates() -> None:
    p = make_policy()
    p.delegate("lead", "c1", can_message=["writer"])
    p.delegate("c1", "c2", can_message=["writer"])
    with pytest.raises(AttenuationError, match="depth"):
        p.delegate("c2", "c3")
    with pytest.raises(AttenuationError, match="exists"):
        p.delegate("lead", "c1")
    with pytest.raises(AttenuationError, match="unknown parent"):
        p.delegate("ghost", "x")


def test_capability_object_accepted() -> None:
    p = make_policy()
    child = p.delegate("lead", "c", capabilities=[Capability(tool="set_mode", constraints={"mode": ArgConstraint(allowed=["safe"])})])
    assert p.check_tool_call("c", "set_mode", {"mode": "safe"}).allowed
    assert not p.check_tool_call("c", "set_mode", {"mode": "fast"}).allowed
    assert child.capability("set_mode") is not None


def test_constraint_intersect() -> None:
    a = ArgConstraint(allowed=[1, 2, 3], min_value=0, max_length=10, prefix="/a")
    b = ArgConstraint(allowed=[2, 3, 4], min_value=2, max_length=5, prefix="/a/b", required=True)
    c = a.intersect(b)
    assert c.allowed == [2, 3] and c.min_value == 2 and c.max_length == 5 and c.prefix == "/a/b" and c.required
