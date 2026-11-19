"""Authentication, scope filtering, permission denial and rate limiting."""
from __future__ import annotations

import pytest

from mcp_server import PERMISSION_DENIED, RATE_LIMITED, RateLimiter, TokenAuthenticator, build_server
from mcp_server.auth import ANONYMOUS, Principal
from mcp_server.testing import DEV_TOKENS, connect, rpc


def names(server, session):
    return [t["name"] for t in rpc(server, session, "tools/list")["result"]["tools"]]


def test_authenticator_lookup():
    auth = TokenAuthenticator.from_config(DEV_TOKENS)
    assert auth.authenticate("admin-token-0001").client_id == "admin"
    assert auth.authenticate("reader-token-001").scopes == frozenset({"fs:read"})
    assert auth.authenticate("wrong-token") is None
    assert auth.authenticate("") is None and auth.authenticate(None) is None
    assert len(auth) == 4


def test_short_tokens_rejected():
    with pytest.raises(ValueError):
        TokenAuthenticator.from_config({"short": {"client_id": "x", "scopes": []}})


def test_scope_matching():
    p = Principal("p", frozenset({"fs:*", "net"}))
    assert p.has_scope("fs:read") and p.has_scope("fs:write") and p.has_scope("net")
    assert not p.has_scope("exec:python")
    assert Principal("root", frozenset({"*"})).has_scope("anything:at_all")
    assert p.missing({"fs:read", "exec:python", "net"}) == ["exec:python"]
    assert ANONYMOUS.missing({"fs:read"}) == ["fs:read"]


def test_tools_list_filtered_by_scope(server, connect):
    assert names(server, connect(None)) == ["echo", "add"]
    assert names(server, connect("reader-token-001")) == ["echo", "add", "fs_read", "fs_list"]
    assert names(server, connect("writer-token-001")) == ["echo", "add", "fs_read", "fs_list", "fs_write"]
    assert names(server, connect("exec-token-0001")) == ["echo", "add", "python_exec"]
    assert len(names(server, connect("admin-token-0001"))) == 7


@pytest.mark.parametrize(
    "token,tool,args,missing",
    [
        (None, "fs_read", {"path": "hello.txt"}, ["fs:read"]),
        ("reader-token-001", "fs_write", {"path": "x.txt", "content": "x"}, ["fs:write"]),
        ("reader-token-001", "python_exec", {"code": "print(1)"}, ["exec:python"]),
        ("exec-token-0001", "http_get", {"url": "https://example.com"}, ["net"]),
    ],
)
def test_denied_calls(server, connect, token, tool, args, missing):
    err = rpc(server, connect(token), "tools/call", {"name": tool, "arguments": args})["error"]
    assert err["code"] == PERMISSION_DENIED
    assert err["data"]["missingScopes"] == missing


def test_permission_checked_before_validation(server, connect):
    # A denied client learns nothing about the argument schema.
    err = rpc(server, connect(None), "tools/call", {"name": "fs_write", "arguments": {"bogus": 1}})["error"]
    assert err["code"] == PERMISSION_DENIED


def test_denied_write_has_no_side_effect(server, connect, root):
    rpc(server, connect("reader-token-001"), "tools/call", {"name": "fs_write", "arguments": {"path": "x.txt", "content": "x"}})
    assert not (root / "x.txt").exists()


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_token_bucket():
    clock = FakeClock()
    rl = RateLimiter(capacity=3, refill_per_sec=1, clock=clock)
    assert [rl.allow("a") for _ in range(4)] == [True, True, True, False]
    assert rl.allow("b")  # independent buckets
    assert rl.retry_after("a") == pytest.approx(1.0)
    clock.t = 1.0
    assert rl.allow("a") and not rl.allow("a")
    clock.t = 100.0
    assert sum(rl.allow("a") for _ in range(10)) == 3  # capped at capacity


def test_rate_limit_enforced_per_client(root):
    clock = FakeClock()
    srv = build_server(root, tokens=DEV_TOKENS, rate_limiter=RateLimiter(capacity=5, refill_per_sec=1, clock=clock))
    a = connect(srv, "admin-token-0001")
    r = connect(srv, "reader-token-001")
    results = [rpc(srv, a, "tools/call", {"name": "echo", "arguments": {"text": "x"}}) for _ in range(7)]
    assert sum("result" in x for x in results) == 5
    limited = results[-1]["error"]
    assert limited["code"] == RATE_LIMITED and limited["data"]["retryAfterSeconds"] > 0
    assert "result" in rpc(srv, r, "tools/list")  # other client unaffected
    assert rpc(srv, a, "ping")["result"] == {}  # ping is unmetered
    clock.t = 2.0
    assert "result" in rpc(srv, a, "tools/list")
