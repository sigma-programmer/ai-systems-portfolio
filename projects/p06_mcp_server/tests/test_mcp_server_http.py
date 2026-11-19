"""Streamable-HTTP-style transport."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from mcp_server import MCPHttpServer
from mcp_server.client import HttpClient


@pytest.fixture
def http(server):
    with MCPHttpServer(server) as h:
        yield h


def test_session_lifecycle(http, server, root):
    c = HttpClient(http.url, "admin-token-0001")
    init = c.initialize()
    assert init["result"]["protocolVersion"] == "2025-06-18" and c.session_id
    assert c.last_status == 202  # the initialized notification
    assert len(c.request("tools/list")["result"]["tools"]) == 7
    assert c.call_tool("fs_write", {"path": "h.txt", "content": "via http"})["result"]["isError"] is False
    assert (root / "h.txt").read_text() == "via http"
    assert c.delete() == 204
    assert c.request("ping") is not None and c.last_status == 404  # session gone


def test_scoped_listing_over_http(http):
    c = HttpClient(http.url, "reader-token-001")
    c.initialize()
    assert [t["name"] for t in c.request("tools/list")["result"]["tools"]] == ["echo", "add", "fs_read", "fs_list"]
    assert c.call_tool("python_exec", {"code": "print(1)"})["error"]["code"] == -32003


def test_auth_required(http):
    for token in (None, "bad-token-xxxx"):
        c = HttpClient(http.url, token)
        body = c.initialize()
        assert c.last_status == 401 and body["error"]["code"] == -32600


def test_missing_and_unknown_session(http):
    c = HttpClient(http.url, "admin-token-0001")
    assert c.request("ping") and c.last_status == 400  # no session header, not initialize
    c.session_id = "forged"
    c.request("ping")
    assert c.last_status == 404


def test_session_bound_to_client(http):
    a = HttpClient(http.url, "admin-token-0001")
    a.initialize()
    thief = HttpClient(http.url, "reader-token-001")
    thief.session_id = a.session_id
    thief.request("tools/list")
    assert thief.last_status == 403


def test_parse_error_and_batch(http):
    c = HttpClient(http.url, "admin-token-0001")
    c.initialize()
    status, _, body = c.post(b"{oops")
    assert status == 400 and body["error"]["code"] == -32700
    status, _, body = c.post(json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"}, {"jsonrpc": "2.0", "id": 2, "method": "x"}]).encode())
    assert status == 200 and [r.get("result", r.get("error", {}).get("code")) for r in body] == [{}, -32601]


def test_header_validation(http):
    c = HttpClient(http.url, "admin-token-0001")
    c.initialize()
    assert c.post(b"{}", {"Content-Type": "text/plain"})[0] == 415
    assert c.post(b"{}", {"Origin": "http://evil.example"})[0] == 403
    assert c.post(b"{}", {"MCP-Protocol-Version": "1900-01-01"})[0] == 400
    status, _, body = c.post(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode(), {"Origin": "http://localhost", "MCP-Protocol-Version": "2025-06-18"})
    assert status == 200 and body["result"] == {}


def test_body_size_limit(server):
    with MCPHttpServer(server, max_body=100) as h:
        c = HttpClient(h.url, "admin-token-0001")
        status, _, _ = c.post(b"[" + b" " * 200 + b"]")
        assert status == 413


def test_failed_initialize_does_not_create_session(http, server):
    c = HttpClient(http.url, "admin-token-0001")
    body = c.request("initialize", {"protocolVersion": "2025-06-18"})
    assert body["error"]["code"] == -32602 and c.session_id is None and server.session_count() == 0


def test_get_and_wrong_path(http):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(http.url, timeout=5)
    assert e.value.code == 405
    c = HttpClient(http.url.replace("/mcp", "/other"), "admin-token-0001")
    c.initialize()
    assert c.last_status == 404
