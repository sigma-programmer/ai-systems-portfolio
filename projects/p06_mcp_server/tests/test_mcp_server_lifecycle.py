"""MCP lifecycle: initialize, version negotiation, gating, ping."""
from __future__ import annotations

from mcp_server import SERVER_NOT_INITIALIZED, build_server
from mcp_server.jsonrpc import INVALID_PARAMS, INVALID_REQUEST

from mcp_server.testing import DEV_TOKENS as TOKENS, handshake, rpc

INIT = {"protocolVersion": "2025-06-18", "capabilities": {"roots": {}}, "clientInfo": {"name": "c", "version": "0.1"}}


def test_initialize_result_shape(server):
    s = server.new_session()
    r = rpc(server, s, "initialize", INIT)["result"]
    assert r["protocolVersion"] == "2025-06-18"
    assert set(r["capabilities"]) == {"tools", "resources", "prompts"}
    assert r["serverInfo"]["name"] == "raw-mcp-server"
    assert s.client_info == {"name": "c", "version": "0.1"}
    assert s.client_capabilities == {"roots": {}}
    assert s.state == "initializing"


def test_older_supported_version_is_echoed(server):
    s = server.new_session()
    r = rpc(server, s, "initialize", {**INIT, "protocolVersion": "2025-03-26"})["result"]
    assert r["protocolVersion"] == "2025-03-26" and s.protocol_version == "2025-03-26"


def test_unknown_version_negotiates_to_latest(server):
    s = server.new_session()
    assert rpc(server, s, "initialize", {**INIT, "protocolVersion": "1999-01-01"})["result"]["protocolVersion"] == "2025-06-18"


def test_initialize_param_validation(server):
    s = server.new_session()
    assert rpc(server, s, "initialize", {"protocolVersion": "2025-06-18"})["error"]["code"] == INVALID_PARAMS
    assert rpc(server, s, "initialize", {**INIT, "clientInfo": {"name": "x"}})["error"]["code"] == INVALID_PARAMS
    assert s.state == "new"  # failed initialize does not advance the state


def test_requests_before_initialize_are_rejected(server):
    s = server.new_session()
    assert rpc(server, s, "tools/list")["error"]["code"] == SERVER_NOT_INITIALIZED
    assert rpc(server, s, "ping")["result"] == {}  # ping is always allowed


def test_strict_lifecycle_requires_initialized_notification(server):
    s = server.new_session()
    rpc(server, s, "initialize", INIT)
    assert rpc(server, s, "tools/list")["error"]["code"] == SERVER_NOT_INITIALIZED
    server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, s)
    assert "tools" in rpc(server, s, "tools/list")["result"]


def test_lenient_lifecycle(root):
    srv = build_server(root, tokens=TOKENS, strict_lifecycle=False)
    s = srv.new_session()
    rpc(srv, s, "initialize", INIT)
    assert "tools" in rpc(srv, s, "tools/list")["result"]


def test_double_initialize_rejected(server):
    s = server.new_session()
    handshake(server, s)
    assert rpc(server, s, "initialize", INIT)["error"]["code"] == INVALID_REQUEST


def test_unknown_method_after_init(server, admin):
    assert rpc(server, admin, "tools/frobnicate")["error"]["code"] == -32601


def test_closed_session_rejects(server, admin):
    server.close_session(admin.id)
    assert rpc(server, admin, "ping")["error"]["code"] == INVALID_REQUEST
    assert server.get_session(admin.id) is None


def test_cancel_notification_recorded(server, admin):
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 9}}, admin) is None
    assert 9 in admin.cancelled


def test_handle_raw_roundtrip(server, admin):
    out = server.handle_raw('{"jsonrpc":"2.0","id":"p","method":"ping"}', admin)
    assert out == '{"jsonrpc":"2.0","id":"p","result":{}}'
