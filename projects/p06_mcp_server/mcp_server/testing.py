"""Helpers for driving an in-process server (used by tests, demo and bench)."""
from __future__ import annotations

from typing import Any

from .server import LATEST_PROTOCOL_VERSION, MCPServer, Session

DEV_TOKENS: dict[str, dict[str, Any]] = {
    "admin-token-0001": {"client_id": "admin", "scopes": ["*"]},
    "reader-token-001": {"client_id": "reader", "scopes": ["fs:read"]},
    "writer-token-001": {"client_id": "writer", "scopes": ["fs:*"]},
    "exec-token-0001": {"client_id": "executor", "scopes": ["exec:python"]},
}


def rpc(server: MCPServer, session: Session, method: str, params: dict | None = None, rid: Any = 1) -> dict:
    """Send one request in-process and return the response object."""
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        msg["params"] = params
    out = server.handle(msg, session)
    if not isinstance(out, dict):
        raise AssertionError(f"expected a response object, got {out!r}")
    return out


def handshake(server: MCPServer, session: Session, protocol_version: str = LATEST_PROTOCOL_VERSION) -> dict:
    """initialize + notifications/initialized."""
    resp = rpc(server, session, "initialize", {
        "protocolVersion": protocol_version, "capabilities": {}, "clientInfo": {"name": "inproc", "version": "1"},
    })
    server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, session)
    return resp


def connect(server: MCPServer, token: str | None) -> Session:
    """Authenticate ``token`` and return a ready session."""
    principal = server.authenticator.authenticate(token) if token else None
    session = server.new_session(principal)
    handshake(server, session)
    return session
