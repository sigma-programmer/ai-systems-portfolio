"""Transports: stdio (newline-delimited JSON) and Streamable-HTTP-style POST.

stdio
    One JSON-RPC message (or batch) per line on stdin; replies on stdout, one
    per line; all logging goes to stderr so stdout stays a clean protocol
    channel. Oversized lines are drained and answered with -32600.

HTTP (subset of MCP "Streamable HTTP")
    ``POST /mcp`` carries a JSON-RPC message; the reply is ``application/json``
    (no SSE streaming). ``Authorization: Bearer <token>`` is required. The
    response to ``initialize`` carries ``Mcp-Session-Id``; later requests must
    echo it (unknown -> 404, missing -> 400). Notification-only bodies get
    ``202 Accepted``. ``DELETE /mcp`` ends a session. ``Origin`` is validated
    against an allowlist (DNS-rebinding defense) and the server binds to
    loopback by default.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import BinaryIO, Iterable

from .auth import Principal
from .jsonrpc import INVALID_REQUEST, PARSE_ERROR, make_error, parse_json
from .server import SUPPORTED_PROTOCOL_VERSIONS, MCPServer

log = logging.getLogger("mcp_server.transport")

MAX_MESSAGE_BYTES = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# stdio
# ---------------------------------------------------------------------------
def serve_stdio(
    server: MCPServer,
    principal: Principal | None = None,
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
    max_line_bytes: int = MAX_MESSAGE_BYTES,
) -> None:
    """Serve one session over byte streams until EOF."""
    rx = stdin or sys.stdin.buffer
    tx = stdout or sys.stdout.buffer
    session = server.new_session(principal)
    log.info("stdio session %s started for client %s", session.id[:8], session.principal.client_id)

    def send(text: str) -> None:
        tx.write(text.encode("utf-8") + b"\n")
        tx.flush()

    try:
        while True:
            line = rx.readline(max_line_bytes + 1)
            if not line:
                break
            if len(line) > max_line_bytes and not line.endswith(b"\n"):
                while True:  # drain the rest of the oversized line
                    chunk = rx.readline(65536)
                    if not chunk or chunk.endswith(b"\n"):
                        break
                send(json.dumps(make_error(None, INVALID_REQUEST, "Invalid Request: message too large")))
                continue
            line = line.strip()
            if not line:
                continue
            reply = server.handle_raw(line, session)
            if reply is not None:
                send(reply)
    except BrokenPipeError:
        log.info("stdout closed by client")
    finally:
        server.close_session(session.id)
        log.info("stdio session %s ended", session.id[:8])


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    server_version = "raw-mcp-http/1.0"
    protocol_version = "HTTP/1.1"
    mcp: MCPServer  # injected via subclassing in MCPHttpServer
    endpoint: str
    allowed_origins: frozenset[str]
    max_body: int

    def log_message(self, fmt: str, *args: object) -> None:  # route to logging (stderr)
        log.debug("%s - %s", self.address_string(), fmt % args)

    # -- helpers ------------------------------------------------------------
    def _send(self, status: int, body: object | None = None, headers: dict[str, str] | None = None) -> None:
        payload = b"" if body is None else json.dumps(body).encode("utf-8")
        self.send_response(status)
        if body is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def _rpc_error(self, status: int, code: int, message: str, headers: dict[str, str] | None = None) -> None:
        self._send(status, make_error(None, code, message), headers)

    def _precheck(self) -> Principal | None:
        """Path, Origin and auth checks shared by all verbs. Sends the error itself."""
        if self.path.split("?", 1)[0] != self.endpoint:
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return None
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.allowed_origins:
            self._rpc_error(HTTPStatus.FORBIDDEN, INVALID_REQUEST, "Origin not allowed")
            return None
        auth = self.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else None
        principal = self.mcp.authenticator.authenticate(token)
        if principal is None:
            self._rpc_error(
                HTTPStatus.UNAUTHORIZED, INVALID_REQUEST, "Unauthorized: missing or invalid bearer token",
                {"WWW-Authenticate": 'Bearer realm="mcp"'},
            )
            return None
        version = self.headers.get("MCP-Protocol-Version")
        if version is not None and version not in SUPPORTED_PROTOCOL_VERSIONS:
            self._rpc_error(HTTPStatus.BAD_REQUEST, INVALID_REQUEST, f"Unsupported MCP-Protocol-Version {version}")
            return None
        return principal

    # -- verbs --------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        principal = self._precheck()
        if principal is None:
            return
        ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self._rpc_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, INVALID_REQUEST, "Content-Type must be application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._rpc_error(HTTPStatus.LENGTH_REQUIRED, INVALID_REQUEST, "Content-Length required")
            return
        if length < 0 or length > self.max_body:
            self._rpc_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, INVALID_REQUEST, "Request body too large")
            return
        raw = self.rfile.read(length)
        try:
            message = parse_json(raw)
        except (ValueError, RecursionError):
            self._rpc_error(HTTPStatus.BAD_REQUEST, PARSE_ERROR, "Parse error")
            return

        session_id = self.headers.get("Mcp-Session-Id")
        is_init = isinstance(message, dict) and message.get("method") == "initialize"
        new_header: dict[str, str] = {}
        if session_id is None:
            if not is_init:
                self._rpc_error(HTTPStatus.BAD_REQUEST, INVALID_REQUEST, "Missing Mcp-Session-Id header (send initialize first)")
                return
            session = self.mcp.new_session(principal)
            new_header = {"Mcp-Session-Id": session.id}
        else:
            session = self.mcp.get_session(session_id)
            if session is None:
                self._rpc_error(HTTPStatus.NOT_FOUND, INVALID_REQUEST, "Unknown or expired session")
                return
            if session.principal.client_id != principal.client_id:
                self._rpc_error(HTTPStatus.FORBIDDEN, INVALID_REQUEST, "Session belongs to a different client")
                return

        reply = self.mcp.handle(message, session)
        if new_header and isinstance(reply, dict) and "error" in reply:
            self.mcp.close_session(session.id)  # failed initialize: don't leak a session
            new_header = {}
        if reply is None:
            self._send(HTTPStatus.ACCEPTED, None, new_header)
        else:
            self._send(HTTPStatus.OK, reply, new_header)

    def do_DELETE(self) -> None:  # noqa: N802
        principal = self._precheck()
        if principal is None:
            return
        sid = self.headers.get("Mcp-Session-Id")
        session = self.mcp.get_session(sid) if sid else None
        if session is None or session.principal.client_id != principal.client_id:
            self._send(HTTPStatus.NOT_FOUND, {"error": "unknown session"})
            return
        self.mcp.close_session(session.id)
        self._send(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:  # noqa: N802
        # Server-initiated SSE streams are not offered by this server.
        self._send(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "GET streams not supported"}, {"Allow": "POST, DELETE"})


class MCPHttpServer:
    """Threaded HTTP transport around an :class:`MCPServer`."""

    def __init__(
        self,
        server: MCPServer,
        host: str = "127.0.0.1",
        port: int = 0,
        endpoint: str = "/mcp",
        allowed_origins: Iterable[str] = ("http://localhost", "http://127.0.0.1"),
        max_body: int = MAX_MESSAGE_BYTES,
    ):
        handler = type(
            "BoundMCPHandler",
            (_Handler,),
            {"mcp": server, "endpoint": endpoint, "allowed_origins": frozenset(allowed_origins), "max_body": max_body},
        )
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.endpoint = endpoint
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}{self.endpoint}"

    def start(self) -> "MCPHttpServer":
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="mcp-http", daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> "MCPHttpServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
