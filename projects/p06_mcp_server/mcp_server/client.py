"""Minimal MCP clients (stdio subprocess and HTTP) for tests, demos and benchmarks."""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .server import LATEST_PROTOCOL_VERSION

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLIENT_INFO = {"name": "raw-mcp-test-client", "version": "1.0.0"}


class _Base:
    def __init__(self) -> None:
        self._ids = itertools.count(1)

    def _send(self, message: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def request(self, method: str, params: dict | None = None) -> dict:
        """Send a request and return the full response object."""
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": next(self._ids), "method": method}
        if params is not None:
            msg["params"] = params
        return self._send(msg)

    def notify(self, method: str, params: dict | None = None) -> Any:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        return self._send(msg)

    def initialize(self, protocol_version: str = LATEST_PROTOCOL_VERSION) -> dict:
        """Full handshake: initialize + notifications/initialized."""
        resp = self.request(
            "initialize",
            {"protocolVersion": protocol_version, "capabilities": {}, "clientInfo": CLIENT_INFO},
        )
        if "result" in resp:
            self.notify("notifications/initialized")
        return resp

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments or {}})


class StdioClient(_Base):
    """Spawns ``python -m mcp_server`` and talks newline-delimited JSON to it."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        token: str | None = None,
        tokens: dict[str, dict] | None = None,
        extra_args: list[str] | None = None,
        python: str = sys.executable,
    ):
        super().__init__()
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": str(PROJECT_DIR),
            "PYTHONUNBUFFERED": "1",
        }
        if tokens is not None:
            env["MCP_TOKENS"] = json.dumps(tokens)
        if token is not None:
            env["MCP_TOKEN"] = token
        self.proc = subprocess.Popen(
            [python, "-m", "mcp_server", "--root", str(root), *(extra_args or [])],
            cwd=str(PROJECT_DIR),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def send_raw(self, line: bytes, expect_reply: bool = True) -> Any:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(line.rstrip(b"\n") + b"\n")
        self.proc.stdin.flush()
        if not expect_reply:
            return None
        out = self.proc.stdout.readline()
        if not out:
            raise ConnectionError("server closed stdout: " + self._stderr_tail())
        return json.loads(out)

    def _send(self, message: Any) -> Any:
        is_notification = isinstance(message, dict) and "id" not in message
        return self.send_raw(json.dumps(message).encode(), expect_reply=not is_notification)

    def _stderr_tail(self) -> str:
        if self.proc.poll() is None:
            return "(server still running)"
        assert self.proc.stderr is not None
        return self.proc.stderr.read().decode(errors="replace")[-2000:]

    def close(self) -> int:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            rc = self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            rc = self.proc.wait()
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream:
                stream.close()
        return rc

    def __enter__(self) -> "StdioClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class HttpClient(_Base):
    """POSTs JSON-RPC to the HTTP transport, tracking ``Mcp-Session-Id``."""

    def __init__(self, url: str, token: str | None):
        super().__init__()
        self.url = url
        self.token = token
        self.session_id: str | None = None
        self.last_status: int | None = None

    def post(self, body: bytes, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], Any]:
        hdrs = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.token:
            hdrs["Authorization"] = f"Bearer {self.token}"
        if self.session_id:
            hdrs["Mcp-Session-Id"] = self.session_id
        hdrs.update(headers or {})
        req = urllib.request.Request(self.url, data=body, headers=hdrs, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                status, rhdrs, data = resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as e:
            status, rhdrs, data = e.code, dict(e.headers), e.read()
        self.last_status = status
        return status, rhdrs, (json.loads(data) if data else None)

    def _send(self, message: Any) -> Any:
        status, headers, body = self.post(json.dumps(message).encode())
        sid = headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid
        return body

    def delete(self) -> int:
        hdrs = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        if self.session_id:
            hdrs["Mcp-Session-Id"] = self.session_id
        req = urllib.request.Request(self.url, headers=hdrs, method="DELETE")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status
        except urllib.error.HTTPError as e:
            return e.code
