"""End-to-end over the stdio transport: spawn ``python -m mcp_server``."""
from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

from mcp_server.client import PROJECT_DIR, StdioClient
from mcp_server.testing import DEV_TOKENS
from mcp_server.transports import serve_stdio


@pytest.fixture
def client(root):
    c = StdioClient(root, token="admin-token-0001", tokens=DEV_TOKENS)
    yield c
    c.close()


def test_full_handshake_and_tool_calls(client, root):
    assert client.request("tools/list")["error"]["code"] == -32000  # before initialize
    init = client.initialize()
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert client.request("ping")["result"] == {}
    tools = [t["name"] for t in client.request("tools/list")["result"]["tools"]]
    assert tools == ["echo", "add", "fs_read", "fs_list", "fs_write", "python_exec", "http_get"]
    assert client.call_tool("add", {"a": 40, "b": 2})["result"]["structuredContent"] == {"sum": 42.0}
    assert client.call_tool("fs_read", {"path": "hello.txt"})["result"]["structuredContent"]["content"] == "hello world\n"
    assert client.call_tool("fs_read", {"path": "../secret.txt"})["result"]["isError"] is True
    assert client.call_tool("fs_write", {"path": "w.txt", "content": "from stdio"})["result"]["isError"] is False
    assert (root / "w.txt").read_text() == "from stdio"
    py = client.call_tool("python_exec", {"code": "print('sandboxed', 1+1)"})["result"]["structuredContent"]
    assert py["stdout"] == "sandboxed 2\n"
    assert client.call_tool("add", {"a": "x"})["error"]["code"] == -32602


def test_protocol_errors_over_the_wire(client):
    client.initialize()
    assert client.send_raw(b"{not json")["error"]["code"] == -32700
    assert client.send_raw(b'{"jsonrpc":"2.0","id":1}')["error"]["code"] == -32600
    assert client.send_raw(b'{"jsonrpc":"2.0","id":1,"method":"nope"}')["error"]["code"] == -32601
    batch = client.send_raw(json.dumps([
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "echo", "arguments": {"text": "b"}}},
    ]).encode())
    assert [r["id"] for r in batch] == [1, 2]
    assert batch[1]["result"]["content"][0]["text"] == "b"
    client.send_raw(b"   ", expect_reply=False)  # blank lines are ignored
    assert client.request("ping")["result"] == {}


def test_anonymous_stdio_client_sees_public_tools_only(root):
    with StdioClient(root, tokens=DEV_TOKENS) as c:
        c.initialize()
        assert [t["name"] for t in c.request("tools/list")["result"]["tools"]] == ["echo", "add"]
        assert c.call_tool("fs_read", {"path": "hello.txt"})["error"]["code"] == -32003


def test_unknown_token_refuses_to_start(root):
    c = StdioClient(root, token="not-a-real-token", tokens=DEV_TOKENS)
    assert c.close() == 2
    assert c.proc.returncode == 2


def test_stdout_is_protocol_only_and_logs_go_to_stderr(root):
    proc = subprocess.run(
        [sys.executable, "-m", "mcp_server", "--root", str(root), "--log-level", "INFO"],
        input=b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n',
        capture_output=True, cwd=PROJECT_DIR, env={"PYTHONPATH": str(PROJECT_DIR), "PATH": "/usr/bin:/bin"}, timeout=30,
    )
    lines = proc.stdout.decode().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["id"] == 1
    assert b"stdio session" in proc.stderr
    assert proc.returncode == 0


def test_oversized_line_in_process(server):
    rx = io.BytesIO(b"x" * 5000 + b"\n" + b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n')
    tx = io.BytesIO()
    serve_stdio(server, None, rx, tx, max_line_bytes=1000)
    out = [json.loads(line) for line in tx.getvalue().splitlines()]
    assert out[0]["error"]["code"] == -32600 and "too large" in out[0]["error"]["message"]
    assert out[1] == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert server.session_count() == 0  # session closed at EOF
