"""End-to-end demo of the raw MCP server (runs fully offline).

    AIKIT_OFFLINE=1 .venv/bin/python projects/p06_mcp_server/demo.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp_server import MCPHttpServer, build_server  # noqa: E402
from mcp_server.client import HttpClient, StdioClient  # noqa: E402
from mcp_server.testing import DEV_TOKENS  # noqa: E402


def show(label: str, obj: object, width: int = 160) -> None:
    text = json.dumps(obj, separators=(",", ":"))
    print(f"  {label:<34} {text[:width]}{'…' if len(text) > width else ''}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "root"
        root.mkdir()
        (root / "notes.txt").write_text("MCP is JSON-RPC 2.0 with a lifecycle.\n")
        (Path(tmp) / "outside.txt").write_text("you should never see this\n")
        (root / "sneaky").symlink_to(Path(tmp) / "outside.txt")

        print("=== 1. stdio transport: spawn `python -m mcp_server`, full handshake (admin token) ===")
        with StdioClient(root, token="admin-token-0001", tokens=DEV_TOKENS) as c:
            show("tools/list before initialize", c.request("tools/list")["error"])
            init = c.initialize()["result"]
            show("initialize ->", {k: init[k] for k in ("protocolVersion", "serverInfo")})
            tools = c.request("tools/list")["result"]["tools"]
            print(f"  tools visible ({len(tools)}): {[t['name'] for t in tools]}")
            show("add inputSchema", next(t for t in tools if t["name"] == "add")["inputSchema"])
            show("tools/call add(2, 40)", c.call_tool("add", {"a": 2, "b": 40})["result"])
            show("tools/call fs_read notes.txt", c.call_tool("fs_read", {"path": "notes.txt"})["result"]["structuredContent"])
            show("add with a='2' (strict schema)", c.call_tool("add", {"a": "2", "b": 1})["error"])
            show("add with extra field", c.call_tool("add", {"a": 1, "b": 1, "c": 1})["error"]["data"])
            show("fs_read ../outside.txt", c.call_tool("fs_read", {"path": "../outside.txt"})["result"])
            show("fs_read sneaky (symlink out)", c.call_tool("fs_read", {"path": "sneaky"})["result"])
            r = c.call_tool("python_exec", {"code": "import statistics\nprint(statistics.mean([3, 4, 5]))"})["result"]["structuredContent"]
            show("python_exec mean([3,4,5])", {k: r[k] for k in ("exit_code", "stdout", "duration_ms")})
            for label, code in [
                ("python_exec open('/etc/passwd')", "print(open('/etc/passwd').read())"),
                ("python_exec import socket", "import socket"),
                ("python_exec os.system", "import os; os.system('id')"),
                ("python_exec infinite loop", "while True: pass"),
            ]:
                r = c.call_tool("python_exec", {"code": code, "timeout_s": 3})["result"]["structuredContent"]
                show(label, {"exit_code": r["exit_code"], "stderr": r["stderr"].strip().splitlines()[-1]})
            show("batch [ping, bad, echo]", c.send_raw(json.dumps([
                {"jsonrpc": "2.0", "id": "a", "method": "ping"},
                {"jsonrpc": "2.0", "id": "b", "method": "no/such"},
                {"jsonrpc": "2.0", "id": "c", "method": "tools/call", "params": {"name": "echo", "arguments": {"text": "hi"}}},
            ]).encode()))
            show("malformed JSON line", c.send_raw(b'{"jsonrpc": "2.0", "id": '))

        print("\n=== 2. permission scoping: reader token (fs:read only) over stdio ===")
        with StdioClient(root, token="reader-token-001", tokens=DEV_TOKENS) as c:
            c.initialize()
            print(f"  tools visible: {[t['name'] for t in c.request('tools/list')['result']['tools']]}")
            show("python_exec (needs exec:python)", c.call_tool("python_exec", {"code": "print(1)"})["error"])
            show("fs_write (needs fs:write)", c.call_tool("fs_write", {"path": "x", "content": "y"})["error"])
            show("resources/list", [r["uri"] for r in c.request("resources/list")["result"]["resources"]])

        print("\n=== 3. Streamable-HTTP-style transport ===")
        server = build_server(root, tokens=DEV_TOKENS)
        with MCPHttpServer(server) as http:
            print(f"  listening on {http.url}")
            anon = HttpClient(http.url, None)
            anon.initialize()
            print(f"  no bearer token -> HTTP {anon.last_status}")
            c = HttpClient(http.url, "writer-token-001")
            c.initialize()
            print(f"  initialize -> Mcp-Session-Id {c.session_id[:10]}…; initialized notification -> HTTP {c.last_status}")
            show("fs_write report.md", c.call_tool("fs_write", {"path": "out/report.md", "content": "# hi"})["result"]["structuredContent"])
            show("prompts/get code_review", c.request("prompts/get", {"name": "code_review", "arguments": {"code": "eval(x)"}})["result"]["messages"][0])
            print(f"  DELETE session -> HTTP {c.delete()}; reuse -> ", end="")
            c.request("ping")
            print(f"HTTP {c.last_status}")

        print("\n=== 4. benchmark ===")
        from mcp_server.bench import benchmark

        for k, v in benchmark(inproc_calls=3000, stdio_calls=500).items():
            print(f"  {k:<36} {v}")


if __name__ == "__main__":
    main()
