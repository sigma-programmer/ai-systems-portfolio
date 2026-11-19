"""Benchmark: throughput (in-process and stdio), fuzz rejection correctness, sandbox escapes."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .builtin import build_server
from .client import StdioClient
from .sandbox import FsJail, PythonSandbox, SandboxViolation
from .server import MCPServer, Session
from .testing import DEV_TOKENS, connect

ADD_CALL = {"name": "add", "arguments": {"a": 1.5, "b": 2.5}}


def _inproc_rps(server: MCPServer, session: Session, n: int) -> float:
    msgs = [json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": ADD_CALL}) for i in range(n)]
    t0 = time.perf_counter()
    for m in msgs:
        out = server.handle_raw(m, session)
        assert out is not None and '"isError":false' in out
    return n / (time.perf_counter() - t0)


def _stdio_rps(root: Path, n: int) -> tuple[float, float]:
    """Sequential round-trip rps and pipelined (write-all-then-read) rps."""
    with StdioClient(root, token="admin-token-0001", tokens=DEV_TOKENS) as c:
        c.initialize()
        t0 = time.perf_counter()
        for _ in range(n):
            assert "result" in c.call_tool("add", ADD_CALL["arguments"])
        seq = n / (time.perf_counter() - t0)
        assert c.proc.stdin is not None and c.proc.stdout is not None
        payload = b"".join(
            json.dumps({"jsonrpc": "2.0", "id": 10_000 + i, "method": "tools/call", "params": ADD_CALL}).encode() + b"\n"
            for i in range(n)
        )
        t0 = time.perf_counter()
        c.proc.stdin.write(payload)
        c.proc.stdin.flush()
        for _ in range(n):
            assert c.proc.stdout.readline()
        pipelined = n / (time.perf_counter() - t0)
    return seq, pipelined


def fuzz_cases() -> list[tuple[str, Any, int | None]]:
    """(label, raw message or str, expected error code or None for 'no reply')."""
    ok_params = {"name": "add", "arguments": {"a": 1, "b": 2}}
    cases: list[tuple[str, Any, int | None]] = [
        ("truncated json", '{"jsonrpc":"2.0","id":1', -32700),
        ("empty string", "", -32700),
        ("garbage", "@@@", -32700),
        ("NaN literal", '{"jsonrpc":"2.0","id":1,"method":"ping","params":{"x":NaN}}', -32700),
        ("trailing comma", '{"jsonrpc":"2.0","id":1,"method":"ping",}', -32700),
        ("single quotes", "{'jsonrpc':'2.0'}", -32700),
        ("empty batch", "[]", -32600),
        ("number", "42", -32600),
        ("string", '"x"', -32600),
        ("null", "null", -32600),
        ("bool", "true", -32600),
        ("missing jsonrpc", {"id": 1, "method": "ping"}, -32600),
        ("wrong version", {"jsonrpc": "1.0", "id": 1, "method": "ping"}, -32600),
        ("numeric version", {"jsonrpc": 2.0, "id": 1, "method": "ping"}, -32600),
        ("missing method", {"jsonrpc": "2.0", "id": 1}, -32600),
        ("method int", {"jsonrpc": "2.0", "id": 1, "method": 7}, -32600),
        ("method empty", {"jsonrpc": "2.0", "id": 1, "method": ""}, -32600),
        ("method null", {"jsonrpc": "2.0", "id": 1, "method": None}, -32600),
        ("id null", {"jsonrpc": "2.0", "id": None, "method": "ping"}, -32600),
        ("id bool", {"jsonrpc": "2.0", "id": False, "method": "ping"}, -32600),
        ("id object", {"jsonrpc": "2.0", "id": {}, "method": "ping"}, -32600),
        ("id array", {"jsonrpc": "2.0", "id": [], "method": "ping"}, -32600),
        ("id float", {"jsonrpc": "2.0", "id": 1.5, "method": "ping"}, -32600),
        ("params string", {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": "x"}, -32600),
        ("params number", {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": 3}, -32600),
        ("params null", {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": None}, -32600),
        ("params array", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": [1]}, -32602),
        ("unknown method", {"jsonrpc": "2.0", "id": 1, "method": "tools/explode"}, -32601),
        ("case-variant method", {"jsonrpc": "2.0", "id": 1, "method": "Tools/List"}, -32601),
        ("rpc. reserved", {"jsonrpc": "2.0", "id": 1, "method": "rpc.discover"}, -32601),
        ("call without name", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {}}, -32602),
        ("call name int", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": 3}}, -32602),
        ("unknown tool", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "rm_rf"}}, -32602),
        ("args array", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "add", "arguments": [1, 2]}}, -32602),
        ("args string", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "add", "arguments": "a=1"}}, -32602),
        ("missing arg", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "add", "arguments": {"a": 1}}}, -32602),
        ("extra arg", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "add", "arguments": {"a": 1, "b": 2, "c": 3}}}, -32602),
        ("coercible string", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "add", "arguments": {"a": "1", "b": 2}}}, -32602),
        ("bool for number", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "add", "arguments": {"a": True, "b": 2}}}, -32602),
        ("oversized text", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "echo", "arguments": {"text": "x" * 20000}}}, -32602),
        ("negative max_bytes", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "fs_read", "arguments": {"path": "a", "max_bytes": -1}}}, -32602),
        ("timeout too high", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "python_exec", "arguments": {"code": "1", "timeout_s": 1e9}}}, -32602),
        ("bad cursor", {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"cursor": "%%%"}}, -32602),
        ("cursor number", {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"cursor": 9}}, -32602),
        ("re-initialize", {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "x", "version": "1"}}}, -32600),
        ("resource missing", {"jsonrpc": "2.0", "id": 1, "method": "resources/read", "params": {"uri": "mcp://none"}}, -32002),
        ("resource no uri", {"jsonrpc": "2.0", "id": 1, "method": "resources/read", "params": {}}, -32602),
        ("prompt missing arg", {"jsonrpc": "2.0", "id": 1, "method": "prompts/get", "params": {"name": "code_review"}}, -32602),
        ("prompt unknown", {"jsonrpc": "2.0", "id": 1, "method": "prompts/get", "params": {"name": "zzz"}}, -32602),
        ("notification unknown", {"jsonrpc": "2.0", "method": "nope"}, None),
        ("notification bad params", {"jsonrpc": "2.0", "method": "tools/call", "params": [1]}, None),
        ("response object", {"jsonrpc": "2.0", "id": 5, "result": {}}, None),
        ("all-notification batch", [{"jsonrpc": "2.0", "method": "a"}, {"jsonrpc": "2.0", "method": "b"}], None),
        ("valid control", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": ok_params}, 0),
    ]
    return cases


def run_fuzz(server: MCPServer, session: Session) -> dict[str, Any]:
    correct, failures = 0, []
    cases = fuzz_cases()
    for label, msg, expected in cases:
        raw = msg if isinstance(msg, str) else json.dumps(msg)
        out = server.handle_raw(raw, session)
        if expected is None:
            ok = out is None
        else:
            resp = json.loads(out) if out else None
            if expected == 0:
                ok = isinstance(resp, dict) and "result" in resp
            else:
                ok = isinstance(resp, dict) and resp.get("error", {}).get("code") == expected
        correct += ok
        if not ok:
            failures.append(label)
    return {"cases": len(cases), "correct": correct, "accuracy": round(correct / len(cases), 4), "failures": failures}


PY_ESCAPES = {
    "read /etc/passwd": "print(open('/etc/passwd').read())",
    "write /tmp": "open('/tmp/mcp_bench_escape','w').write('x')",
    "subprocess": "import subprocess; subprocess.run(['id'])",
    "os.system": "import os; os.system('id')",
    "os.fork": "import os; os.fork()",
    "posix_spawn": "import os; os.posix_spawn('/usr/bin/id', ['id'], {})",
    "execv": "import os; os.execv('/usr/bin/id', ['id'])",
    "raw socket": "import socket; socket.socket()",
    "urllib egress": "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9')",
    "ctypes": "import ctypes; ctypes.CDLL(None)",
    "symlink trick": "import os; os.symlink('/etc/hosts','h'); print(open('h').read())",
    "chdir /": "import os; os.chdir('/'); print(os.listdir('.'))",
    "kill parent": "import os; os.kill(os.getppid(), 9)",
    "env leak": "import os, sys; sys.exit(0 if 'BENCH_SECRET' not in os.environ else 3)\n",
    "cpu burn": "while True: pass",
    "sleep forever": "import time; time.sleep(60)",
    "output flood": "while True: print('A'*4096)",
    "memory bomb": "x = bytearray(2 * 1024**3); import time; time.sleep(2)",
    "disk fill": "with open('f','wb') as f:\n    f.write(b'x' * 10_000_000)",
}


def _py_blocked(name: str, r: Any) -> bool:
    if name == "env leak":
        return r.exit_code == 0  # exits 0 only when the secret is absent
    return (not r.ok) or r.output_truncated


def run_escapes(root: Path) -> dict[str, Any]:
    jail = FsJail(root)
    outside = root.parent / "outside_secret.txt"
    outside.write_text("secret")
    (root / "link_out").symlink_to(outside)
    (root / "dir_out").symlink_to(root.parent)
    fs_attempts = {
        "dotdot": lambda: jail.read_text("../outside_secret.txt"),
        "deep dotdot": lambda: jail.read_text("a/b/../../../outside_secret.txt"),
        "absolute": lambda: jail.read_text(str(outside)),
        "etc passwd": lambda: jail.read_text("../../../../../../etc/passwd"),
        "nul byte": lambda: jail.read_text("ok.txt\x00../../etc/passwd"),
        "file symlink": lambda: jail.read_text("link_out"),
        "dir symlink": lambda: jail.read_text("dir_out/outside_secret.txt"),
        "write via dir symlink": lambda: jail.write_text("dir_out/pwned.txt", "x"),
        "write dotdot": lambda: jail.write_text("../pwned.txt", "x"),
        "list parent": lambda: jail.list_dir(".."),
    }
    fs_results: dict[str, bool] = {}
    for name, fn in fs_attempts.items():
        try:
            fn()
            fs_results[name] = False  # the operation went through: not blocked
        except (SandboxViolation, OSError):
            fs_results[name] = True
    if (root.parent / "pwned.txt").exists():
        fs_results["write dotdot"] = fs_results["write via dir symlink"] = False
    fs_blocked = sum(fs_results.values())

    sbx = PythonSandbox(cpu_seconds=1, wall_seconds=2, max_output_bytes=8192, memory_bytes=256 * 1024 * 1024)
    os.environ["BENCH_SECRET"] = "1"
    py_results = {}
    try:
        for name, code in PY_ESCAPES.items():
            py_results[name] = _py_blocked(name, sbx.run(code))
    finally:
        os.environ.pop("BENCH_SECRET", None)
    if os.path.exists("/tmp/mcp_bench_escape"):
        py_results["write /tmp"] = False
    py_blocked = sum(py_results.values())
    total = len(fs_attempts) + len(PY_ESCAPES)
    return {
        "attempts": total,
        "blocked": fs_blocked + py_blocked,
        "fs_attempts": len(fs_attempts),
        "fs_blocked": fs_blocked,
        "python_attempts": len(PY_ESCAPES),
        "python_blocked": py_blocked,
        "not_blocked": [k for k, v in {**fs_results, **py_results}.items() if not v],
    }


def benchmark(inproc_calls: int = 5000, stdio_calls: int = 1000) -> dict[str, Any]:
    """Run all benchmarks; returns JSON-serializable metrics."""
    with tempfile.TemporaryDirectory(prefix="mcp-bench-") as tmp:
        root = Path(tmp) / "root"
        root.mkdir()
        (root / "ok.txt").write_text("fine")
        server = build_server(root, tokens=DEV_TOKENS, sandbox=PythonSandbox(cpu_seconds=1, wall_seconds=2))
        session = connect(server, "admin-token-0001")
        _inproc_rps(server, session, 200)  # warm-up
        inproc = _inproc_rps(server, session, inproc_calls)
        seq, pipelined = _stdio_rps(root, stdio_calls)
        fuzz = run_fuzz(server, connect(server, "admin-token-0001"))
        escapes = run_escapes(root)
    return {
        "inproc_tools_call_rps": round(inproc),
        "stdio_tools_call_rps_sequential": round(seq),
        "stdio_tools_call_rps_pipelined": round(pipelined),
        "inproc_calls": inproc_calls,
        "stdio_calls": stdio_calls,
        "fuzz_cases": fuzz["cases"],
        "fuzz_correct": fuzz["correct"],
        "fuzz_accuracy": fuzz["accuracy"],
        "fuzz_failures": fuzz["failures"],
        "sandbox_escape_attempts": escapes["attempts"],
        "sandbox_escapes_blocked": escapes["blocked"],
        "sandbox_fs_blocked": f'{escapes["fs_blocked"]}/{escapes["fs_attempts"]}',
        "sandbox_python_blocked": f'{escapes["python_blocked"]}/{escapes["python_attempts"]}',
        "sandbox_not_blocked": escapes["not_blocked"],
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
