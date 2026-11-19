"""Filesystem jail and Python sandbox."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from mcp_server import FsJail, PythonSandbox, SandboxViolation
from mcp_server.testing import rpc


# -- FsJail -------------------------------------------------------------------
@pytest.fixture
def jail(root: Path) -> FsJail:
    return FsJail(root)


def test_resolve_inside(jail, root):
    assert jail.resolve("hello.txt") == Path(os.path.realpath(root / "hello.txt"))
    assert jail.resolve("/sub/nested.txt") == Path(os.path.realpath(root / "sub/nested.txt"))  # absolute -> rooted
    assert jail.resolve("sub/../hello.txt").name == "hello.txt"
    assert jail.resolve(".") == jail.root
    assert jail.resolve("") == jail.root


@pytest.mark.parametrize("path", ["../secret.txt", "sub/../../secret.txt", "../../../../etc/passwd", "a\x00b", "..", "/../secret.txt"])
def test_traversal_blocked(jail, path):
    with pytest.raises(SandboxViolation):
        jail.resolve(path)


def test_symlink_escape_blocked(jail, root):
    (root / "evil").symlink_to(root.parent / "secret.txt")
    (root / "evildir").symlink_to(root.parent)
    with pytest.raises(SandboxViolation):
        jail.read_text("evil")
    with pytest.raises(SandboxViolation):
        jail.read_text("evildir/secret.txt")
    with pytest.raises(SandboxViolation):
        jail.write_text("evildir/new.txt", "x")
    assert not (root.parent / "new.txt").exists()


def test_internal_symlink_allowed(jail, root):
    (root / "alias").symlink_to(root / "hello.txt")
    assert jail.read_text("alias")[0] == "hello world\n"


def test_read_truncation_and_errors(jail, root):
    (root / "big.txt").write_text("x" * 100)
    text, size, truncated = jail.read_text("big.txt", max_bytes=10)
    assert text == "x" * 10 and size == 100 and truncated
    with pytest.raises(IsADirectoryError):
        jail.read_text("sub")
    with pytest.raises(FileNotFoundError):
        jail.read_text("missing.txt")


def test_write_semantics(jail, root):
    assert jail.write_text("new/deep/file.txt", "abc") == 3
    assert (root / "new/deep/file.txt").read_text() == "abc"
    with pytest.raises(FileExistsError):
        jail.write_text("new/deep/file.txt", "zzz")
    jail.write_text("new/deep/file.txt", "zzz", overwrite=True)
    assert (root / "new/deep/file.txt").read_text() == "zzz"
    with pytest.raises(SandboxViolation):
        jail.write_text(".", "x")
    with pytest.raises(ValueError):
        FsJail(root, max_write_bytes=2).write_text("t.txt", "abc")


def test_list_and_walk(jail, root):
    (root / "link").symlink_to(root.parent)
    entries = {e["name"]: e for e in jail.list_dir(".")}
    assert entries["hello.txt"] == {"name": "hello.txt", "type": "file", "size": 12}
    assert entries["sub"]["type"] == "dir" and entries["link"]["type"] == "symlink"
    assert jail.walk_files() == ["hello.txt", "sub/nested.txt"]
    with pytest.raises(NotADirectoryError):
        jail.list_dir("hello.txt")


def test_jail_root_must_exist(tmp_path):
    with pytest.raises(ValueError):
        FsJail(tmp_path / "nope")


def test_fs_tools_via_protocol(server, admin, root):
    call = lambda name, args: rpc(server, admin, "tools/call", {"name": name, "arguments": args})["result"]
    assert call("fs_read", {"path": "hello.txt"})["structuredContent"]["content"] == "hello world\n"
    bad = call("fs_read", {"path": "../secret.txt"})
    assert bad["isError"] is True and "escapes" in bad["content"][0]["text"]
    assert "TOP SECRET" not in str(bad)
    assert call("fs_write", {"path": "out/a.txt", "content": "hi"})["structuredContent"] == {"path": "out/a.txt", "bytes_written": 2}
    assert call("fs_write", {"path": "out/a.txt", "content": "hi"})["isError"] is True
    assert [e["name"] for e in call("fs_list", {"path": "out"})["structuredContent"]["entries"]] == ["a.txt"]


# -- Python sandbox -----------------------------------------------------------
@pytest.fixture(scope="module")
def sbx() -> PythonSandbox:
    return PythonSandbox(cpu_seconds=2, wall_seconds=3, max_output_bytes=4096, memory_bytes=200 * 1024 * 1024)


def test_exec_happy_path(sbx):
    r = sbx.run("import json, math\nprint(json.dumps({'pi': round(math.pi, 2)}))\nopen('f.txt','w').write('ok')\nprint(open('f.txt').read())")
    assert r.ok and r.stdout == '{"pi": 3.14}\nok\n'
    assert r.limits_applied["RLIMIT_CPU"] == 2 and r.limits_applied["RLIMIT_NPROC"] == 0


def test_exec_environment_is_empty_and_isolated(sbx):
    os.environ["MCP_TEST_SECRET"] = "leaky"
    try:
        r = sbx.run("import os, sys\nprint('MCP_TEST_SECRET' in os.environ)\nprint(sys.flags.isolated, sys.flags.no_site)")
    finally:
        del os.environ["MCP_TEST_SECRET"]
    assert r.stdout.split() == ["False", "1", "1"]


@pytest.mark.parametrize(
    "name,code",
    [
        ("read_outside", "print(open('/etc/passwd').read())"),
        ("write_outside", "open('/tmp/mcp_sbx_escape','w').write('x')"),
        ("subprocess", "import subprocess; subprocess.run(['echo','hi'])"),
        ("os_system", "import os; os.system('echo pwned')"),
        ("fork", "import os; os.fork()"),
        ("posix_spawn", "import os; os.posix_spawn('/bin/echo', ['echo'], {})"),
        ("execv", "import os; os.execv('/bin/echo', ['echo'])"),
        ("socket", "import socket; socket.socket()"),
        ("urllib", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:1')"),
        ("ctypes", "import ctypes"),
        ("symlink_trick", "import os; os.symlink('/etc/passwd', 'p'); print(open('p').read())"),
        ("chdir_root", "import os; os.chdir('/'); print(os.listdir('.'))"),
        ("listdir_home", "import os; print(os.listdir('/Users'))"),
        ("remove_outside", "import os; os.remove('/tmp/whatever')"),
        ("kill_parent", "import os; os.kill(os.getppid(), 9)"),
        ("gc_tamper", "import gc"),
        ("extra_hook_cannot_disable", "import sys; sys.addaudithook(lambda *a: None); open('/etc/passwd')"),
    ],
)
def test_escape_attempts_blocked(sbx, name, code):
    r = sbx.run(code)
    assert r.exit_code != 0, f"{name} unexpectedly succeeded: {r.stdout}"
    assert "PermissionError" in r.stderr or "sandbox" in r.stderr
    assert "root:" not in r.stdout
    assert not os.path.exists("/tmp/mcp_sbx_escape")


def test_cpu_limit(sbx):
    r = sbx.run("while True: pass")
    # SIGXCPU = soft CPU limit; SIGKILL = hard-limit backstop (platform-dependent delivery).
    assert not r.ok and (r.signal in ("SIGXCPU", "SIGKILL") or r.timed_out)
    assert r.duration_ms < 10_000  # killed by the CPU limit, not left spinning


def test_wall_clock_timeout(sbx):
    r = sbx.run("import time; time.sleep(30)", wall_seconds=0.5)
    assert r.timed_out and r.duration_ms < 3000


def test_output_cap(sbx):
    r = sbx.run("while True: print('A' * 1000)")
    assert r.output_truncated and len(r.stdout) <= 4096 and not r.ok


def test_memory_limit(sbx):
    r = sbx.run("x = bytearray(1024 * 1024 * 1024)\nimport time; time.sleep(1)\nprint(len(x))")
    assert not r.ok and "1073741824" not in r.stdout


def test_disk_write_capped(sbx):
    r = sbx.run("with open('big.bin','wb') as f:\n    f.write(b'x' * 100000)")
    assert "File too large" in r.stderr and r.output_truncated


def test_user_exception_reported(sbx):
    r = sbx.run("raise ValueError('boom')")
    assert r.exit_code == 1 and "ValueError: boom" in r.stderr and "runner.py" not in r.stderr


def test_python_exec_tool(server, admin):
    res = rpc(server, admin, "tools/call", {"name": "python_exec", "arguments": {"code": "print(6*7)"}})["result"]
    assert res["isError"] is False and res["structuredContent"]["stdout"] == "42\n"
    res = rpc(server, admin, "tools/call", {"name": "python_exec", "arguments": {"code": "import socket"}})["result"]
    assert res["structuredContent"]["exit_code"] == 1 and "not permitted" in res["structuredContent"]["stderr"]
    err = rpc(server, admin, "tools/call", {"name": "python_exec", "arguments": {"code": "1", "timeout_s": 99}})["error"]
    assert err["code"] == -32602
