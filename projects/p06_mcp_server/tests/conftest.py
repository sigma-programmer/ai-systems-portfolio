"""Shared fixtures for the mcp_server test suite."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from mcp_server import PythonSandbox, build_server
from mcp_server.server import MCPServer, Session
from mcp_server.testing import DEV_TOKENS, connect as _connect


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "root"
    r.mkdir()
    (r / "hello.txt").write_text("hello world\n")
    (r / "sub").mkdir()
    (r / "sub" / "nested.txt").write_text("nested\n")
    (tmp_path / "secret.txt").write_text("TOP SECRET\n")
    return r


@pytest.fixture
def server(root: Path) -> MCPServer:
    return build_server(root, tokens=DEV_TOKENS, sandbox=PythonSandbox(cpu_seconds=2, wall_seconds=3))


@pytest.fixture
def connect(server: MCPServer) -> Callable[[str | None], Session]:
    """Factory returning a ready (handshaken) session for a token."""
    return lambda token="admin-token-0001": _connect(server, token)


@pytest.fixture
def admin(connect: Callable[[str | None], Session]) -> Session:
    return connect("admin-token-0001")
