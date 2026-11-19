"""The default tool set and a factory that wires a complete server."""
from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from pydantic import Field

from .auth import RateLimiter, TokenAuthenticator
from .sandbox import FsJail, PythonSandbox
from .server import SERVER_INFO, MCPServer
from .tools import PromptArgument, PromptSpec, ResourceSpec, StrictModel, ToolContext, ToolError, ToolRegistry

registry = ToolRegistry()


# --------------------------------------------------------------------------
# Public tools (no scope required)
# --------------------------------------------------------------------------
class EchoIn(StrictModel):
    text: str = Field(max_length=10_000, description="Text to echo back")


@registry.tool(read_only=True, title="Echo")
def echo(args: EchoIn) -> str:
    """Echo the input text unchanged."""
    return args.text


class AddIn(StrictModel):
    a: float = Field(description="First addend")
    b: float = Field(description="Second addend")


class AddOut(StrictModel):
    sum: float


@registry.tool(read_only=True, title="Add numbers")
def add(args: AddIn) -> AddOut:
    """Add two numbers and return a structured result."""
    return AddOut(sum=args.a + args.b)


# --------------------------------------------------------------------------
# Filesystem tools (jailed)
# --------------------------------------------------------------------------
class FsReadIn(StrictModel):
    path: str = Field(min_length=1, max_length=1024, description="Path relative to the sandbox root")
    max_bytes: int = Field(default=65_536, ge=1, le=1_000_000)


class FsReadOut(StrictModel):
    path: str
    content: str
    size: int
    truncated: bool


@registry.tool(scopes={"fs:read"}, read_only=True, title="Read file")
def fs_read(args: FsReadIn, ctx: ToolContext) -> FsReadOut:
    """Read a UTF-8 text file inside the sandbox root."""
    jail: FsJail = ctx.service("jail")
    text, size, truncated = jail.read_text(args.path, args.max_bytes)
    return FsReadOut(path=jail.relative(jail.resolve(args.path)), content=text, size=size, truncated=truncated)


class FsListIn(StrictModel):
    path: str = Field(default=".", max_length=1024)


class FsEntry(StrictModel):
    name: str
    type: str
    size: int


class FsListOut(StrictModel):
    path: str
    entries: list[FsEntry]


@registry.tool(scopes={"fs:read"}, read_only=True, title="List directory")
def fs_list(args: FsListIn, ctx: ToolContext) -> FsListOut:
    """List a directory inside the sandbox root (symlinks are not followed)."""
    jail: FsJail = ctx.service("jail")
    entries = [FsEntry(**e) for e in jail.list_dir(args.path)]
    return FsListOut(path=jail.relative(jail.resolve(args.path)), entries=entries)


class FsWriteIn(StrictModel):
    path: str = Field(min_length=1, max_length=1024)
    content: str = Field(max_length=1_000_000)
    overwrite: bool = False


class FsWriteOut(StrictModel):
    path: str
    bytes_written: int


@registry.tool(scopes={"fs:write"}, destructive=True, title="Write file")
def fs_write(args: FsWriteIn, ctx: ToolContext) -> FsWriteOut:
    """Write a UTF-8 text file inside the sandbox root."""
    jail: FsJail = ctx.service("jail")
    n = jail.write_text(args.path, args.content, overwrite=args.overwrite)
    return FsWriteOut(path=jail.relative(jail.resolve(args.path)), bytes_written=n)


# --------------------------------------------------------------------------
# Sandboxed Python
# --------------------------------------------------------------------------
class PythonExecIn(StrictModel):
    code: str = Field(min_length=1, max_length=20_000, description="Python 3 source to execute")
    timeout_s: float = Field(default=5.0, gt=0, le=30, description="Wall-clock timeout (capped by server)")


class PythonExecOut(StrictModel):
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    output_truncated: bool
    memory_exceeded: bool
    duration_ms: float
    limits_applied: dict[str, int]


@registry.tool(scopes={"exec:python"}, open_world=False, title="Run Python (sandboxed)")
def python_exec(args: PythonExecIn, ctx: ToolContext) -> PythonExecOut:
    """Run Python in an isolated, resource-limited subprocess (no network, no subprocesses, no files outside a temp dir)."""
    sandbox: PythonSandbox = ctx.service("python")
    r = sandbox.run(args.code, wall_seconds=args.timeout_s)
    return PythonExecOut(
        exit_code=r.exit_code,
        stdout=r.stdout,
        stderr=r.stderr,
        timed_out=r.timed_out,
        output_truncated=r.output_truncated,
        memory_exceeded=r.memory_exceeded,
        duration_ms=r.duration_ms,
        limits_applied=r.limits_applied,
    )


# --------------------------------------------------------------------------
# Network (egress allowlist)
# --------------------------------------------------------------------------
class HttpGetIn(StrictModel):
    url: str = Field(min_length=1, max_length=2048)
    max_bytes: int = Field(default=65_536, ge=1, le=1_000_000)


class HttpGetOut(StrictModel):
    url: str
    status: int
    content_type: str
    body: str
    truncated: bool


def check_egress(url: str, allowed_hosts: Iterable[str]) -> urllib.parse.ParseResult:
    """Allow only http(s) to allowlisted hostnames that do not resolve to private IPs."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolError(f"scheme {parsed.scheme!r} not allowed")
    host = (parsed.hostname or "").lower()
    if not host or host not in {h.lower() for h in allowed_hosts}:
        raise ToolError(f"host {host!r} is not on the egress allowlist")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as e:
        raise ToolError(f"cannot resolve {host!r}: {e}") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ToolError(f"host {host!r} resolves to non-public address {ip}")
    return parsed


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a: Any, **k: Any) -> None:  # redirects could bypass the allowlist
        return None


@registry.tool(scopes={"net"}, read_only=True, open_world=True, title="HTTP GET")
def http_get(args: HttpGetIn, ctx: ToolContext) -> HttpGetOut:
    """Fetch a URL from an allowlisted host (no redirects, private IPs blocked)."""
    allowed: Iterable[str] = ctx.services.get("net_allowlist", ())
    check_egress(args.url, allowed)
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(urllib.request.Request(args.url, headers={"User-Agent": "raw-mcp-server"}), timeout=10) as resp:
            data = resp.read(args.max_bytes + 1)
            return HttpGetOut(
                url=args.url,
                status=resp.status,
                content_type=resp.headers.get("Content-Type", ""),
                body=data[: args.max_bytes].decode("utf-8", errors="replace"),
                truncated=len(data) > args.max_bytes,
            )
    except urllib.error.URLError as e:
        raise ToolError(f"request failed: {e}") from None


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def _review_prompt(args: dict[str, str]) -> list[dict[str, Any]]:
    focus = args.get("focus", "correctness and security")
    text = f"Review the following {args.get('language', 'code')} for {focus}. Be specific.\n\n{args['code']}"
    return [{"role": "user", "content": {"type": "text", "text": text}}]


def build_server(
    root: str | Path,
    *,
    tokens: dict[str, dict[str, Any]] | None = None,
    rate_limiter: RateLimiter | None = None,
    sandbox: PythonSandbox | None = None,
    net_allowlist: Iterable[str] = (),
    page_size: int = 50,
    strict_lifecycle: bool = True,
) -> MCPServer:
    """Create a fully wired server with the built-in tools, a resource and a prompt."""
    jail = FsJail(Path(root))
    server = MCPServer(
        registry,
        authenticator=TokenAuthenticator.from_config(tokens or {}),
        rate_limiter=rate_limiter,
        services={"jail": jail, "python": sandbox or PythonSandbox(), "net_allowlist": tuple(net_allowlist)},
        page_size=page_size,
        strict_lifecycle=strict_lifecycle,
        instructions="Tools are filtered by your token's scopes. File paths are relative to the sandbox root.",
    )
    server.add_resource(
        ResourceSpec(
            uri="mcp://server/info",
            name="server-info",
            description="Server name, version and tool count",
            mime_type="application/json",
            reader=lambda: json.dumps({**SERVER_INFO, "tools": len(registry.all())}),
        )
    )
    server.add_prompt(
        PromptSpec(
            name="code_review",
            description="Ask the model to review a code snippet",
            arguments=[
                PromptArgument("code", "The code to review", required=True),
                PromptArgument("language", "Programming language"),
                PromptArgument("focus", "What to focus on"),
            ],
            render=_review_prompt,
        )
    )
    return server
