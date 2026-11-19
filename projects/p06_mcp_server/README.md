# P06 — Raw MCP Tool Server

> You understand tooling at the protocol level.

A Model Context Protocol server written from scratch on the standard library and
Pydantic v2: no MCP SDK. It implements JSON-RPC 2.0, the MCP lifecycle, tools,
resources and prompts, token-scoped permissions, per-client rate limiting, a
filesystem jail, a resource-limited Python sandbox, and two transports (stdio and
a Streamable-HTTP-style POST endpoint).

## Problem

Agent frameworks hide the tool protocol behind SDKs, so the hard parts stay
out of sight: which errors are *protocol* errors and which are *tool* errors,
how a server decides what one client may see and call, and what really stops a
tool from reading `/etc/passwd`. This project makes each of those decisions in
code you can read and tests that exercise it.

## Architecture

```
  stdin/stdout (NDJSON)            HTTP POST /mcp (Bearer, Mcp-Session-Id)
          |                                     |
   serve_stdio()                         MCPHttpServer (ThreadingHTTPServer)
          |   one Session per connection        |   one Session per initialize
          +-----------------+-------------------+
                            v
                 MCPServer.handle_raw(raw, session)
                            |
                 jsonrpc.Dispatcher  ── parse (-32700), shape (-32600),
                            |            batches, id rules, notifications
                            v
             _guard: lifecycle (initialize first, -32002)
                     rate limit (token bucket per client, -32029)
                            |
     +-----------+----------+-----------+-------------+
     v           v                      v             v
 initialize   tools/list (scope-     tools/call    resources/*  prompts/*
 ping         filtered, cursor)         |
                                        v
                         ToolRegistry: scope check (-32003)
                         Pydantic(extra="forbid") validation (-32602)
                         handler(args, ToolContext)
                           |-- FsJail (realpath + O_NOFOLLOW)
                           |-- PythonSandbox (subprocess, -I -S, empty env,
                           |     setrlimit, wall timeout, audit hook)
                           '-- http_get (egress allowlist, no redirects)
                         result -> content[] + structuredContent / isError
```

| Module | Role |
|---|---|
| `jsonrpc.py` | JSON-RPC 2.0 from scratch: strict JSON parsing (rejects `NaN`), batches, notification vs request, id validation, standard error codes |
| `server.py` | MCP lifecycle, protocol version negotiation, sessions, scope checks, cursor pagination, resources and prompts |
| `tools.py` | `@registry.tool(scopes=...)` decorator; input schema from a Pydantic model (must be `extra="forbid"`), `outputSchema` + `structuredContent` from the return model |
| `auth.py` | Token to `Principal(client_id, scopes)` (constant-time digest lookup), token-bucket `RateLimiter` with an injectable clock |
| `sandbox.py` | `FsJail` and `PythonSandbox` |
| `builtin.py` | `echo`, `add`, `fs_read`, `fs_list`, `fs_write`, `python_exec`, `http_get`, plus resources and a `code_review` prompt |
| `transports.py` | stdio loop and HTTP transport |
| `client.py` | Minimal stdio and HTTP clients, used by tests, the demo and the benchmark |

## Key design decisions and tradeoffs

- **Protocol error or tool error.** Unknown tool, schema violation (including extra
  keys) and missing scope come back as JSON-RPC errors (`-32602`, `-32003`): the
  model never sees them as tool output, and the caller has to fix the call. A tool
  that *runs* and fails (a sandbox violation, a file that doesn't exist, a
  `ToolError`) returns `result.isError=true` with a text content block, so the model
  can read it and recover. Unexpected exceptions are logged to stderr and reported
  as a generic `isError` result, so internals don't leak.
- **Version negotiation.** If the client asks for a version the server supports
  (`2025-06-18`, `2025-03-26`, `2024-11-05`), the server echoes it. Otherwise it
  answers with its latest version and the client decides whether to continue.
- **Deny by default.** `tools/list` shows only the tools whose scopes the
  principal holds. A call to a hidden tool is still checked and refused with the
  missing scopes in `error.data`. Anonymous stdio sessions get no scopes.
- **Sandbox in layers, not one wall.** The filesystem jail resolves the real path
  and requires it to stay under the root, so `..`, absolute paths and symlinks that
  point outside are all refused. Opens use `O_NOFOLLOW`. `python_exec` runs
  `python -I -S` with an empty environment, a private temp working directory, a new
  process group, `setrlimit` (CPU, file size, open files, processes, and address
  space where macOS enforces it), a wall-clock SIGKILL and an output size cap. A
  PEP 578 audit hook denies subprocesses, sockets, ctypes and file access outside
  the work directory. Tradeoff: this is defense in depth for a portfolio server,
  not a substitute for a container, VM or seccomp boundary.
- **The HTTP transport is a subset of Streamable HTTP.** Replies are plain JSON,
  with no SSE streaming. The transport handles session ids, returns 202 for bodies
  that are only notifications, validates `Origin` against DNS rebinding and binds
  to loopback by default.
- **Stdout carries only protocol.** All logging goes to stderr, and oversized lines
  are drained and answered with `-32600`.

## How to run

```bash
# tests (147: JSON-RPC, lifecycle, tools, auth/scopes, sandbox, HTTP, stdio subprocess)
.venv/bin/python -m pytest projects/p06_mcp_server

# end-to-end demo (stdio subprocess + HTTP transport)
AIKIT_OFFLINE=1 .venv/bin/python projects/p06_mcp_server/demo.py

# stdio server (credentials via env, per MCP guidance)
cd projects/p06_mcp_server
MCP_TOKENS='{"t1":{"client_id":"me","scopes":["fs:read","exec:python"]}}' MCP_TOKEN=t1 \
  ../../.venv/bin/python -m mcp_server --root /tmp/sandbox
# HTTP transport
../../.venv/bin/python -m mcp_server --root /tmp/sandbox --transport http --port 8765

# benchmark
cd projects/p06_mcp_server && ../../.venv/bin/python -c "import json, mcp_server.bench as b; print(json.dumps(b.benchmark(), indent=1))"
```

## Benchmark (measured on an Apple Silicon Mac, Python 3.14)

| Metric | Value |
|---|---|
| `tools/call` in-process dispatch | ~51k–74k req/s (5,000 calls) |
| `tools/call` over stdio, sequential round-trips | ~31k req/s (1,000 calls) |
| `tools/call` over stdio, pipelined | ~67k req/s |
| Malformed-request fuzz set classified correctly | **54 / 54** (accuracy 1.0) |
| Sandbox escape attempts blocked | **29 / 29** (filesystem 10/10, python_exec 19/19) |

The fuzz set covers invalid JSON (truncated input, `NaN`, trailing commas, single
quotes), non-object messages, empty batches, bad `jsonrpc` versions, invalid ids
(null, bool, object, array, float), bad or reserved method names, params that
aren't objects, a second `initialize`, unknown tools, missing, extra, coerced or
wrong-typed tool arguments, values out of bounds, bad cursors, missing resources
and prompts, and notifications, responses and batches that must get no reply.
Each case has to produce exactly the expected error code, or no reply. Escape
attempts: `..` and deep `..` traversal, absolute paths, NUL bytes, file and
directory symlinks out of the root (read and write), listing the parent;
`open('/etc/passwd')`, writing to `/tmp`, `subprocess`, `os.system`, `fork`,
`posix_spawn`, `execv`, raw sockets, urllib egress, `ctypes`, a symlink trick,
`chdir('/')`, killing the parent, environment leakage, CPU burn, sleeping forever,
output flood, a 2 GiB memory bomb and a disk fill.
