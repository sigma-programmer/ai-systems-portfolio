"""``python -m mcp_server`` entrypoint.

Configuration (stdio credentials come from the environment, per MCP guidance):

* ``MCP_TOKENS``  JSON ``{token: {"client_id": ..., "scopes": [...]}}`` (or ``--tokens-file``)
* ``MCP_TOKEN``   the token identifying the stdio client (absent -> anonymous, no scopes)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .auth import RateLimiter
from .builtin import build_server
from .sandbox import PythonSandbox
from .transports import MCPHttpServer, serve_stdio


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mcp_server", description="Raw MCP tool server (no SDK)")
    ap.add_argument("--root", required=True, help="sandbox root for filesystem tools")
    ap.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--tokens-file", help="JSON token->principal map (overrides MCP_TOKENS)")
    ap.add_argument("--rate", type=float, default=0.0, help="requests/sec per client (0 = unlimited)")
    ap.add_argument("--burst", type=float, default=50.0)
    ap.add_argument("--allow-host", action="append", default=[], help="egress allowlist for http_get")
    ap.add_argument("--exec-timeout", type=float, default=10.0)
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)

    logging.basicConfig(stream=sys.stderr, level=args.log_level.upper(), format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.tokens_file:
        with open(args.tokens_file, encoding="utf-8") as fh:
            tokens = json.load(fh)
    else:
        tokens = json.loads(os.environ.get("MCP_TOKENS", "{}"))
    limiter = RateLimiter(capacity=args.burst, refill_per_sec=args.rate) if args.rate > 0 else None
    server = build_server(
        args.root,
        tokens=tokens,
        rate_limiter=limiter,
        sandbox=PythonSandbox(wall_seconds=args.exec_timeout),
        net_allowlist=args.allow_host,
    )

    if args.transport == "http":
        http = MCPHttpServer(server, host=args.host, port=args.port)
        print(f"MCP HTTP transport listening on {http.url}", file=sys.stderr)
        try:
            http.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0

    token = os.environ.get("MCP_TOKEN")
    principal = server.authenticator.authenticate(token) if token else None
    if token and principal is None:
        print("mcp_server: MCP_TOKEN is not a known token", file=sys.stderr)
        return 2
    serve_stdio(server, principal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
