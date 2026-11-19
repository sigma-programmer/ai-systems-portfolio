"""Raw MCP tool server: JSON-RPC 2.0 + MCP lifecycle, strict schemas, scoped auth, sandboxing.

No MCP SDK: the protocol is implemented directly on the stdlib + Pydantic v2.
"""
from .auth import ANONYMOUS, Principal, RateLimiter, TokenAuthenticator
from .builtin import build_server, registry as builtin_registry
from .jsonrpc import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PERMISSION_DENIED,
    RATE_LIMITED,
    RESOURCE_NOT_FOUND,
    SERVER_NOT_INITIALIZED,
    Dispatcher,
    JsonRpcError,
)
from .sandbox import ExecResult, FsJail, PythonSandbox, SandboxViolation
from .server import LATEST_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS, MCPServer, Session
from .tools import StrictModel, ToolContext, ToolError, ToolRegistry
from .transports import MCPHttpServer, serve_stdio

__all__ = [
    "ANONYMOUS", "Principal", "RateLimiter", "TokenAuthenticator",
    "build_server", "builtin_registry",
    "INTERNAL_ERROR", "INVALID_PARAMS", "INVALID_REQUEST", "METHOD_NOT_FOUND", "PARSE_ERROR",
    "PERMISSION_DENIED", "RATE_LIMITED", "RESOURCE_NOT_FOUND", "SERVER_NOT_INITIALIZED",
    "Dispatcher", "JsonRpcError",
    "ExecResult", "FsJail", "PythonSandbox", "SandboxViolation",
    "LATEST_PROTOCOL_VERSION", "SUPPORTED_PROTOCOL_VERSIONS", "MCPServer", "Session",
    "StrictModel", "ToolContext", "ToolError", "ToolRegistry",
    "MCPHttpServer", "serve_stdio",
]
