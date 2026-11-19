"""The MCP server core: lifecycle, sessions, authorization and method handlers.

Transport-agnostic: transports create a :class:`Session` bound to an
authenticated :class:`Principal` and feed messages to :meth:`MCPServer.handle_raw`
or :meth:`MCPServer.handle`.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .auth import ANONYMOUS, Principal, RateLimiter, TokenAuthenticator
from .jsonrpc import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    PERMISSION_DENIED,
    RATE_LIMITED,
    RESOURCE_NOT_FOUND,
    SERVER_NOT_INITIALIZED,
    Dispatcher,
    JsonRpcError,
)
from .tools import PromptSpec, ResourceSpec, ToolContext, ToolRegistry

log = logging.getLogger("mcp_server.server")

LATEST_PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "raw-mcp-server", "title": "Raw MCP Tool Server", "version": "1.0.0"}

# Methods allowed before the client completes the initialize handshake.
_PRE_INIT_METHODS = frozenset({"initialize", "ping"})
# Methods that do not consume rate-limit budget.
_UNMETERED = frozenset({"ping", "initialize", "notifications/initialized", "notifications/cancelled"})


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")


class ClientInfo(_Lenient):
    name: str
    version: str


class InitializeParams(_Lenient):
    protocolVersion: str = Field(min_length=1)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    clientInfo: ClientInfo


class CallToolParams(_Lenient):
    name: str = Field(min_length=1)
    arguments: Any = None


class ListParams(_Lenient):
    cursor: str | None = None


class ReadResourceParams(_Lenient):
    uri: str = Field(min_length=1)


class GetPromptParams(_Lenient):
    name: str = Field(min_length=1)
    arguments: dict[str, str] = Field(default_factory=dict)


@dataclass
class Session:
    """State for one client connection."""

    id: str
    principal: Principal
    state: str = "new"  # new -> initializing -> ready -> closed
    protocol_version: str | None = None
    client_info: dict[str, Any] = field(default_factory=dict)
    client_capabilities: dict[str, Any] = field(default_factory=dict)
    cancelled: set[str | int] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def _parse(model: type[BaseModel], params: dict[str, Any]) -> Any:
    try:
        return model.model_validate(params)
    except ValidationError as e:
        details = [{"loc": list(x["loc"]), "msg": x["msg"]} for x in e.errors(include_url=False, include_input=False)]
        raise JsonRpcError(INVALID_PARAMS, "Invalid params", {"errors": details}) from None


def encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"o": offset}).encode()).decode().rstrip("=")


def decode_cursor(cursor: str) -> int:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode()))
        offset = data["o"]
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError
        return offset
    except (ValueError, KeyError, TypeError, binascii.Error):
        raise JsonRpcError(INVALID_PARAMS, "Invalid params: malformed cursor") from None


class MCPServer:
    """An MCP server speaking JSON-RPC 2.0 over any transport."""

    def __init__(
        self,
        tools: ToolRegistry | None = None,
        *,
        authenticator: TokenAuthenticator | None = None,
        rate_limiter: RateLimiter | None = None,
        services: dict[str, Any] | None = None,
        page_size: int = 50,
        strict_lifecycle: bool = True,
        instructions: str | None = None,
    ):
        self.tools = tools or ToolRegistry()
        self.authenticator = authenticator or TokenAuthenticator()
        self.rate_limiter = rate_limiter
        self.services = services or {}
        self.page_size = page_size
        self.strict_lifecycle = strict_lifecycle
        self.instructions = instructions
        self.resources: dict[str, ResourceSpec] = {}
        self.prompts: dict[str, PromptSpec] = {}
        self._sessions: dict[str, Session] = {}
        self._sessions_lock = threading.Lock()
        self.dispatcher: Dispatcher[Session] = Dispatcher(guard=self._guard)
        self._register_methods()

    # -- sessions -----------------------------------------------------------
    def new_session(self, principal: Principal | None = None) -> Session:
        session = Session(id=secrets.token_urlsafe(18), principal=principal or ANONYMOUS)
        with self._sessions_lock:
            self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        with self._sessions_lock:
            return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> bool:
        with self._sessions_lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.state = "closed"
        return session is not None

    def session_count(self) -> int:
        with self._sessions_lock:
            return len(self._sessions)

    # -- registration helpers -----------------------------------------------
    def add_resource(self, spec: ResourceSpec) -> None:
        self.resources[spec.uri] = spec

    def add_prompt(self, spec: PromptSpec) -> None:
        self.prompts[spec.name] = spec

    # -- entry points -------------------------------------------------------
    def handle_raw(self, raw: str | bytes, session: Session) -> str | None:
        """Process one wire message (single or batch); returns the reply text."""
        return self.dispatcher.process_raw(raw, session)

    def handle(self, message: Any, session: Session) -> dict | list | None:
        """Process one already-parsed message."""
        return self.dispatcher.process(message, session)

    # -- guard: lifecycle + rate limit ---------------------------------------
    def _guard(self, method: str, is_notification: bool, session: Session) -> None:
        if session.state == "closed":
            raise JsonRpcError(INVALID_REQUEST, "session is closed")
        if not is_notification and method not in _PRE_INIT_METHODS:
            if session.state == "new" or (self.strict_lifecycle and session.state != "ready"):
                raise JsonRpcError(SERVER_NOT_INITIALIZED, "Server not initialized: complete the initialize handshake first")
        if self.rate_limiter is not None and method not in _UNMETERED:
            key = session.principal.client_id
            if not self.rate_limiter.allow(key):
                raise JsonRpcError(
                    RATE_LIMITED,
                    "Rate limit exceeded",
                    {"retryAfterSeconds": round(self.rate_limiter.retry_after(key), 3)},
                )

    @staticmethod
    def _require(session: Session, scopes: frozenset[str], what: str) -> None:
        missing = session.principal.missing(scopes)
        if missing:
            raise JsonRpcError(PERMISSION_DENIED, f"Permission denied for {what}", {"missingScopes": missing})

    def _page(self, items: list[dict], cursor: str | None, key: str) -> dict[str, Any]:
        offset = decode_cursor(cursor) if cursor is not None else 0
        if offset > len(items):
            raise JsonRpcError(INVALID_PARAMS, "Invalid params: cursor out of range")
        page = items[offset : offset + self.page_size]
        out: dict[str, Any] = {key: page}
        if offset + self.page_size < len(items):
            out["nextCursor"] = encode_cursor(offset + self.page_size)
        return out

    # -- methods ------------------------------------------------------------
    def _register_methods(self) -> None:
        d = self.dispatcher

        @d.method("initialize")
        def initialize(params: dict, session: Session) -> dict:
            if session.state != "new":
                raise JsonRpcError(INVALID_REQUEST, "initialize may only be sent once per session")
            p = _parse(InitializeParams, params)
            requested = p.protocolVersion
            version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
            with session.lock:
                session.protocol_version = version
                session.client_info = p.clientInfo.model_dump()
                session.client_capabilities = p.capabilities
                session.state = "initializing"
            result: dict[str, Any] = {
                "protocolVersion": version,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "serverInfo": dict(SERVER_INFO),
            }
            if self.instructions:
                result["instructions"] = self.instructions
            log.info("session %s initialized by %s (protocol %s)", session.id[:8], session.client_info.get("name"), version)
            return result

        @d.notification("notifications/initialized")
        def initialized(params: dict, session: Session) -> None:
            if session.state == "initializing":
                session.state = "ready"

        @d.notification("notifications/cancelled")
        def cancelled(params: dict, session: Session) -> None:
            rid = params.get("requestId")
            if isinstance(rid, (str, int)):
                session.cancelled.add(rid)

        @d.method("ping")
        def ping(params: dict, session: Session) -> dict:
            return {}

        @d.method("tools/list")
        def tools_list(params: dict, session: Session) -> dict:
            p = _parse(ListParams, params)
            items = [t.descriptor() for t in self.tools.visible_to(session.principal)]
            return self._page(items, p.cursor, "tools")

        @d.method("tools/call")
        def tools_call(params: dict, session: Session) -> dict:
            p = _parse(CallToolParams, params)
            spec = self.tools.get(p.name)
            # Unauthorized tools are indistinguishable from unknown ones in
            # tools/list, but a direct call gets an explicit denial.
            if spec is None:
                raise JsonRpcError(INVALID_PARAMS, f"Unknown tool: {p.name}")
            self._require(session, spec.scopes, f"tool {p.name!r}")
            ctx = ToolContext(principal=session.principal, session_id=session.id, services=self.services)
            return self.tools.invoke(spec, p.arguments, ctx)

        @d.method("resources/list")
        def resources_list(params: dict, session: Session) -> dict:
            p = _parse(ListParams, params)
            items = [r.descriptor() for r in self._visible_resources(session)]
            return self._page(items, p.cursor, "resources")

        @d.method("resources/read")
        def resources_read(params: dict, session: Session) -> dict:
            p = _parse(ReadResourceParams, params)
            spec = self._resolve_resource(p.uri)
            if spec is None:
                raise JsonRpcError(RESOURCE_NOT_FOUND, "Resource not found", {"uri": p.uri})
            self._require(session, spec.scopes, f"resource {p.uri!r}")
            try:
                text = spec.reader()
            except (OSError, ValueError) as e:
                raise JsonRpcError(RESOURCE_NOT_FOUND, f"Resource unreadable: {e}", {"uri": p.uri}) from None
            return {"contents": [{"uri": spec.uri, "mimeType": spec.mime_type, "text": text}]}

        @d.method("prompts/list")
        def prompts_list(params: dict, session: Session) -> dict:
            p = _parse(ListParams, params)
            items = [s.descriptor() for s in self.prompts.values() if not session.principal.missing(s.scopes)]
            return self._page(items, p.cursor, "prompts")

        @d.method("prompts/get")
        def prompts_get(params: dict, session: Session) -> dict:
            p = _parse(GetPromptParams, params)
            spec = self.prompts.get(p.name)
            if spec is None:
                raise JsonRpcError(INVALID_PARAMS, f"Unknown prompt: {p.name}")
            self._require(session, spec.scopes, f"prompt {p.name!r}")
            known = {a.name for a in spec.arguments}
            unknown = sorted(set(p.arguments) - known)
            missing = sorted(a.name for a in spec.arguments if a.required and a.name not in p.arguments)
            if unknown or missing:
                raise JsonRpcError(INVALID_PARAMS, "Invalid prompt arguments", {"missing": missing, "unknown": unknown})
            return {"description": spec.description, "messages": spec.render(dict(p.arguments))}

    # -- resources: static + dynamic sandbox files --------------------------
    def _visible_resources(self, session: Session) -> list[ResourceSpec]:
        out = [r for r in self.resources.values() if not session.principal.missing(r.scopes)]
        jail = self.services.get("jail")
        if jail is not None and session.principal.has_scope("fs:read"):
            out.extend(self._file_resource(jail, rel) for rel in jail.walk_files())
        return out

    def _resolve_resource(self, uri: str) -> ResourceSpec | None:
        if uri in self.resources:
            return self.resources[uri]
        jail = self.services.get("jail")
        prefix = "sandbox://files/"
        if jail is not None and uri.startswith(prefix):
            return self._file_resource(jail, uri[len(prefix):])
        return None

    @staticmethod
    def _file_resource(jail: Any, rel: str) -> ResourceSpec:
        def reader() -> str:
            text, _size, _trunc = jail.read_text(rel)
            return text

        return ResourceSpec(
            uri=f"sandbox://files/{rel}",
            name=rel,
            reader=reader,
            description="File inside the sandbox root",
            mime_type="text/plain",
            scopes=frozenset({"fs:read"}),
        )
