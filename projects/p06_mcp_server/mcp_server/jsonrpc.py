"""JSON-RPC 2.0, implemented from the spec (https://www.jsonrpc.org/specification).

This module knows nothing about MCP. It turns raw bytes/text into method
invocations and turns results/exceptions back into spec-compliant responses:

* single requests, notifications (no ``id`` -> never answered) and batches
* standard error codes -32700 / -32600 / -32601 / -32602 / -32603
* strict ``id`` handling: string or integer only (``null``, booleans, floats
  and containers are rejected, matching MCP's stricter rule that ids must not
  be ``null``)
* responses *received* by the server (objects with ``result``/``error`` and no
  ``method``) are accepted and ignored, as a server never issued requests here
* unexpected handler exceptions become -32603 without leaking internals
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

log = logging.getLogger("mcp_server.jsonrpc")

JSONRPC_VERSION = "2.0"

# Standard codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Implementation-defined server errors (reserved range -32000..-32099).
SERVER_NOT_INITIALIZED = -32000
RESOURCE_NOT_FOUND = -32002  # the code MCP's spec uses for resources/read misses
PERMISSION_DENIED = -32003
RATE_LIMITED = -32029

_STANDARD_MESSAGES = {
    PARSE_ERROR: "Parse error",
    INVALID_REQUEST: "Invalid Request",
    METHOD_NOT_FOUND: "Method not found",
    INVALID_PARAMS: "Invalid params",
    INTERNAL_ERROR: "Internal error",
}

RequestId = str | int
Ctx = TypeVar("Ctx")
Handler = Callable[[dict[str, Any], Any], Any]


class JsonRpcError(Exception):
    """Raise from a method handler to return a JSON-RPC error object."""

    def __init__(self, code: int, message: str | None = None, data: Any = None):
        self.code = code
        self.message = message or _STANDARD_MESSAGES.get(code, "Server error")
        self.data = data
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err


def make_error(request_id: RequestId | None, code: int, message: str | None = None, data: Any = None) -> dict:
    """Build a response carrying an error object."""
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": JsonRpcError(code, message, data).to_dict()}


def make_result(request_id: RequestId, result: Any) -> dict:
    """Build a successful response."""
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant {name!r}")


def parse_json(raw: str | bytes) -> Any:
    """Strict JSON parse: rejects NaN/Infinity, invalid UTF-8 and empty input."""
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8")  # UnicodeDecodeError is a ValueError
    return json.loads(raw, parse_constant=_reject_constant)


def is_valid_id(value: Any) -> bool:
    """Ids must be strings or integers (bool is an int subclass, so exclude it)."""
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


@dataclass
class Dispatcher(Generic[Ctx]):
    """Routes JSON-RPC messages to registered handlers.

    Handlers receive ``(params: dict, ctx)`` and return a JSON-serializable
    result or raise :class:`JsonRpcError`. ``guard`` (optional) runs before
    every handler and may raise to veto a call (used for MCP lifecycle gating,
    auth and rate limiting).
    """

    max_batch: int = 100
    methods: dict[str, Handler] = field(default_factory=dict)
    notifications: dict[str, Handler] = field(default_factory=dict)
    guard: Callable[[str, bool, Any], None] | None = None

    # -- registration -------------------------------------------------------
    def method(self, name: str) -> Callable[[Handler], Handler]:
        def deco(fn: Handler) -> Handler:
            self.methods[name] = fn
            return fn

        return deco

    def notification(self, name: str) -> Callable[[Handler], Handler]:
        def deco(fn: Handler) -> Handler:
            self.notifications[name] = fn
            return fn

        return deco

    # -- processing ---------------------------------------------------------
    def process_raw(self, raw: str | bytes, ctx: Ctx) -> str | None:
        """Parse, dispatch and serialize. Returns ``None`` when nothing is owed."""
        try:
            message = parse_json(raw)
        except (ValueError, RecursionError):
            return json.dumps(make_error(None, PARSE_ERROR))
        response = self.process(message, ctx)
        return None if response is None else json.dumps(response, separators=(",", ":"))

    def process(self, message: Any, ctx: Ctx) -> dict | list | None:
        """Dispatch an already-parsed message (object or batch array)."""
        if isinstance(message, list):
            if not message:
                return make_error(None, INVALID_REQUEST, "Invalid Request: empty batch")
            if len(message) > self.max_batch:
                return make_error(None, INVALID_REQUEST, f"Invalid Request: batch exceeds {self.max_batch} messages")
            out = [r for r in (self._process_one(m, ctx) for m in message) if r is not None]
            return out or None
        return self._process_one(message, ctx)

    def _process_one(self, msg: Any, ctx: Ctx) -> dict | None:
        if not isinstance(msg, dict):
            return make_error(None, INVALID_REQUEST, "Invalid Request: message must be an object")
        if "method" not in msg and ("result" in msg or "error" in msg):
            log.debug("ignoring response object sent to server: id=%r", msg.get("id"))
            return None

        has_id = "id" in msg
        rid = msg.get("id")
        if has_id and not is_valid_id(rid):
            return make_error(None, INVALID_REQUEST, "Invalid Request: id must be a string or integer")
        reply_id = rid if has_id else None

        if msg.get("jsonrpc") != JSONRPC_VERSION:
            return make_error(reply_id, INVALID_REQUEST, 'Invalid Request: "jsonrpc" must be exactly "2.0"')
        method = msg.get("method")
        if not isinstance(method, str) or not method:
            return make_error(reply_id, INVALID_REQUEST, 'Invalid Request: "method" must be a non-empty string')
        if "params" in msg:
            params = msg["params"]
            if isinstance(params, list):
                # Structurally valid JSON-RPC, but MCP only defines by-name params.
                if not has_id:
                    return None
                return make_error(rid, INVALID_PARAMS, "Invalid params: positional (array) params are not supported")
            if not isinstance(params, dict):
                return make_error(reply_id, INVALID_REQUEST, 'Invalid Request: "params" must be an object or array')
        else:
            params = {}

        if not has_id:
            self._notify(method, params, ctx)
            return None
        return self._call(rid, method, params, ctx)

    def _notify(self, method: str, params: dict, ctx: Ctx) -> None:
        handler = self.notifications.get(method)
        if handler is None:
            log.debug("unhandled notification %s", method)
            return
        try:
            if self.guard is not None:
                self.guard(method, True, ctx)
            handler(params, ctx)
        except Exception:  # notifications never produce responses
            log.exception("notification handler %s failed", method)

    def _call(self, rid: RequestId, method: str, params: dict, ctx: Ctx) -> dict:
        handler = self.methods.get(method)
        if handler is None:
            return make_error(rid, METHOD_NOT_FOUND, f"Method not found: {method}")
        try:
            if self.guard is not None:
                self.guard(method, False, ctx)
            result = handler(params, ctx)
            json.dumps(result)  # fail here (-32603), not in the transport
        except JsonRpcError as e:
            return {"jsonrpc": JSONRPC_VERSION, "id": rid, "error": e.to_dict()}
        except Exception:
            log.exception("method %s raised", method)
            return make_error(rid, INTERNAL_ERROR)
        return make_result(rid, result)
