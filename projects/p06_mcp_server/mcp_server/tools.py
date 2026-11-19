"""Tool, resource and prompt registries.

Tools are plain functions registered with a decorator. The first parameter's
annotation must be a Pydantic v2 model; it becomes the tool's ``inputSchema``
(the model must forbid extra keys, enforced at registration). If the return
annotation is a model, it becomes ``outputSchema`` and results are returned as
``structuredContent`` *and* as a JSON text block for older clients.

    registry = ToolRegistry()

    @registry.tool(scopes={"fs:read"}, read_only=True)
    def fs_read(args: FsReadIn, ctx: ToolContext) -> FsReadOut:
        "Read a UTF-8 file inside the sandbox root."
"""
from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, get_type_hints

from pydantic import BaseModel, ConfigDict, ValidationError

from .auth import Principal
from .jsonrpc import INTERNAL_ERROR, INVALID_PARAMS, JsonRpcError

log = logging.getLogger("mcp_server.tools")


class StrictModel(BaseModel):
    """Base for tool I/O: unknown keys rejected, no lax type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ToolError(Exception):
    """Expected, user-facing tool failure -> ``isError: true`` result."""


@dataclass
class ToolContext:
    """Per-call context handed to tool handlers that accept a second arg."""

    principal: Principal
    session_id: str
    services: dict[str, Any] = field(default_factory=dict)

    def service(self, name: str) -> Any:
        try:
            return self.services[name]
        except KeyError:
            raise ToolError(f"service {name!r} is not configured on this server") from None


@dataclass
class ToolSpec:
    """A registered tool."""

    name: str
    description: str
    handler: Callable[..., Any]
    input_model: type[BaseModel]
    output_model: type[BaseModel] | None
    scopes: frozenset[str]
    takes_ctx: bool
    title: str | None = None
    annotations: dict[str, Any] = field(default_factory=dict)

    def input_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.setdefault("type", "object")
        return schema

    def descriptor(self) -> dict[str, Any]:
        """The object advertised in ``tools/list``."""
        d: dict[str, Any] = {"name": self.name, "description": self.description, "inputSchema": self.input_schema()}
        if self.title:
            d["title"] = self.title
        if self.output_model is not None:
            d["outputSchema"] = self.output_model.model_json_schema()
        if self.annotations:
            d["annotations"] = dict(self.annotations)
        if self.scopes:
            d["_meta"] = {"requiredScopes": sorted(self.scopes)}
        return d


def _validation_details(err: ValidationError) -> list[dict[str, Any]]:
    return [
        {"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]}
        for e in err.errors(include_url=False, include_context=False, include_input=False)
    ]


def text_block(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


class ToolRegistry:
    """Ordered collection of tools with schema generation and invocation."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        scopes: Iterable[str] = (),
        title: str | None = None,
        read_only: bool | None = None,
        destructive: bool | None = None,
        open_world: bool | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator registering ``fn`` as a tool."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                fn, name=name, description=description, scopes=scopes, title=title,
                read_only=read_only, destructive=destructive, open_world=open_world,
            )
            return fn

        return deco

    def register(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        scopes: Iterable[str] = (),
        title: str | None = None,
        read_only: bool | None = None,
        destructive: bool | None = None,
        open_world: bool | None = None,
    ) -> ToolSpec:
        tool_name = name or fn.__name__
        if tool_name in self._tools:
            raise ValueError(f"duplicate tool name {tool_name!r}")
        params = list(inspect.signature(fn).parameters.values())
        if not params or len(params) > 2:
            raise TypeError(f"tool {tool_name!r} must take (args) or (args, ctx)")
        hints = get_type_hints(fn)
        input_model = hints.get(params[0].name)
        if not (isinstance(input_model, type) and issubclass(input_model, BaseModel)):
            raise TypeError(f"tool {tool_name!r}: first parameter must be annotated with a pydantic model")
        if input_model.model_config.get("extra") != "forbid":
            raise TypeError(f"tool {tool_name!r}: input model must set extra='forbid'")
        ret = hints.get("return")
        output_model = ret if isinstance(ret, type) and issubclass(ret, BaseModel) else None
        annotations: dict[str, Any] = {}
        if title:
            annotations["title"] = title
        for key, val in (("readOnlyHint", read_only), ("destructiveHint", destructive), ("openWorldHint", open_world)):
            if val is not None:
                annotations[key] = val
        spec = ToolSpec(
            name=tool_name,
            description=(description or inspect.getdoc(fn) or "").strip(),
            handler=fn,
            input_model=input_model,
            output_model=output_model,
            scopes=frozenset(scopes),
            takes_ctx=len(params) == 2,
            title=title,
            annotations=annotations,
        )
        self._tools[tool_name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def visible_to(self, principal: Principal) -> list[ToolSpec]:
        return [t for t in self._tools.values() if not principal.missing(t.scopes)]

    # -- invocation ---------------------------------------------------------
    def invoke(self, spec: ToolSpec, arguments: Any, ctx: ToolContext) -> dict[str, Any]:
        """Validate, run and wrap a tool call as an MCP ``CallToolResult``.

        Protocol errors (bad arguments) raise :class:`JsonRpcError`; failures
        *inside* the tool become ``isError: true`` results so the model can
        see and react to them.
        """
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise JsonRpcError(INVALID_PARAMS, "Invalid params: 'arguments' must be an object")
        try:
            args = spec.input_model.model_validate(arguments)
        except ValidationError as e:
            raise JsonRpcError(
                INVALID_PARAMS,
                f"Invalid arguments for tool {spec.name!r}",
                {"errors": _validation_details(e)},
            ) from None
        try:
            out = spec.handler(args, ctx) if spec.takes_ctx else spec.handler(args)
        except JsonRpcError:
            raise
        except ToolError as e:
            return {"content": [text_block(str(e))], "isError": True}
        except (OSError, ValueError, PermissionError, TimeoutError) as e:
            return {"content": [text_block(f"{type(e).__name__}: {e}")], "isError": True}
        except Exception:
            log.exception("tool %s crashed", spec.name)
            return {"content": [text_block(f"tool {spec.name!r} failed unexpectedly")], "isError": True}
        return self._wrap(spec, out)

    def _wrap(self, spec: ToolSpec, out: Any) -> dict[str, Any]:
        if spec.output_model is not None:
            try:
                model = out if isinstance(out, spec.output_model) else spec.output_model.model_validate(out)
            except ValidationError as e:
                log.error("tool %s returned invalid output: %s", spec.name, e)
                raise JsonRpcError(INTERNAL_ERROR, f"tool {spec.name!r} produced output violating its outputSchema") from None
            data = model.model_dump(mode="json")
            return {"content": [text_block(json.dumps(data))], "structuredContent": data, "isError": False}
        if isinstance(out, str):
            return {"content": [text_block(out)], "isError": False}
        if isinstance(out, BaseModel):
            return {"content": [text_block(out.model_dump_json())], "isError": False}
        return {"content": [text_block(json.dumps(out, default=str))], "isError": False}


# ---------------------------------------------------------------------------
# Resources & prompts (minimal)
# ---------------------------------------------------------------------------
@dataclass
class ResourceSpec:
    uri: str
    name: str
    reader: Callable[[], str]
    description: str = ""
    mime_type: str = "text/plain"
    scopes: frozenset[str] = frozenset()

    def descriptor(self) -> dict[str, Any]:
        return {"uri": self.uri, "name": self.name, "description": self.description, "mimeType": self.mime_type}


@dataclass
class PromptArgument:
    name: str
    description: str = ""
    required: bool = False


@dataclass
class PromptSpec:
    name: str
    description: str
    arguments: list[PromptArgument]
    render: Callable[[dict[str, str]], list[dict[str, Any]]]
    scopes: frozenset[str] = frozenset()

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": [{"name": a.name, "description": a.description, "required": a.required} for a in self.arguments],
        }
