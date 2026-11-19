"""Tool registration, schemas, validation mapping and pagination."""
from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, Field

from mcp_server import MCPServer, StrictModel, ToolError, ToolRegistry
from mcp_server.jsonrpc import INTERNAL_ERROR, INVALID_PARAMS
from mcp_server.testing import connect, rpc


def _tools(server, session):
    return {t["name"]: t for t in rpc(server, session, "tools/list")["result"]["tools"]}


def test_builtin_schemas_are_strict(server, admin):
    tools = _tools(server, admin)
    add = tools["add"]
    assert add["inputSchema"]["type"] == "object"
    assert add["inputSchema"]["additionalProperties"] is False
    assert set(add["inputSchema"]["required"]) == {"a", "b"}
    assert add["outputSchema"]["properties"]["sum"]["type"] == "number"
    assert add["annotations"]["readOnlyHint"] is True
    assert "outputSchema" not in tools["echo"]  # str return -> unstructured
    assert tools["fs_write"]["_meta"]["requiredScopes"] == ["fs:write"]
    assert tools["fs_write"]["annotations"]["destructiveHint"] is True
    assert tools["fs_read"]["inputSchema"]["properties"]["max_bytes"]["maximum"] == 1_000_000


def test_structured_content(server, admin):
    r = rpc(server, admin, "tools/call", {"name": "add", "arguments": {"a": 2, "b": 0.5}})["result"]
    assert r["isError"] is False
    assert r["structuredContent"] == {"sum": 2.5}
    assert r["content"] == [{"type": "text", "text": '{"sum": 2.5}'}]


def test_text_content(server, admin):
    r = rpc(server, admin, "tools/call", {"name": "echo", "arguments": {"text": "hi"}})["result"]
    assert r == {"content": [{"type": "text", "text": "hi"}], "isError": False}


@pytest.mark.parametrize(
    "arguments,loc",
    [
        ({"a": 1}, ["b"]),                       # missing field
        ({"a": "1", "b": 2}, ["a"]),             # strict: no str->float coercion
        ({"a": 1, "b": 2, "c": 3}, ["c"]),       # extra forbidden
        ({"a": None, "b": 2}, ["a"]),
        ({"a": [1], "b": 2}, ["a"]),
    ],
)
def test_invalid_arguments_are_protocol_errors(server, admin, arguments, loc):
    err = rpc(server, admin, "tools/call", {"name": "add", "arguments": arguments})["error"]
    assert err["code"] == INVALID_PARAMS
    assert any(e["loc"] == loc for e in err["data"]["errors"])


def test_arguments_must_be_object(server, admin):
    assert rpc(server, admin, "tools/call", {"name": "add", "arguments": [1, 2]})["error"]["code"] == INVALID_PARAMS


def test_missing_arguments_treated_as_empty(server, admin):
    assert rpc(server, admin, "tools/call", {"name": "add"})["error"]["code"] == INVALID_PARAMS  # a,b required
    r = rpc(server, admin, "tools/call", {"name": "fs_list"})["result"]  # all defaults
    assert r["isError"] is False


def test_unknown_tool_and_bad_call_params(server, admin):
    assert rpc(server, admin, "tools/call", {"name": "nope", "arguments": {}})["error"]["code"] == INVALID_PARAMS
    assert rpc(server, admin, "tools/call", {"arguments": {}})["error"]["code"] == INVALID_PARAMS
    assert rpc(server, admin, "tools/call", {"name": ""})["error"]["code"] == INVALID_PARAMS


def test_bool_bool_strictness(server, admin):
    err = rpc(server, admin, "tools/call", {"name": "fs_write", "arguments": {"path": "x", "content": "y", "overwrite": "yes"}})
    assert err["error"]["code"] == INVALID_PARAMS


def test_field_constraints(server, admin):
    err = rpc(server, admin, "tools/call", {"name": "echo", "arguments": {"text": "x" * 10_001}})["error"]
    assert err["code"] == INVALID_PARAMS and err["data"]["errors"][0]["type"] == "string_too_long"


# -- custom registry behaviors -----------------------------------------------
class In(StrictModel):
    n: int = Field(ge=0)


class Out(StrictModel):
    doubled: int


def make_custom() -> MCPServer:
    reg = ToolRegistry()

    @reg.tool(description="explicit description")
    def double(args: In) -> Out:
        return Out(doubled=args.n * 2)

    @reg.tool()
    def boom(args: In) -> str:
        """Fails with an expected error."""
        raise ToolError("cannot do that")

    @reg.tool()
    def crash(args: In) -> str:
        raise RuntimeError("internal secret")

    @reg.tool()
    def liar(args: In) -> Out:
        return {"doubled": "not-an-int"}  # type: ignore[return-value]

    @reg.tool()
    def dict_out(args: In) -> Out:
        return {"doubled": args.n * 2}  # type: ignore[return-value]

    def mk(i):
        def f(args: In) -> str:
            return str(i)
        f.__doc__ = f"bulk tool {i}"
        return f

    for i in range(7):
        reg.register(mk(i), name=f"bulk_{i}")
    return MCPServer(reg, page_size=4)


def test_custom_tools_results():
    srv = make_custom()
    s = connect(srv, None)
    assert rpc(srv, s, "tools/call", {"name": "double", "arguments": {"n": 4}})["result"]["structuredContent"] == {"doubled": 8}
    assert rpc(srv, s, "tools/call", {"name": "dict_out", "arguments": {"n": 4}})["result"]["structuredContent"] == {"doubled": 8}
    boom = rpc(srv, s, "tools/call", {"name": "boom", "arguments": {"n": 1}})["result"]
    assert boom == {"content": [{"type": "text", "text": "cannot do that"}], "isError": True}
    crash = rpc(srv, s, "tools/call", {"name": "crash", "arguments": {"n": 1}})["result"]
    assert crash["isError"] is True and "secret" not in crash["content"][0]["text"]
    assert rpc(srv, s, "tools/call", {"name": "liar", "arguments": {"n": 1}})["error"]["code"] == INTERNAL_ERROR
    assert rpc(srv, s, "tools/call", {"name": "double", "arguments": {"n": -1}})["error"]["code"] == INVALID_PARAMS


def test_description_from_docstring_or_arg():
    srv = make_custom()
    s = connect(srv, None)
    first = rpc(srv, s, "tools/list")["result"]["tools"]
    names = {t["name"]: t for t in first}
    assert names["double"]["description"] == "explicit description"
    assert names["boom"]["description"] == "Fails with an expected error."


def test_pagination_walks_all_tools():
    srv = make_custom()
    s = connect(srv, None)
    seen, cursor, pages = [], None, 0
    while True:
        params = {"cursor": cursor} if cursor else {}
        res = rpc(srv, s, "tools/list", params)["result"]
        seen += [t["name"] for t in res["tools"]]
        pages += 1
        assert len(res["tools"]) <= 4
        cursor = res.get("nextCursor")
        if not cursor:
            break
    assert pages == 3 and len(seen) == 12 == len(set(seen))


@pytest.mark.parametrize("cursor", ["garbage!!", "eyJ4IjoxfQ", "eyJvIjotMX0", "eyJvIjo5OTl9", 5])
def test_bad_cursor(cursor):
    srv = make_custom()
    s = connect(srv, None)
    assert rpc(srv, s, "tools/list", {"cursor": cursor})["error"]["code"] == INVALID_PARAMS


class Loose(BaseModel):
    x: int


class Ok(BaseModel):
    model_config = ConfigDict(extra="forbid")


def loose_tool(args: Loose) -> str:
    return ""


def untyped_tool(args: int) -> str:
    return ""


def noargs_tool() -> str:
    return ""


def ok_tool(args: Ok) -> str:
    return ""


def test_registration_errors():
    reg = ToolRegistry()
    with pytest.raises(TypeError, match="extra='forbid'"):
        reg.register(loose_tool)
    with pytest.raises(TypeError, match="pydantic model"):
        reg.register(untyped_tool)
    with pytest.raises(TypeError, match=r"\(args\)"):
        reg.register(noargs_tool)
    reg.register(ok_tool)
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(ok_tool)
    assert reg.get("ok_tool") is not None and reg.get("missing") is None


# -- egress policy for http_get (no real network needed) ----------------------
def test_egress_policy():
    from mcp_server.builtin import check_egress

    with pytest.raises(ToolError, match="scheme"):
        check_egress("file:///etc/passwd", ["localhost"])
    with pytest.raises(ToolError, match="allowlist"):
        check_egress("https://evil.example/x", ["good.example"])
    with pytest.raises(ToolError, match="non-public"):
        check_egress("http://localhost:8080/admin", ["localhost"])


def test_http_get_tool_denies_unlisted_host(server, admin):
    res = rpc(server, admin, "tools/call", {"name": "http_get", "arguments": {"url": "https://example.com"}})["result"]
    assert res["isError"] is True and "allowlist" in res["content"][0]["text"]
