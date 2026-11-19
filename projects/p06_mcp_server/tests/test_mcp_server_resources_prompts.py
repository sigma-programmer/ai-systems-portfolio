"""resources/* and prompts/* minimal implementations."""
from __future__ import annotations

import json

from mcp_server import PERMISSION_DENIED, RESOURCE_NOT_FOUND
from mcp_server.jsonrpc import INVALID_PARAMS
from mcp_server.testing import rpc


def test_resources_list_scoped(server, connect):
    anon = rpc(server, connect(None), "resources/list")["result"]["resources"]
    assert [r["uri"] for r in anon] == ["mcp://server/info"]
    reader = rpc(server, connect("reader-token-001"), "resources/list")["result"]["resources"]
    assert [r["uri"] for r in reader] == ["mcp://server/info", "sandbox://files/hello.txt", "sandbox://files/sub/nested.txt"]


def test_read_static_resource(server, connect):
    res = rpc(server, connect(None), "resources/read", {"uri": "mcp://server/info"})["result"]["contents"][0]
    assert res["mimeType"] == "application/json" and json.loads(res["text"])["tools"] == 7


def test_read_file_resource(server, connect):
    s = connect("reader-token-001")
    assert rpc(server, s, "resources/read", {"uri": "sandbox://files/sub/nested.txt"})["result"]["contents"][0]["text"] == "nested\n"


def test_file_resource_requires_scope(server, connect):
    err = rpc(server, connect(None), "resources/read", {"uri": "sandbox://files/hello.txt"})["error"]
    assert err["code"] == PERMISSION_DENIED


def test_resource_not_found_and_escape(server, connect):
    s = connect("reader-token-001")
    assert rpc(server, s, "resources/read", {"uri": "mcp://nope"})["error"]["code"] == RESOURCE_NOT_FOUND
    assert rpc(server, s, "resources/read", {"uri": "sandbox://files/missing"})["error"]["code"] == RESOURCE_NOT_FOUND
    err = rpc(server, s, "resources/read", {"uri": "sandbox://files/../secret.txt"})["error"]
    assert err["code"] == RESOURCE_NOT_FOUND and "SECRET" not in json.dumps(err)
    assert rpc(server, s, "resources/read", {})["error"]["code"] == INVALID_PARAMS


def test_prompts(server, admin):
    prompts = rpc(server, admin, "prompts/list")["result"]["prompts"]
    assert prompts[0]["name"] == "code_review"
    assert {"name": "code", "description": "The code to review", "required": True} in prompts[0]["arguments"]
    got = rpc(server, admin, "prompts/get", {"name": "code_review", "arguments": {"code": "x=1", "language": "python"}})["result"]
    msg = got["messages"][0]
    assert msg["role"] == "user" and msg["content"]["type"] == "text" and "python" in msg["content"]["text"]


def test_prompt_argument_errors(server, admin):
    err = rpc(server, admin, "prompts/get", {"name": "code_review", "arguments": {}})["error"]
    assert err["code"] == INVALID_PARAMS and err["data"]["missing"] == ["code"]
    err = rpc(server, admin, "prompts/get", {"name": "code_review", "arguments": {"code": "x", "bogus": "y"}})["error"]
    assert err["data"]["unknown"] == ["bogus"]
    assert rpc(server, admin, "prompts/get", {"name": "nope"})["error"]["code"] == INVALID_PARAMS
    assert rpc(server, admin, "prompts/get", {"name": "code_review", "arguments": {"code": 5}})["error"]["code"] == INVALID_PARAMS
