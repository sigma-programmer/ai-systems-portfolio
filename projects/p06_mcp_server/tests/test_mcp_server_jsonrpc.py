"""JSON-RPC 2.0 conformance of the bare dispatcher."""
from __future__ import annotations

import json

import pytest

from mcp_server.jsonrpc import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    Dispatcher,
    JsonRpcError,
)


@pytest.fixture
def d() -> Dispatcher:
    disp: Dispatcher = Dispatcher(max_batch=5)
    seen: list = []

    @disp.method("sum")
    def _sum(params, ctx):
        return params["a"] + params["b"]

    @disp.method("fail")
    def _fail(params, ctx):
        raise JsonRpcError(-32042, "custom", {"why": "because"})

    @disp.method("crash")
    def _crash(params, ctx):
        raise RuntimeError("secret internals")

    @disp.method("unserializable")
    def _bad(params, ctx):
        return {1, 2}

    @disp.notification("note")
    def _note(params, ctx):
        seen.append(params)
        raise RuntimeError("notifications swallow errors")

    disp.seen = seen  # type: ignore[attr-defined]
    return disp


def call(d: Dispatcher, raw: str):
    out = d.process_raw(raw, None)
    return None if out is None else json.loads(out)


def test_success_and_id_echo(d):
    assert call(d, '{"jsonrpc":"2.0","id":7,"method":"sum","params":{"a":1,"b":2}}') == {"jsonrpc": "2.0", "id": 7, "result": 3}
    assert call(d, '{"jsonrpc":"2.0","id":"abc","method":"sum","params":{"a":1,"b":2}}')["id"] == "abc"


@pytest.mark.parametrize("raw", ['{"jsonrpc":"2.0",', "", "not json", '{"a":NaN}', b"\xff\xfe".decode("latin1") + "{"])
def test_parse_error(d, raw):
    resp = call(d, raw)
    assert resp["error"]["code"] == PARSE_ERROR and resp["id"] is None


def test_invalid_utf8_bytes_is_parse_error(d):
    assert json.loads(d.process_raw(b'{"x":"\xff"}', None))["error"]["code"] == PARSE_ERROR


@pytest.mark.parametrize(
    "msg,expected_id",
    [
        ('{"jsonrpc":"1.0","id":1,"method":"sum"}', 1),
        ('{"id":1,"method":"sum"}', 1),
        ('{"jsonrpc":"2.0","id":1,"method":5}', 1),
        ('{"jsonrpc":"2.0","id":1,"method":""}', 1),
        ('{"jsonrpc":"2.0","id":1}', 1),
        ('{"jsonrpc":"2.0","id":1,"method":"sum","params":"x"}', 1),
        ('{"jsonrpc":"2.0","id":null,"method":"sum"}', None),
        ('{"jsonrpc":"2.0","id":true,"method":"sum"}', None),
        ('{"jsonrpc":"2.0","id":[1],"method":"sum"}', None),
        ('{"jsonrpc":"2.0","id":{"a":1},"method":"sum"}', None),
        ('{"jsonrpc":"2.0","method":1,"params":"bar"}', None),
        ("1", None),
        ('"hello"', None),
        ("null", None),
    ],
)
def test_invalid_request(d, msg, expected_id):
    resp = call(d, msg)
    assert resp["error"]["code"] == INVALID_REQUEST
    assert resp["id"] == expected_id


def test_method_not_found(d):
    resp = call(d, '{"jsonrpc":"2.0","id":1,"method":"nope"}')
    assert resp["error"]["code"] == METHOD_NOT_FOUND and resp["id"] == 1


def test_positional_params_rejected_as_invalid_params(d):
    assert call(d, '{"jsonrpc":"2.0","id":1,"method":"sum","params":[1,2]}')["error"]["code"] == INVALID_PARAMS


def test_custom_error_passthrough(d):
    err = call(d, '{"jsonrpc":"2.0","id":1,"method":"fail"}')["error"]
    assert err == {"code": -32042, "message": "custom", "data": {"why": "because"}}


def test_internal_error_hides_details(d):
    err = call(d, '{"jsonrpc":"2.0","id":1,"method":"crash"}')["error"]
    assert err["code"] == INTERNAL_ERROR and "secret" not in json.dumps(err)


def test_unserializable_result_is_internal_error(d):
    assert call(d, '{"jsonrpc":"2.0","id":1,"method":"unserializable"}')["error"]["code"] == INTERNAL_ERROR


def test_notifications_get_no_response_even_on_error(d):
    assert call(d, '{"jsonrpc":"2.0","method":"note","params":{"x":1}}') is None
    assert d.seen == [{"x": 1}]
    assert call(d, '{"jsonrpc":"2.0","method":"unknown"}') is None
    assert call(d, '{"jsonrpc":"2.0","method":"sum","params":[1]}') is None


def test_response_objects_are_ignored(d):
    assert call(d, '{"jsonrpc":"2.0","id":3,"result":{}}') is None
    assert call(d, '{"jsonrpc":"2.0","id":3,"error":{"code":1,"message":"x"}}') is None


def test_batch_mixed(d):
    raw = json.dumps([
        {"jsonrpc": "2.0", "id": 1, "method": "sum", "params": {"a": 1, "b": 1}},
        {"jsonrpc": "2.0", "method": "note"},
        {"foo": "boo"},
        {"jsonrpc": "2.0", "id": "x", "method": "nope"},
        5,
    ])
    resp = call(d, raw)
    assert isinstance(resp, list) and len(resp) == 4
    assert resp[0]["result"] == 2
    assert resp[1]["error"]["code"] == INVALID_REQUEST
    assert resp[2]["error"]["code"] == METHOD_NOT_FOUND and resp[2]["id"] == "x"
    assert resp[3]["error"]["code"] == INVALID_REQUEST


def test_empty_batch_is_single_invalid_request(d):
    resp = call(d, "[]")
    assert isinstance(resp, dict) and resp["error"]["code"] == INVALID_REQUEST


def test_all_notification_batch_returns_nothing(d):
    assert call(d, json.dumps([{"jsonrpc": "2.0", "method": "note"}] * 3)) is None


def test_batch_size_cap(d):
    big = json.dumps([{"jsonrpc": "2.0", "id": i, "method": "sum", "params": {"a": 0, "b": 0}} for i in range(6)])
    assert call(d, big)["error"]["code"] == INVALID_REQUEST


def test_guard_can_veto(d):
    def guard(method, is_note, ctx):
        if method == "sum":
            raise JsonRpcError(-32003, "denied")

    d.guard = guard
    assert call(d, '{"jsonrpc":"2.0","id":1,"method":"sum","params":{"a":1,"b":2}}')["error"]["code"] == -32003


def test_deeply_nested_json_is_rejected_without_crashing(d):
    resp = call(d, "[" * 100_000 + "]" * 100_000)
    errors = resp if isinstance(resp, list) else [resp]
    assert errors and all(e["error"]["code"] in (PARSE_ERROR, INVALID_REQUEST) for e in errors)
