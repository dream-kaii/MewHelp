import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import tool

from app.agent.tool_runner import execute_tool_calls, stream_first_round
from app.tools.registry import ToolRegistry

pytestmark = pytest.mark.anyio


class FakeBoundModel:
    """模拟已 bind_tools 的模型:按脚本吐出 chunk。"""

    def __init__(self, chunks):
        self._chunks = chunks

    async def astream(self, _messages, **_kw):
        for c in self._chunks:
            yield c


def _text_chunk(text):
    return AIMessageChunk(content=text)


def _tool_chunk(name, args_json, call_id):
    return AIMessageChunk(content="", tool_call_chunks=[
        {"name": name, "args": args_json, "id": call_id, "index": 0, "type": "tool_call_chunk"}
    ])


async def test_stream_first_round_collects_text_and_no_tools():
    model = FakeBoundModel([_text_chunk("你好"), _text_chunk("呀")])
    seen = []
    calls, text = await stream_first_round(model, [], lambda t: seen.append(t))
    assert text == "你好呀" and seen == ["你好", "呀"] and calls == []


async def test_stream_first_round_collects_tool_calls():
    model = FakeBoundModel([_tool_chunk("query_order", '{"order_id": "1001"}', "call_a")])
    calls, text = await stream_first_round(model, [], lambda t: None)
    assert text == "" and len(calls) == 1
    assert calls[0]["name"] == "query_order" and calls[0]["args"] == {"order_id": "1001"}


async def test_stream_first_round_merges_tool_call_across_chunks():
    """真实流式里 tool_call 的 name/args 会拆到多个 chunk,靠 acc + chunk 合并。"""
    model = FakeBoundModel([
        AIMessageChunk(content="", tool_call_chunks=[
            {"name": "query_order", "args": '{"order', "id": "call_a", "index": 0, "type": "tool_call_chunk"}
        ]),
        AIMessageChunk(content="", tool_call_chunks=[
            {"name": None, "args": '_id": "1001"}', "id": None, "index": 0, "type": "tool_call_chunk"}
        ]),
    ])
    calls, text = await stream_first_round(model, [], lambda t: None)
    assert text == ""
    assert calls == [{"id": "call_a", "name": "query_order", "args": {"order_id": "1001"}}]


async def test_stream_first_round_raises_on_unparseable_tool_call_json():
    """模型吐出的工具调用 JSON 无法解析(tool_calls 空、invalid_tool_calls 非空)必须抛错,
    否则本轮会静默无输出。调用方(ChatService)会把它转成 error 帧。"""
    model = FakeBoundModel([
        AIMessageChunk(content="", tool_call_chunks=[
            {"name": "query_order", "args": "not-json", "id": "call_x", "index": 0, "type": "tool_call_chunk"}
        ])
    ])
    with pytest.raises(ValueError, match="工具调用"):
        await stream_first_round(model, [], lambda t: None)


@tool
def ping(x: str) -> str:
    """ping。"""
    return "pong:" + x


async def test_execute_tool_calls_reports_status_for_each():
    reg = ToolRegistry([ping])
    events = []

    def on_status(phase, call_id, name, args, summary):
        events.append((phase, name, summary))

    results = await execute_tool_calls(
        reg, [{"id": "c1", "name": "ping", "args": {"x": "a"}}],
        on_status, timeout=2, retries=0,
    )
    assert results[0].ok and results[0].content == "pong:a"
    assert events[0][0] == "running" and events[-1][0] == "ok"


async def test_execute_tool_calls_runs_all_calls_and_reports_each():
    """一轮多个调用全部执行:每个调用 running -> ok,结果顺序与入参一致。"""
    reg = ToolRegistry([ping])
    events = []

    def on_status(phase, call_id, name, args, summary):
        events.append((call_id, phase, summary))

    results = await execute_tool_calls(
        reg,
        [{"id": "c1", "name": "ping", "args": {"x": "a"}},
         {"id": "c2", "name": "ping", "args": {"x": "b"}}],
        on_status, timeout=2, retries=0,
    )
    assert [r.content for r in results] == ["pong:a", "pong:b"]
    per_call = {cid: [ph for c, ph, _ in events if c == cid] for cid in ("c1", "c2")}
    assert per_call == {"c1": ["running", "ok"], "c2": ["running", "ok"]}


async def test_execute_tool_calls_empty_returns_empty_without_status():
    reg = ToolRegistry([ping])
    events = []
    results = await execute_tool_calls(
        reg, [], lambda *a: events.append(a), timeout=2, retries=0
    )
    assert results == [] and events == []


async def test_execute_tool_calls_reports_error_status():
    @tool
    def boom() -> str:
        """总是失败。"""
        raise RuntimeError("kaboom")

    reg = ToolRegistry([boom])
    events = []
    results = await execute_tool_calls(
        reg, [{"id": "c9", "name": "boom", "args": {}}],
        lambda phase, cid, name, args, summary: events.append((phase, summary)),
        timeout=2, retries=0,
    )
    assert not results[0].ok
    assert [e[0] for e in events] == ["running", "error"]
    assert "kaboom" in events[-1][1]
