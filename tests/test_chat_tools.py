import pytest
from langchain_core.messages import AIMessageChunk

from app.chat import ChatService
from app.config import Settings
from app.tools.registry import ToolRegistry

pytestmark = pytest.mark.anyio

SETTINGS = Settings(history_budget_tokens=2048, tool_timeout_seconds=2, tool_max_retries=0)


class ScriptedModel:
    """第一次 astream(bind_tools 后)出工具调用; 第二段(不带工具)出文本。

    真实 LangChain 的 ``bind_tools`` 返回一个新的 Runnable,不改动原模型;
    这里同样返回一份 ``bound=True`` 的新实例,使第二段仍拿得到未绑工具的模型。
    """

    def __init__(self, first_chunks, final_text, *, bound: bool = False):
        self._first, self._final = first_chunks, final_text
        self.bound = bound

    def bind_tools(self, tools, **_kw):
        return ScriptedModel(self._first, self._final, bound=True)

    async def astream(self, _messages, **_kw):
        if self.bound:
            for c in self._first:
                yield c
        else:
            for ch in self._final:
                yield AIMessageChunk(content=ch)


def _toolcall(name, args_json, cid):
    return AIMessageChunk(content="", tool_call_chunks=[
        {"name": name, "args": args_json, "id": cid, "index": 0, "type": "tool_call_chunk"}
    ])


async def _collect(gen):
    return [e async for e in gen]


async def test_tool_round_emits_frames_and_persists(session_factory, db_session):
    from app.tools.business import query_order

    model = ScriptedModel([_toolcall("query_order", '{"order_id": "1001"}', "c1")], ["订单", "已发货"])
    svc = ChatService(
        model=model,
        registry_factory=lambda cid: ToolRegistry([query_order]),
        session_factory=session_factory,
        system_prompt="你是客服",
        settings=SETTINGS,
    )
    events = await _collect(svc.stream_turn("u1", None, "订单1001到哪了"))
    names = [e.event for e in events]
    assert names[0] == "session" and names[-1] == "done"
    assert "tool" in names and "delta" in names
    tool_frames = [e.data for e in events if e.event == "tool"]
    assert tool_frames[0]["status"] == "running" and tool_frames[-1]["status"] == "ok"
    final_text = "".join(e.data["content"] for e in events if e.event == "delta")
    assert final_text == "订单已发货"

    from app.db import repository as repo

    conv_id = events[0].data["conversation_id"]
    hist = await repo.load_history(db_session, conv_id)
    assert [m["role"] for m in hist][:3] == ["user", "assistant", "tool"]
    assert hist[-1]["role"] == "assistant" and hist[-1]["content"] == "订单已发货"


async def test_pure_chat_path_still_streams_without_tools(session_factory):
    model = ScriptedModel([], ["你好", "呀"])
    svc = ChatService(
        model=model, registry_factory=lambda cid: ToolRegistry([]), session_factory=session_factory,
        system_prompt="你是客服", settings=SETTINGS,
    )
    events = await _collect(svc.stream_turn("u1", None, "你好"))
    assert "tool" not in [e.event for e in events]
    assert "".join(e.data["content"] for e in events if e.event == "delta") == "你好呀"


async def test_unparseable_tool_call_json_surfaces_as_error_frame(session_factory, db_session):
    """模型吐出的工具调用 JSON 坏掉时,本轮不应静默无输出:调用方把它转成 error 帧。"""
    from app.tools.business import query_order

    model = ScriptedModel([_toolcall("query_order", "not-json", "c1")], ["不会走到这里"])
    svc = ChatService(
        model=model, registry_factory=lambda cid: ToolRegistry([query_order]), session_factory=session_factory,
        system_prompt="你是客服", settings=SETTINGS,
    )
    events = await _collect(svc.stream_turn("u1", None, "订单1001到哪了"))
    names = [e.event for e in events]
    assert "error" in names and "done" not in names
    assert all(e.event != "tool" for e in events)

    from app.db import repository as repo

    cid = events[0].data["conversation_id"]
    hist = await repo.load_history(db_session, cid)
    assert [m["role"] for m in hist] == ["user"]  # 坏轮不留 assistant/tool 行
