from langchain_core.messages import AIMessageChunk

from app.main import app


class ScriptedModel:
    def __init__(self, first, final):
        self._first, self._final, self.bound = first, final, False

    def bind_tools(self, tools, **_kw):
        self.bound = True
        return self

    async def astream(self, _m, **_kw):
        for c in (self._first if self.bound else self._final):
            yield c


def _toolcall(name, args_json, cid):
    return AIMessageChunk(content="", tool_call_chunks=[
        {"name": name, "args": args_json, "id": cid, "index": 0, "type": "tool_call_chunk"}
    ])


def _parse(body):
    import json
    import re

    return [(m[0], json.loads(m[1])) for m in re.findall(r"event: (\w+)\ndata: (.+)", body)]


def test_route_returns_conversation_id_and_tool_frames(client):
    app.state.chat_model = ScriptedModel([_toolcall("query_order", '{"order_id":"1001"}', "c1")], [AIMessageChunk(content="已发货")])
    r1 = client.post("/api/chat", json={"user_id": "u1", "message": "订单1001到哪了"})
    assert r1.status_code == 200
    frames = _parse(r1.text)
    names = [n for n, _ in frames]
    assert names[0] == "session" and names[-1] == "done" and "tool" in names
    cid = dict(frames)["session"]["conversation_id"]
    assert cid > 0

    # 第二轮:带 conversation_id,不应再发 session 帧
    app.state.chat_model = ScriptedModel([], [AIMessageChunk(content="嗯嗯")])
    r2 = client.post("/api/chat", json={"user_id": "u1", "conversation_id": cid, "message": "谢谢"})
    frames2 = _parse(r2.text)
    assert "session" not in [n for n, _ in frames2]
    assert dict(frames2)["done"]["conversation_id"] == cid
