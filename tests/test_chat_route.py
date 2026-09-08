import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from app.main import app
from app.sessions import get_store


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """把 SSE 文本解析成 [(event, data_dict), ...]。data 是 JSON,处理 \\uXXXX 转义。"""
    events: list[tuple[str, dict]] = []
    cur_event: str | None = None
    data_lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("event: "):
            cur_event = line[len("event: "):]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: "):])
        elif line == "" and cur_event:
            events.append((cur_event, json.loads("\n".join(data_lines))))
            cur_event, data_lines = None, []
    if cur_event is not None:
        events.append((cur_event, json.loads("\n".join(data_lines))))
    return events


def _post(client, payload):
    return client.post("/api/chat", json=payload)


def test_chat_route_streams_sse_and_returns_session(client):
    app.state.chat_model = GenericFakeChatModel(messages=iter(["喵,我在。"]))
    r = _post(client, {"message": "在吗"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    names = [e for e, _ in events]
    assert "session" in names and "delta" in names and "done" in names
    content = "".join(data["content"] for name, data in events if name == "delta")
    assert content == "喵,我在。"
    # 首轮缺省 session_id → 应回传新 id
    sid = next(data["session_id"] for name, data in events if name == "session")
    assert sid


def test_chat_route_two_turns_share_session_context(client):
    app.state.chat_model = GenericFakeChatModel(messages=iter(["第一轮:收到订单查询。"]))
    r1 = _post(client, {"session_id": "demo-1", "message": "我的订单怎么了"})
    assert r1.status_code == 200
    events1 = _parse_sse(r1.text)
    content1 = "".join(d["content"] for name, d in events1 if name == "delta")
    assert content1 == "第一轮:收到订单查询。"

    # 第二轮换一个 fake 输出,验证会话仍挂同一 session_id 且历史已累积
    app.state.chat_model = GenericFakeChatModel(messages=iter(["第二轮:已结合上文回复你。"]))
    r2 = _post(client, {"session_id": "demo-1", "message": "那能退货吗"})
    assert r2.status_code == 200
    content2 = "".join(d["content"] for name, d in _parse_sse(r2.text) if name == "delta")
    assert content2 == "第二轮:已结合上文回复你。"

    turns = get_store().get("demo-1").turns
    assert [m["role"] for m in turns] == ["user", "assistant", "user", "assistant"]
    assert turns[0]["content"] == "我的订单怎么了"
    assert turns[2]["content"] == "那能退货吗"


def test_chat_route_model_error_streams_error_event(client):
    class Boom:
        async def astream(self, messages, **kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    app.state.chat_model = Boom()
    r = _post(client, {"message": "hi"})
    body = r.text.replace("\n\n", "\n")
    assert r.status_code == 200
    assert "event: error" in body
    assert "event: done" not in body
