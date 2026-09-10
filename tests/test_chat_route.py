import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from app.main import app


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


class RecordingModel:
    """包装 GenericFakeChatModel:astream 前记录收到的消息,用于断言跨轮历史(来自 DB)是否传进模型。

    bind_tools 返回自身:ch02 的 ChatService 会先 bind_tools 再跑第一段,
    这里保持“同一模型实例、bound 与否都照常出文本”,单轮只发生一次 astream。
    """

    def __init__(self, responses):
        self._inner = GenericFakeChatModel(messages=iter(responses))
        self.seen_inputs: list[list] = []

    def bind_tools(self, tools, **kwargs):
        return self

    async def astream(self, messages, **kwargs):
        self.seen_inputs.append(list(messages))
        async for chunk in self._inner.astream(messages, **kwargs):
            yield chunk


def _post(client, payload):
    return client.post("/api/chat", json=payload)


def test_chat_route_streams_sse_and_returns_conversation_id(client):
    app.state.chat_model = RecordingModel(["喵,我在。"])
    r = _post(client, {"user_id": "u1", "message": "在吗"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    names = [e for e, _ in events]
    assert names[0] == "session" and "delta" in names and names[-1] == "done"
    content = "".join(data["content"] for name, data in events if name == "delta")
    assert content == "喵,我在。"
    # 首轮无 conversation_id → 应新建并回传自增主键
    cid = dict(events)["session"]["conversation_id"]
    assert cid > 0
    assert dict(events)["done"]["conversation_id"] == cid


def test_chat_route_second_turn_carries_history_from_db(client):
    model1 = RecordingModel(["第一轮:收到订单查询。"])
    app.state.chat_model = model1
    r1 = _post(client, {"user_id": "u1", "message": "我的订单怎么了"})
    assert r1.status_code == 200
    events1 = _parse_sse(r1.text)
    assert "".join(d["content"] for name, d in events1 if name == "delta") == "第一轮:收到订单查询。"
    cid = dict(events1)["session"]["conversation_id"]

    # 第二轮带 conversation_id:不再发 session 帧,且第一轮问答已从 DB 回灌进模型输入
    model2 = RecordingModel(["第二轮:已结合上文回复你。"])
    app.state.chat_model = model2
    r2 = _post(client, {"user_id": "u1", "conversation_id": cid, "message": "那能退货吗"})
    assert r2.status_code == 200
    events2 = _parse_sse(r2.text)
    assert "session" not in [e for e, _ in events2]
    assert "".join(d["content"] for name, d in events2 if name == "delta") == "第二轮:已结合上文回复你。"
    assert dict(events2)["done"]["conversation_id"] == cid

    assert len(model2.seen_inputs) == 1
    roles = [type(m).__name__ for m in model2.seen_inputs[0]]
    assert roles == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]
    contents = [getattr(m, "content", "") for m in model2.seen_inputs[0]]
    assert "我的订单怎么了" in contents and "第一轮:收到订单查询。" in contents and "那能退货吗" in contents


def test_chat_route_model_error_streams_error_event(client):
    class Boom:
        def bind_tools(self, tools, **kwargs):
            return self

        async def astream(self, messages, **kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    app.state.chat_model = Boom()
    r = _post(client, {"message": "hi"})
    body = r.text.replace("\n\n", "\n")
    assert r.status_code == 200
    assert "event: error" in body
    assert "event: done" not in body


def test_chat_route_does_not_continue_another_users_conversation(client):
    """会话归属:拿别人的 conversation_id 只会新建自己的会话(session 帧回传另一个 id)。"""
    app.state.chat_model = RecordingModel(["第一轮:收到订单查询。"])
    r1 = _post(client, {"user_id": "u1", "message": "我的订单怎么了"})
    cid1 = dict(_parse_sse(r1.text))["session"]["conversation_id"]

    model2 = RecordingModel(["你好呀。"])
    app.state.chat_model = model2
    r2 = _post(client, {"user_id": "u2", "conversation_id": cid1, "message": "那能退货吗"})
    events2 = _parse_sse(r2.text)
    assert "session" in [e for e, _ in events2]
    cid2 = dict(events2)["session"]["conversation_id"]
    assert cid2 != cid1
    assert dict(events2)["done"]["conversation_id"] == cid2

    # u2 的模型输入里不得出现 u1 的任何内容
    contents = [getattr(m, "content", "") for m in model2.seen_inputs[0]]
    assert "我的订单怎么了" not in contents and "第一轮:收到订单查询。" not in contents


def test_chat_route_rejects_blank_message(client):
    r = _post(client, {"message": "   "})
    assert r.status_code == 422


def test_chat_route_ignores_legacy_session_id(client):
    """ch01 的 session_id 已退役:带了也不当会话标识用,服务端只看 conversation_id 决定新建与否。"""
    app.state.chat_model = RecordingModel(["喵。"])
    r1 = _post(client, {"session_id": "demo-1", "message": "在吗"})
    events1 = _parse_sse(r1.text)
    assert "session" in [e for e, _ in events1]
    assert isinstance(dict(events1)["session"]["conversation_id"], int)

    # 复用同一 session_id、不带 conversation_id → 仍视为新会话(session_id 不参与寻址)
    app.state.chat_model = RecordingModel(["喵。"])
    r2 = _post(client, {"session_id": "demo-1", "message": "还在吗"})
    assert "session" in [e for e, _ in _parse_sse(r2.text)]


def test_build_registry_wires_ops_tools_and_non_retryable_create_ticket():
    """装配契约:已知 conversation_id 才装 ops 工具;create_ticket 登记为不可重试(见 ruling)。

    非重试行为本身由 tests/test_tools_registry.py 黑盒覆盖,这里只钉住装配时的接线。
    """
    from app.routers.chat import build_registry

    without_cid = build_registry(session_factory=object(), conversation_id=None)
    assert "create_ticket" not in without_cid.names()

    with_cid = build_registry(session_factory=object(), conversation_id=7)
    assert set(with_cid.names()) == {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}
    assert "create_ticket" in with_cid._non_retryable
