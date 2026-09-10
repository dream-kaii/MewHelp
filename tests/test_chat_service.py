import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from app.chat import ChatService, text_of_chunk
from app.config import Settings
from app.context import total_tokens
from app.db import repository as repo
from app.tools.registry import ToolRegistry

pytestmark = pytest.mark.anyio

SETTINGS = Settings(history_budget_tokens=2048, tool_timeout_seconds=2, tool_max_retries=0)


class RecordingChatModel:
    """包装 GenericFakeChatModel,在 astream 前记录收到的消息列表,用于断言上下文是否传入模型。

    (1.x 里 GenericFakeChatModel 是 pydantic 模型,子类不能随意加未知字段,故用组合而非继承。)
    """

    def __init__(self, responses):
        self._inner = GenericFakeChatModel(messages=iter(responses))
        self.seen_inputs: list[list] = []

    async def astream(self, messages, **kwargs):
        self.seen_inputs.append(list(messages))
        async for chunk in self._inner.astream(messages, **kwargs):
            yield chunk


class FailingChatModel:
    async def astream(self, messages, **kwargs):
        raise RuntimeError("upstream down")
        yield  # pragma: no cover


def _service(model, session_factory) -> ChatService:
    """构造无工具(纯聊天路径)的 ChatService:历史落 DB,模型输入即 messages 表内容。"""
    return ChatService(
        model=model,
        registry_factory=lambda cid: ToolRegistry([]),
        session_factory=session_factory,
        system_prompt="你是客服",
        settings=SETTINGS,
    )


async def _collect(agen):
    return [e async for e in agen]


def _text_of(events):
    return "".join(e.data.get("content", "") for e in events if e.event == "delta")


class _Chunk:
    def __init__(self, content):
        self.content = content


def test_text_of_chunk_uses_text_property_first():
    class ChunkWithText:
        content = [{"type": "text", "text": "你"}, {"type": "text", "text": "好"}]

        @property
        def text(self):
            return "你好"

    assert text_of_chunk(ChunkWithText()) == "你好"


def test_text_of_chunk_handles_str_and_blocks_when_no_text():
    assert text_of_chunk(_Chunk("你好")) == "你好"
    assert text_of_chunk(_Chunk([{"type": "text", "text": "你"}, {"type": "text", "text": "好"}])) == "你好"
    assert text_of_chunk(_Chunk(None)) == ""


async def test_new_conversation_emits_session_delta_done_and_persists(session_factory, db_session):
    model = RecordingChatModel(["我是客服,已收到。"])
    events = await _collect(_service(model, session_factory).stream_turn("u1", None, "你好"))
    assert [e.event for e in events] == ["session", "delta", "done"]
    cid = events[0].data["conversation_id"]
    assert cid > 0 and events[-1].data["conversation_id"] == cid
    assert _text_of(events) == "我是客服,已收到。"

    # 历史以 messages 表为准(user + assistant 各一行)
    async with session_factory() as session:
        hist = await repo.load_history(session, cid)
    assert [(m["role"], m["content"]) for m in hist] == [
        ("user", "你好"),
        ("assistant", "我是客服,已收到。"),
    ]


async def test_second_turn_receives_history_and_skips_session_frame(session_factory, db_session):
    model = RecordingChatModel(["第一轮回复", "第二轮回复"])
    svc = _service(model, session_factory)

    first = await _collect(svc.stream_turn("u1", None, "第一问"))
    cid = first[0].data["conversation_id"]
    second = await _collect(svc.stream_turn("u1", cid, "第二问"))

    assert [e.event for e in first] == ["session", "delta", "done"]
    # 既有 conversation_id → 不再发 session 帧
    assert [e.event for e in second] == ["delta", "done"]

    inputs = model.seen_inputs
    assert len(inputs) == 2
    roles = [type(m).__name__ for m in inputs[1]]
    assert roles == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]
    contents = [getattr(m, "content", "") for m in inputs[1]]
    assert "第一问" in contents and "第一轮回复" in contents and "第二问" in contents


async def test_model_error_emits_error_and_no_assistant_append(session_factory, db_session):
    async with session_factory() as session:
        conv = await repo.get_or_create_conversation(session, conversation_id=None, user_id="u1")
        await session.commit()
        cid = conv.id

    events = await _collect(_service(FailingChatModel(), session_factory).stream_turn("u1", cid, "hi"))

    # 既有会话:不走 session 事件分支 → 精确只有 error,且不发 done
    assert [e.event for e in events] == ["error"]
    async with session_factory() as session:
        hist = await repo.load_history(session, cid)
    # 用户消息先落库(第一段之前),失败轮不留 assistant 行
    assert [m["role"] for m in hist] == ["user"]


async def test_tool_history_window_starts_with_user_and_never_orphans_tool(session_factory, db_session):
    """真实 chat 路径:带工具的历史(assistant.tool_calls + tool 行)跨预算裁剪后,
    送给模型的窗口必须以 Human 开头,不残留会触发上游 400 的孤儿 ToolMessage/AIMessage。"""
    async with session_factory() as session:
        conv = await repo.get_or_create_conversation(session, conversation_id=None, user_id="u1")
        await session.commit()
        cid = conv.id
        await repo.append_message(session, conversation_id=cid, role="user", content="订单1001到哪了" * 3)
        await repo.append_message(
            session, conversation_id=cid, role="assistant", content=None,
            tool_calls=[{"id": "c1", "name": "query_order", "args": {"order_id": "1001"}}],
        )
        await repo.append_message(session, conversation_id=cid, role="tool", content="已发货", tool_call_id="c1")
        await repo.append_message(session, conversation_id=cid, role="assistant", content="已发货,请留意" * 3)
        await session.commit()
        hist = await repo.load_history(session, cid)

    current = "那退款呢" * 3
    # 复刻 _build_lc_messages 的 token 口径,把预算刻在「旧算法会以 tool 行开头」的窗口上
    rows = [{"role": h["role"], "content": h["content"]} for h in hist]
    rows.append({"role": "user", "content": current})
    budget = total_tokens([{"role": "system", "content": "你是客服"}, *rows[2:]])

    model = RecordingChatModel(["好的"])
    svc = ChatService(
        model=model, registry_factory=lambda cid: ToolRegistry([]), session_factory=session_factory,
        system_prompt="你是客服",
        settings=Settings(history_budget_tokens=budget, tool_timeout_seconds=2, tool_max_retries=0),
    )
    events = await _collect(svc.stream_turn("u1", cid, current))
    assert events[-1].event == "done"

    roles = [type(m).__name__ for m in model.seen_inputs[0]]
    assert roles[0] == "SystemMessage"
    assert roles[1] == "HumanMessage"  # 旧实现此处为 ToolMessage
    assert "ToolMessage" not in roles and "AIMessage" not in roles
