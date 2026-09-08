import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from app.chat import ChatService, text_of_chunk
from app.config import Settings
from app.sessions import SessionStore

pytestmark = pytest.mark.anyio

SETTINGS = Settings(history_budget_tokens=2048, session_max_turns=30, session_max_count=50)


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


async def test_new_session_emits_session_delta_done_and_appends():
    model = RecordingChatModel(["我是客服,已收到。"])
    store = SessionStore(max_turns=10, max_sessions=10)
    svc = ChatService(model=model, store=store, system_prompt="你是客服", settings=SETTINGS)
    events = await _collect(svc.stream_turn(None, "你好"))
    assert [e.event for e in events] == ["session", "delta", "done"]
    sid = events[0].data["session_id"]
    assert "已收到" in _text_of(events)
    turns = store.get(sid).turns
    assert turns == [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "我是客服,已收到。"}]


async def test_second_turn_receives_history_in_messages():
    model = RecordingChatModel(["第一轮回复", "第二轮回复"])
    store = SessionStore(max_turns=10, max_sessions=10)
    svc = ChatService(model=model, store=store, system_prompt="你是客服", settings=SETTINGS)
    store.get_or_create("sess-1")
    await _collect(svc.stream_turn("sess-1", "第一问"))
    await _collect(svc.stream_turn("sess-1", "第二问"))
    inputs = model.seen_inputs
    assert len(inputs) == 2
    roles = [type(m).__name__ for m in inputs[1]]
    assert roles == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]
    contents = [getattr(m, "content", "") for m in inputs[1]]
    assert "第一问" in contents and "第一轮回复" in contents and "第二问" in contents


async def test_model_error_emits_error_and_no_append():
    model = FailingChatModel()
    store = SessionStore(max_turns=10, max_sessions=10)
    store.get_or_create("bad")  # 既有会话:不走 session 事件分支,events 精确等于 ["error"]
    svc = ChatService(model=model, store=store, system_prompt="你是客服", settings=SETTINGS)
    events = await _collect(svc.stream_turn("bad", "hi"))
    assert [e.event for e in events] == ["error"]
    assert store.get("bad").turns == []
