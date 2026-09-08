import asyncio
import logging

from fastapi.sse import ServerSentEvent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import Settings
from app.context import build_messages
from app.sessions import SessionStore

logger = logging.getLogger("mewhelp.chat")

_ROLE_TO_CLS = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


def to_langchain_messages(messages: list[dict]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in messages:
        cls = _ROLE_TO_CLS.get(m.get("role"), HumanMessage)
        out.append(cls(m.get("content") or ""))
    return out


def text_of_chunk(chunk) -> str:
    """从 AIMessageChunk 提取文本增量。

    1.x 优先用 .text 投影(自动合并 str / content blocks);退化分支兼容只有 .content 的对象。
    """
    text = getattr(chunk, "text", None)
    if isinstance(text, str) and text:
        return text
    content = getattr(chunk, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "".join(parts)
    return str(content)


class ChatService:
    def __init__(self, model, store: SessionStore, system_prompt: str, settings: Settings):
        self._model = model
        self._store = store
        self._system_prompt = system_prompt
        self._settings = settings

    async def stream_turn(self, session_id: str | None, user_text: str):
        session, created = self._store.get_or_create(session_id)
        if created:
            yield ServerSentEvent(event="session", data={"session_id": session.session_id})

        messages = build_messages(
            self._system_prompt,
            session.turns,
            user_text,
            self._settings.history_budget_tokens,
        )
        collected: list[str] = []
        try:
            async for chunk in self._model.astream(to_langchain_messages(messages)):
                text = text_of_chunk(chunk)
                if not text:
                    continue
                collected.append(text)
                yield ServerSentEvent(event="delta", data={"content": text})
                await asyncio.sleep(0)  # 让出事件循环,便于取消
        except Exception:
            logger.exception("chat stream error session=%s", session.session_id)
            yield ServerSentEvent(event="error", data={"message": "抱歉,服务暂时不可用,请稍后重试"})
            return

        if collected:
            self._store.append_turn(session.session_id, user_text, "".join(collected))
        yield ServerSentEvent(event="done", data={"finish": True})
