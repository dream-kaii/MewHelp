import asyncio
import logging

from fastapi.sse import ServerSentEvent

from app.agent.tool_runner import execute_tool_calls, stream_first_round
from app.config import Settings
from app.context import to_langchain_messages, trim_to_budget
from app.db import repository as repo

logger = logging.getLogger("mewhelp.chat")


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
    """单轮两段式:①带工具流式跑第一段(顺带吐文本)②需要时用不绑工具的模型收敛流式。

    历史以 messages 表为准;每次 stream_turn 一个会话一轮,user/assistant/tool 全程落库。
    """

    def __init__(self, model, registry_factory, session_factory, system_prompt: str, settings: Settings):
        self._model = model
        self._registry_factory = registry_factory
        self._session_factory = session_factory
        self._system_prompt = system_prompt
        self._settings = settings

    def _build_lc_messages(self, history: list[dict]) -> list:
        """messages 表历史 → 裁到 token 预算内 → LangChain 消息(system 置顶;当前 user 已在 history 尾部)。"""
        rows = [
            {
                "role": h["role"],
                "content": h["content"],
                **({"tool_calls": h["tool_calls"]} if h.get("tool_calls") else {}),
                **({"tool_call_id": h["tool_call_id"]} if h.get("tool_call_id") else {}),
            }
            for h in history
        ]
        body = trim_to_budget(rows, self._settings.history_budget_tokens)
        return to_langchain_messages([{"role": "system", "content": self._system_prompt}, *body])

    async def _stream_text(self, messages):
        """用未绑定工具的模型流式产出文本增量(收敛段 / 纯聊天段共用)。"""
        async for chunk in self._model.astream(messages):
            text = text_of_chunk(chunk)
            if text:
                yield text

    async def stream_turn(self, user_id: str, conversation_id: int | None, user_text: str):
        async with self._session_factory() as session:
            conv = await repo.get_or_create_conversation(
                session, conversation_id=conversation_id, user_id=user_id
            )
            created = conversation_id is None or conv.id != conversation_id
            await repo.append_message(session, conversation_id=conv.id, role="user", content=user_text)
            await session.commit()
            cid = conv.id

        registry = self._registry_factory(cid)  # 该轮工具集(create_ticket 需要 cid,首轮也可用)

        if created:
            yield ServerSentEvent(event="session", data={"conversation_id": cid})

        try:
            async with self._session_factory() as session:
                lc_messages = self._build_lc_messages(await repo.load_history(session, cid))

            bindable = registry.bindable()
            tool_calls: list[dict] = []
            first_text = ""
            if bindable:
                # ── 第一段:绑定工具,唯一一次工具往返 ──
                first_parts: list[str] = []
                tool_calls, first_text = await stream_first_round(
                    self._model.bind_tools(bindable), lc_messages, first_parts.append
                )
                if first_text:
                    yield ServerSentEvent(event="delta", data={"content": first_text})
                    await asyncio.sleep(0)

            final_text: str | None = None
            if tool_calls:
                # on_status 由 gather 内的协程直接调用 → 绝不抛异常、不做 I/O,只往列表里塞帧
                status_frames: list[ServerSentEvent] = []

                def on_status(phase, call_id, name, args, summary):
                    data = {"call_id": call_id, "name": name, "status": phase}
                    if phase == "running":
                        data["args"] = args
                    else:
                        data["summary"] = (summary or "")[:200]
                    status_frames.append(ServerSentEvent(event="tool", data=data))

                results = await execute_tool_calls(
                    registry, tool_calls, on_status,
                    timeout=self._settings.tool_timeout_seconds,
                    retries=self._settings.tool_max_retries,
                )
                # 帧序不保证等于调用序(并行完成),只有 results 的顺序与入参一致
                for frame in status_frames:
                    yield frame

                async with self._session_factory() as session:
                    await repo.append_message(
                        session, conversation_id=cid, role="assistant", content=None,
                        tool_calls=[{"id": c["id"], "name": c["name"], "args": c["args"]} for c in tool_calls],
                    )
                    for r in results:
                        await repo.append_message(
                            session, conversation_id=cid, role="tool",
                            content=r.content if r.ok else f"[工具失败] {r.error}",
                            tool_call_id=r.call_id,
                        )
                    if any(r.name == "create_ticket" and r.ok for r in results):
                        await repo.set_conversation_status(session, cid, "已转人工")
                    await session.commit()
                    converge_messages = self._build_lc_messages(await repo.load_history(session, cid))
            elif bindable:
                # 模型第一段就用文本作答(未调工具):该文本即最终答复,无需二次收敛
                final_text = first_text
            else:
                # 无工具(纯聊天路径):直接流式,保持 ch01 行为
                converge_messages = lc_messages

            if final_text is None:
                # ── 第二段:不绑工具 → 物理上不可能再调工具,必然收敛 ──
                final_parts: list[str] = []
                async for text in self._stream_text(converge_messages):
                    final_parts.append(text)
                    yield ServerSentEvent(event="delta", data={"content": text})
                    await asyncio.sleep(0)  # 让出事件循环,便于取消
                final_text = "".join(final_parts)

            if final_text:
                async with self._session_factory() as session:
                    await repo.append_message(session, conversation_id=cid, role="assistant", content=final_text)
                    await session.commit()
        except Exception:
            logger.exception("chat stream error conversation=%s", cid)
            yield ServerSentEvent(event="error", data={"message": "抱歉,服务暂时不可用,请稍后重试"})
            return

        yield ServerSentEvent(event="done", data={"conversation_id": cid, "finish": True})
