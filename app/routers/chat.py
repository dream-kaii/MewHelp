from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.chat import ChatService
from app.config import get_settings
from app.db.base import get_sessionmaker
from app.llm import get_chat_model
from app.prompts import format_chat_system_prompt
from app.schemas import ChatRequest
from app.tools.business import query_logistics, query_order, query_product
from app.tools.kb import make_kb_tools
from app.tools.ops import make_ops_tools
from app.tools.registry import ToolRegistry

router = APIRouter()


def build_registry(session_factory, conversation_id: int | None) -> ToolRegistry:
    tools = [query_order, query_product, query_logistics, *make_kb_tools(session_factory)]
    if conversation_id is not None:
        tools += make_ops_tools(session_factory, conversation_id=conversation_id)
    # create_ticket 写 MySQL,而超时无法取消线程里的同步工具 → 重试会落两张工单
    return ToolRegistry(tools, non_retryable={"create_ticket"})


def resolve_chat_service(request: Request) -> ChatService:
    model = getattr(request.app.state, "chat_model", None) or get_chat_model()
    s = get_settings()
    return ChatService(
        model=model,
        registry_factory=lambda cid: build_registry(get_sessionmaker(), cid),
        session_factory=get_sessionmaker(),
        system_prompt=format_chat_system_prompt(s),
        settings=s,
    )


@router.post("/api/chat", response_class=EventSourceResponse)
async def chat(
    body: ChatRequest, service: ChatService = Depends(resolve_chat_service)
) -> AsyncIterator[ServerSentEvent]:
    # FastAPI 0.141 的 SSE 编码在路由层:endpoint 必须是 async generator +
    # response_class=EventSourceResponse,逐事件 yield ServerSentEvent。
    async for event in service.stream_turn(body.user_id, body.conversation_id, body.message):
        yield event
