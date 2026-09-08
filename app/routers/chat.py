from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.chat import ChatService
from app.config import get_settings
from app.llm import get_chat_model
from app.prompts import format_chat_system_prompt
from app.schemas import ChatRequest
from app.sessions import get_store

router = APIRouter()


def resolve_chat_service(request: Request) -> ChatService:
    model = getattr(request.app.state, "chat_model", None)
    if model is None:
        model = get_chat_model()
    s = get_settings()
    return ChatService(
        model=model,
        store=get_store(),
        system_prompt=format_chat_system_prompt(s),
        settings=s,
    )


@router.post("/api/chat", response_class=EventSourceResponse)
async def chat(
    body: ChatRequest, service: ChatService = Depends(resolve_chat_service)
) -> AsyncIterator[ServerSentEvent]:
    # FastAPI 0.141 的 SSE 编码在路由层:endpoint 必须是 async generator +
    # response_class=EventSourceResponse,逐事件 yield ServerSentEvent。
    async for event in service.stream_turn(body.session_id, body.message):
        yield event
