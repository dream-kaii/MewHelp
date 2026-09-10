import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Faq, Message, Ticket


async def get_or_create_conversation(
    session: AsyncSession, *, conversation_id: int | None, user_id: str
) -> Conversation:
    if conversation_id is not None:
        # 会话 id 是可猜的自增整数 → 必须按归属过滤,别人的 id 一律视为新会话
        existing = (
            await session.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id, Conversation.user_id == user_id
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing
    conv = Conversation(user_id=user_id)
    session.add(conv)
    await session.flush()
    # status/created_at 由 server_default 生成,flush 后仍是未加载状态;
    # 同步读取会触发 MissingGreenlet,这里显式 refresh 加载服务端默认值
    await session.refresh(conv)
    return conv


async def append_message(
    session: AsyncSession,
    *,
    conversation_id: int,
    role: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
) -> None:
    session.add(
        Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
    )
    await session.flush()


async def load_history(session: AsyncSession, conversation_id: int) -> list[dict]:
    rows = (
        await session.execute(
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
        )
    ).scalars().all()
    return [
        {"role": r.role, "content": r.content, "tool_calls": r.tool_calls, "tool_call_id": r.tool_call_id}
        for r in rows
    ]


async def search_faq(session: AsyncSession, keyword: str, limit: int = 3) -> list[dict]:
    kw = (keyword or "").strip()
    if not kw:
        return []
    rows = (
        await session.execute(
            select(Faq)
            .where(Faq.question.like(f"%{kw}%") | Faq.answer.like(f"%{kw}%"))
            .limit(limit)
        )
    ).scalars().all()
    return [{"question": r.question, "answer": r.answer, "category": r.category} for r in rows]


async def next_ticket_no(session: AsyncSession) -> str:
    day = datetime.date.today().strftime("%Y%m%d")
    prefix = f"T{day}"
    rows = (
        await session.execute(select(Ticket.ticket_no).where(Ticket.ticket_no.like(f"{prefix}%")))
    ).scalars().all()
    seq = max([int(x[len(prefix):] or 0) for x in rows], default=0) + 1
    return f"{prefix}{seq:03d}"


async def create_ticket(
    session: AsyncSession, *, conversation_id: int, description: str, ticket_type: str
) -> str:
    last_err: Exception | None = None
    for _ in range(5):
        no = await next_ticket_no(session)
        try:
            # 用 SAVEPOINT 包住插入:撞号只回滚到保存点,
            # 不丢掉调用方在同一 session 里的其它未提交写入
            async with session.begin_nested():
                session.add(
                    Ticket(
                        ticket_no=no,
                        conversation_id=conversation_id,
                        description=description,
                        ticket_type=ticket_type,
                    )
                )
                await session.flush()
            return no
        except IntegrityError as exc:  # 并发撞号,重试
            last_err = exc
    raise RuntimeError(f"生成工单号失败: {last_err}")


async def set_conversation_status(session: AsyncSession, conversation_id: int, status: str) -> None:
    conv = await session.get(Conversation, conversation_id)
    if conv is not None:
        conv.status = status
        await session.flush()
