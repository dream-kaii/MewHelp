import pytest

from app.tools.kb import make_kb_tools
from app.tools.ops import make_ops_tools

pytestmark = pytest.mark.anyio


async def test_query_faq_finds_seeded_row(session_factory, db_session):
    from app.db.models import Faq

    db_session.add(Faq(question="退货政策是什么", answer="7 天无理由退货", category="售后"))
    await db_session.commit()

    [query_faq] = make_kb_tools(session_factory)
    out = await query_faq.ainvoke({"keyword": "退货"})
    assert "7 天无理由退货" in out


async def test_query_faq_returns_explicit_miss(session_factory):
    [query_faq] = make_kb_tools(session_factory)
    out = await query_faq.ainvoke({"keyword": "邮费"})
    assert "未找到" in out  # 漏召回时给模型一个明确信号,而非空字符串


async def test_create_ticket_writes_row_and_returns_no(session_factory, db_session):
    from app.db import repository as repo
    from app.db.models import Ticket

    conv = await repo.get_or_create_conversation(db_session, conversation_id=None, user_id="u")
    await db_session.commit()

    [create_ticket] = make_ops_tools(session_factory, conversation_id=conv.id)
    out = await create_ticket.ainvoke({"description": "需要人工处理", "ticket_type": "售后"})
    assert "T" in out and "已创建" in out

    from sqlalchemy import select

    rows = (await db_session.execute(select(Ticket).where(Ticket.conversation_id == conv.id))).scalars().all()
    assert len(rows) == 1 and rows[0].description == "需要人工处理"
