import datetime

import pytest

from app.db import repository as repo

pytestmark = pytest.mark.anyio


async def test_get_or_create_conversation_creates_and_reuses(db_session):
    c = await repo.get_or_create_conversation(db_session, conversation_id=None, user_id="u1")
    await db_session.commit()
    assert c.id and c.user_id == "u1" and c.status == "进行中"
    same = await repo.get_or_create_conversation(db_session, conversation_id=c.id, user_id="u1")
    assert same.id == c.id


async def test_append_and_load_history_roundtrip(db_session):
    c = await repo.get_or_create_conversation(db_session, conversation_id=None, user_id="u")
    await repo.append_message(db_session, conversation_id=c.id, role="user", content="你好")
    await repo.append_message(
        db_session, conversation_id=c.id, role="assistant", content=None,
        tool_calls=[{"id": "call_1", "name": "query_order", "args": {"order_id": "1001"}}],
    )
    await repo.append_message(
        db_session, conversation_id=c.id, role="tool", content="订单已发货", tool_call_id="call_1"
    )
    await db_session.commit()
    hist = await repo.load_history(db_session, c.id)
    assert [m["role"] for m in hist] == ["user", "assistant", "tool"]
    assert hist[1]["tool_calls"][0]["name"] == "query_order"
    assert hist[2]["tool_call_id"] == "call_1"


async def test_search_faq_matches_keyword_else_empty(db_session):
    from app.db.models import Faq

    db_session.add_all([
        Faq(question="退货政策是什么", answer="7 天无理由退货", category="售后"),
        Faq(question="发票怎么开", answer="下单时勾选", category="发票"),
    ])
    await db_session.commit()
    hits = await repo.search_faq(db_session, "退货")
    assert hits and "无理由" in hits[0]["answer"]
    assert await repo.search_faq(db_session, "邮费") == []


async def test_create_ticket_returns_no_and_sets_status(db_session):
    c = await repo.get_or_create_conversation(db_session, conversation_id=None, user_id="u")
    await db_session.commit()
    no = await repo.create_ticket(db_session, conversation_id=c.id, description="要人工", ticket_type="售后")
    await db_session.commit()
    assert no.startswith("T" + datetime.date.today().strftime("%Y%m%d"))
    await repo.set_conversation_status(db_session, c.id, "已转人工")
    await db_session.commit()
    fresh = await repo.get_or_create_conversation(db_session, conversation_id=c.id, user_id="u")
    assert fresh.status == "已转人工"
