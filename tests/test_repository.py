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


async def test_next_ticket_no_empty_then_increments(db_session):
    from app.db.models import Ticket

    day = datetime.date.today().strftime("%Y%m%d")
    assert await repo.next_ticket_no(db_session) == f"T{day}001"

    c = await repo.get_or_create_conversation(db_session, conversation_id=None, user_id="u")
    db_session.add(
        Ticket(ticket_no=f"T{day}001", conversation_id=c.id, description="占位", ticket_type="咨询")
    )
    await db_session.commit()
    assert await repo.next_ticket_no(db_session) == f"T{day}002"


async def test_create_ticket_skips_taken_number(db_session):
    from sqlalchemy import select

    from app.db.models import Ticket

    day = datetime.date.today().strftime("%Y%m%d")
    taken = f"T{day}001"
    c = await repo.get_or_create_conversation(db_session, conversation_id=None, user_id="u")
    await db_session.commit()
    db_session.add(
        Ticket(ticket_no=taken, conversation_id=c.id, description="占位", ticket_type="咨询")
    )
    await db_session.commit()

    no = await repo.create_ticket(db_session, conversation_id=c.id, description="要人工", ticket_type="售后")
    await db_session.commit()

    assert no != taken
    rows = (await db_session.execute(select(Ticket.ticket_no).order_by(Ticket.ticket_no))).scalars().all()
    assert rows == [taken, no]


async def test_create_ticket_retry_keeps_session_work(db_session, monkeypatch):
    """确定性触发 IntegrityError 重试分支,并断言 SAVEPOINT 语义:

    只回滚到保存点,不丢掉同一 session 里其它未提交写入(整事务回滚会丢掉)。
    """
    from sqlalchemy import func, select

    from app.db.models import Conversation, Ticket

    day = datetime.date.today().strftime("%Y%m%d")
    taken = f"T{day}001"
    c = await repo.get_or_create_conversation(db_session, conversation_id=None, user_id="u")
    await db_session.commit()
    db_session.add(
        Ticket(ticket_no=taken, conversation_id=c.id, description="占位", ticket_type="咨询")
    )
    await db_session.commit()

    # 同一 session 里的另一笔未提交写入:重试路径不得丢弃它
    pending = await repo.get_or_create_conversation(db_session, conversation_id=None, user_id="pending")

    # next_ticket_no 会看到已占用的号并返回下一个,单靠预置无法撞号;
    # 强制第一次调用返回被占用的号,从而确定性走进重试分支
    real_next = repo.next_ticket_no
    calls = {"n": 0}

    async def flaky_next(session):
        calls["n"] += 1
        return taken if calls["n"] == 1 else await real_next(session)

    monkeypatch.setattr(repo, "next_ticket_no", flaky_next)

    no = await repo.create_ticket(db_session, conversation_id=c.id, description="要人工", ticket_type="售后")
    await db_session.commit()

    assert calls["n"] >= 2  # 确实进入了重试循环
    assert no != taken and no == f"T{day}002"
    rows = (await db_session.execute(select(Ticket.ticket_no).order_by(Ticket.ticket_no))).scalars().all()
    assert rows == [taken, no]
    assert pending.id is not None
    conv_count = (await db_session.execute(select(func.count()).select_from(Conversation))).scalar()
    assert conv_count == 2  # c 与 pending 都在;整事务回滚只会剩 1

