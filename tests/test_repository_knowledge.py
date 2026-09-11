import pytest

from app.db import repository_knowledge as rk

pytestmark = pytest.mark.anyio


def _chunk(**kw):
    base = dict(
        doc_id="policy.md", category="售后政策", questions="退货政策",
        answer="7 天无理由退货。", text="售后政策\n退货政策\n7 天无理由退货。",
        section_path="售后政策 > 退货政策", content_type="政策",
        is_key_clause=False, order_index=0,
    )
    base.update(kw)
    base["content_hash"] = rk.content_hash(base["text"])
    return base


async def test_insert_chunks_is_idempotent_by_hash(db_session):
    ids1 = await rk.insert_chunks(db_session, [_chunk()])
    await db_session.commit()
    ids2 = await rk.insert_chunks(db_session, [_chunk()])  # 同内容再插
    await db_session.commit()
    assert ids2 == [] and len(ids1) == 1
    pending = await rk.list_pending(db_session)
    assert [c["id"] for c in pending] == ids1


async def test_mark_embedded_removes_from_pending(db_session):
    ids = await rk.insert_chunks(db_session, [_chunk()])
    await db_session.commit()
    await rk.mark_embedded(db_session, ids)
    await db_session.commit()
    assert await rk.list_pending(db_session) == []
    rows = await rk.fetch_by_ids(db_session, ids)
    assert rows[0]["vector_id"] == str(ids[0]) and rows[0]["status"] == "embedded"


async def test_link_neighbors_sets_pointers(db_session):
    ids = await rk.insert_chunks(
        db_session, [_chunk(text="A", content_hash=""), _chunk(text="B", content_hash="")]
    )
    await db_session.commit()
    await rk.link_neighbors(db_session, ids)
    await db_session.commit()
    rows = await rk.fetch_by_ids(db_session, ids)
    assert rows[0]["next_id"] == ids[1] and rows[1]["prev_id"] == ids[0]


async def test_fetch_by_ids_preserves_input_order(db_session):
    ids = await rk.insert_chunks(
        db_session, [_chunk(text="A"), _chunk(text="B"), _chunk(text="C")]
    )
    await db_session.commit()
    rows = await rk.fetch_by_ids(db_session, list(reversed(ids)))
    assert [r["id"] for r in rows] == list(reversed(ids))


async def test_staging_insert_dedupes_and_marks(db_session):
    items = [dict(conversation_id=1, source_message_ids="1,2", questions='["邮费怎么算"]',
                  answer="按地区收取", category="运费", dedupe_hash=rk.content_hash("邮费怎么算|按地区收取"))]
    assert await rk.staging_insert(db_session, items) == 1
    await db_session.commit()
    assert await rk.staging_insert(db_session, items) == 0  # 幂等
    staged = await rk.staging_list(db_session)
    assert len(staged) == 1
    await rk.staging_mark(db_session, [staged[0]["id"]], "promoted")
    await db_session.commit()
    assert await rk.staging_list(db_session) == []
