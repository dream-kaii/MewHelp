import pytest
from sqlalchemy import text

from app.config import get_settings
from app.db import repository_knowledge as rk
from app.rag.chunker import chunk_markdown
from app.rag.embedder import FakeEmbedder
from app.rag.store import VectorStore
from scripts.build_knowledge import embed_pending, ingest_markdown

pytestmark = pytest.mark.anyio

DOC = "# 售后政策\n\n## 运费说明\n\n下单运费按地区收取。退货寄回的运费由责任方承担。\n"

DOC2 = (
    "# 退换货政策\n\n## 七天无理由\n\n自签收起七天可无理由退货。\n\n"
    "## 质量问题\n\n质量问题由商家承担来回运费。\n"
)


@pytest.fixture
def store():
    s = get_settings()
    vs = VectorStore(uri=s.milvus_uri, token=s.milvus_token, collection=s.milvus_test_collection)
    vs.drop()
    vs.ensure_collection()
    yield vs
    vs.drop()


async def test_ingest_writes_mysql_then_embeds(session_factory, db_session, store):
    stats = await ingest_markdown(session_factory, FakeEmbedder(), store, "policy.md", DOC)
    assert stats["inserted"] >= 1 and stats["embedded"] == stats["inserted"]
    assert await rk.list_pending(db_session) == []
    assert store.count() == stats["inserted"]


async def test_ingest_is_idempotent_on_rerun(session_factory, db_session, store):
    first = await ingest_markdown(session_factory, FakeEmbedder(), store, "policy.md", DOC)
    second = await ingest_markdown(session_factory, FakeEmbedder(), store, "policy.md", DOC)
    assert second["inserted"] == 0
    assert store.count() == first["inserted"]


async def test_interrupted_run_resumes_only_pending(session_factory, db_session, store):
    """验收②:模拟中断 —— 只写 MySQL 不向量化,再跑 embed_pending 补齐。"""
    await ingest_markdown(session_factory, FakeEmbedder(), store, "policy.md", DOC, skip_embed=True)
    pending = await rk.list_pending(db_session)
    assert pending, "应有未向量化的块"
    stats = await embed_pending(session_factory, FakeEmbedder(), store)
    assert stats["embedded"] == len(pending)
    # db_session 前面读过一次 pending,已在 MySQL REPEATABLE READ 下固定了快照,
    # 看不到 embed_pending(另一连接)的提交 —— 结束本事务后才能读到新状态。
    await db_session.commit()
    assert await rk.list_pending(db_session) == []
    assert store.count() == len(pending)


async def test_duplicate_content_hash_is_skipped_not_raised(session_factory, db_session, store, monkeypatch):
    """并发/重跑:同 content_hash 的行在本事务 SELECT 之后被另一连接提交。

    此时 insert_chunks 的「先 SELECT 再 INSERT」看不见它,flush 会抛 IntegrityError。
    建库必须吸收这次 UNIQUE 冲突(跳过该行)而不是崩掉,且后续 commit 仍能成功。
    """
    real_insert = rk.insert_chunks

    async def racing_insert(session, payload):
        # 1) 先真读一次本表,让本事务建立 REPEATABLE READ 快照
        #    (注意:MySQL 对 `SELECT 1` 这类不碰表的读并不置快照,必须读 knowledge_chunks)
        await session.execute(text("SELECT COUNT(*) FROM knowledge_chunks"))
        # 2) 另一连接抢先提交同 content_hash 的行
        async with session_factory() as other:
            await real_insert(other, list(payload))
            await other.commit()
        # 3) 本事务再走正版逻辑:SELECT 走快照 → 看不见 → INSERT 撞 UNIQUE
        return await real_insert(session, payload)

    monkeypatch.setattr(rk, "insert_chunks", racing_insert)

    stats = await ingest_markdown(session_factory, FakeEmbedder(), store, "policy.md", DOC, skip_embed=True)

    assert stats["inserted"] == 0, "重复行应被跳过,不算新增"
    assert await rk.list_pending(db_session) != [], "行已在库(由另一连接写入)"
    assert store.count() == 0


async def test_duplicate_row_is_skipped_while_rest_still_inserted(session_factory, db_session, store, monkeypatch):
    """只有一行撞 UNIQUE 时:该行跳过,同批其余行照常入库(保存点回滚不污染会话)。"""
    real_insert = rk.insert_chunks
    chunks = chunk_markdown("policy.md", DOC2)
    assert len(chunks) >= 2, "本用例要求多块文档"
    collided = False  # _insert_tolerant 逐行调用,这里只让第一行撞车

    async def racing_insert(session, payload):
        nonlocal collided
        await session.execute(text("SELECT COUNT(*) FROM knowledge_chunks"))
        if not collided:
            collided = True
            async with session_factory() as other:
                await real_insert(other, list(payload))  # 并发写入者只抢先提交了第一块
                await other.commit()
        return await real_insert(session, payload)

    monkeypatch.setattr(rk, "insert_chunks", racing_insert)

    stats = await ingest_markdown(session_factory, FakeEmbedder(), store, "policy.md", DOC2, skip_embed=True)

    assert stats["inserted"] == len(chunks) - 1, "除撞车的块外都应入库"
    assert len(await rk.list_pending(db_session)) == len(chunks)
    assert store.count() == 0
