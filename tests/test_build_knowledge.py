import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.config import get_settings
from app.db import repository_knowledge as rk
from app.rag.chunker import chunk_markdown
from app.rag.embedder import FakeEmbedder
from app.rag.store import VectorStore
from scripts import build_knowledge as bk
from scripts.build_knowledge import embed_pending, ingest_markdown

pytestmark = pytest.mark.anyio

DOC = "# 售后政策\n\n## 运费说明\n\n下单运费按地区收取。退货寄回的运费由责任方承担。\n"

DOC2 = (
    "# 退换货政策\n\n## 七天无理由\n\n自签收起七天可无理由退货。\n\n"
    "## 质量问题\n\n质量问题由商家承担来回运费。\n"
)

# 三段正文 → 三块(用于需要"多条 pending"的用例)
DOC3 = (
    DOC2
    + "\n## 运费\n\n质量问题的退货运费由商家承担,无理由退货运费由买家承担。\n"
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


async def test_non_duplicate_integrity_error_is_not_swallowed(session_factory, db_session, store, monkeypatch):
    """非重复类的完整性错误必须上抛。

    容错只能收窄到「content_hash 撞 UNIQUE」这一种情形;把 NOT NULL / 其它约束错误
    也当重复跳过,会**无声丢块**、`inserted` 少计,且归因完全错误。
    """
    real_chunk_markdown = bk.chunk_markdown

    def null_questions(doc_id, markdown, **kw):
        # questions 是 NOT NULL → 制造一个与重复内容无关的真实完整性错误(1048)
        chunks = real_chunk_markdown(doc_id, markdown, **kw)
        chunks[0].questions = None
        return chunks

    monkeypatch.setattr(bk, "chunk_markdown", null_questions)

    with pytest.raises(IntegrityError):
        await ingest_markdown(session_factory, FakeEmbedder(), store, "policy.md", DOC3, skip_embed=True)

    assert await rk.list_pending(db_session) == [], "整批失败,不得留下半截数据"
    assert store.count() == 0


async def test_zero_or_negative_limit_is_not_unlimited(session_factory, db_session, store):
    """`--limit 0` / 负数不能退化成「无限制」而把积压全量向量化。"""
    await ingest_markdown(session_factory, FakeEmbedder(), store, "policy.md", DOC3, skip_embed=True)

    assert (await embed_pending(session_factory, FakeEmbedder(), store, limit=0))["embedded"] == 0
    assert (await embed_pending(session_factory, FakeEmbedder(), store, limit=-3))["embedded"] == 0
    assert store.count() == 0
    assert len(await rk.list_pending(db_session)) == 3


async def test_cli_limit_embeds_only_part_of_backlog(session_factory, db_session, store, monkeypatch, tmp_path):
    """CLI `--limit N` 必须真的只向量化 N 块。

    旧实现把 `--limit` 解析了却从不传下去 → 面对积压会**全量**向量化且无提示,
    与计划里「分段执行」的用途相反。
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "policy.md").write_text(DOC3, encoding="utf-8")

    # 先造 3 条 pending 积压(不向量化)
    await ingest_markdown(session_factory, FakeEmbedder(), store, "policy.md", DOC3, skip_embed=True)
    assert len(await rk.list_pending(db_session)) == 3

    # 让 CLI 走测试集合 + 假向量器(main 自己构造 store/embedder)
    monkeypatch.setattr(bk, "get_embedder", lambda s: FakeEmbedder())
    monkeypatch.setattr(bk, "VectorStore", lambda **kw: store)

    rc = await bk.main(["--dir", str(docs_dir), "--limit", "1"])
    await db_session.commit()  # main 用的是另一个连接,换掉本会话的快照

    assert rc == 0
    assert store.count() == 1, "只应向量化 1 块"
    assert len(await rk.list_pending(db_session)) == 2, "其余仍 pending,留给下次重跑"

    # `--limit 0` 同理:一块都不向量化,不能变成"无限制"
    rc0 = await bk.main(["--dir", str(docs_dir), "--limit", "0"])
    await db_session.commit()
    assert rc0 == 0
    assert store.count() == 1
    assert len(await rk.list_pending(db_session)) == 2


async def test_cli_empty_doc_set_returns_2(monkeypatch, tmp_path):
    """空文档集:提示到 stderr + 退出码 2,不静默成功。"""
    monkeypatch.setattr(bk, "get_embedder", lambda s: FakeEmbedder())
    monkeypatch.setattr(bk, "VectorStore", lambda **kw: object())

    rc = await bk.main(["--dir", str(tmp_path / "missing_dir")])

    assert rc == 2
