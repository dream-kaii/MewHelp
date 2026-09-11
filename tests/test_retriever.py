import logging

import pytest

from app.config import get_settings
from app.db import repository_knowledge as rk
from app.rag.embedder import FakeEmbedder
from app.rag.retriever import KnowledgeRetriever

pytestmark = pytest.mark.anyio


class StubStore:
    def __init__(self, hits):
        self._hits = hits
        self.upserted = None

    def ensure_collection(self): ...
    def upsert(self, ids, vectors): self.upserted = (ids, vectors)
    def search(self, vector, top_k=5): return self._hits[:top_k]
    def count(self): return len(self._hits)
    def delete(self, ids): return 0
    def drop(self): ...


async def _seed(db_session):
    ids = await rk.insert_chunks(
        db_session,
        [dict(doc_id="policy.md", category="售后政策", questions="运费说明",
              answer="退货寄回的运费由责任方承担。", text="售后政策\n运费说明\n退货寄回的运费由责任方承担。",
              section_path="售后政策 > 运费说明", content_type="政策", is_key_clause=False)],
    )
    await db_session.commit()
    return ids


async def test_search_returns_answer_text_on_hit(session_factory, db_session):
    ids = await _seed(db_session)
    r = KnowledgeRetriever(FakeEmbedder(), StubStore([(ids[0], 0.88)]), session_factory, top_k=5, score_threshold=0.5)
    out = await r.search("邮费是多少")
    assert "责任方承担" in out and "未找到" not in out


async def test_search_returns_not_found_below_threshold(session_factory, db_session):
    """边界:低于阈值属「干净未命中」,不得改走关键词/SQL LIKE 召回(spec §12:本章只跑 dense 单路)。

    特意在 faq 表塞一条**能命中该关键词**的行 —— 降级一旦写宽(把无命中也算失败),这里就会
    把关键词结果捞回来,用例失败。这就是"关键词成了正式检索路径"的回归探针。
    """
    from app.db.models import Faq

    ids = await _seed(db_session)
    db_session.add(Faq(question="邮费是多少", answer="只有关键词路径才会给出的答案", category="售后"))
    await db_session.commit()

    r = KnowledgeRetriever(FakeEmbedder(), StubStore([(ids[0], 0.10)]), session_factory, top_k=5, score_threshold=0.5)
    out = await r.search("邮费是多少")
    assert "未找到" in out
    assert "只有关键词路径才会给出的答案" not in out


async def test_search_returns_not_found_on_empty_store(session_factory):
    r = KnowledgeRetriever(FakeEmbedder(), StubStore([]), session_factory, top_k=5, score_threshold=0.5)
    assert "未找到" in await r.search("任何问题")


async def test_query_faq_uses_retriever_and_keeps_contract(session_factory, db_session):
    ids = await _seed(db_session)
    from app.tools.kb import make_kb_tools

    r = KnowledgeRetriever(FakeEmbedder(), StubStore([(ids[0], 0.9)]), session_factory, top_k=5, score_threshold=0.5)
    [query_faq] = make_kb_tools(session_factory, retriever=r)
    out = await query_faq.ainvoke({"keyword": "邮费是多少"})
    assert isinstance(out, str) and "责任方承担" in out


# --- 降级边界:只有向量路径**抛服务异常**才回退 ch02 关键词检索;干净未命中不回退 ---


class RaisingStore(StubStore):
    """Milvus 不可达 / 集合不存在 —— 真实形态是 MilvusException,属服务类异常。"""

    def search(self, vector, top_k=5):
        from pymilvus.exceptions import MilvusException

        raise MilvusException(2, "Fail connecting to server on 127.0.0.1:19530")


async def test_store_failure_falls_back_to_keyword_search(session_factory, db_session):
    from app.db.models import Faq

    db_session.add(Faq(question="邮费怎么算", answer="满 99 包邮", category="售后"))
    await db_session.commit()

    r = KnowledgeRetriever(FakeEmbedder(), RaisingStore([]), session_factory, top_k=5, score_threshold=0.5)
    out = await r.search("邮费")  # 向量路径炸了,但关键词路径能命中
    assert "满 99 包邮" in out and "未找到" not in out


async def test_both_paths_miss_returns_not_found_without_raising(session_factory, db_session):
    r = KnowledgeRetriever(FakeEmbedder(), RaisingStore([]), session_factory, top_k=5, score_threshold=0.5)
    out = await r.search("两路都查不到的问题 zzz")
    assert isinstance(out, str) and "未找到" in out  # 不抛出


async def test_vector_hit_with_deleted_rows_warns_and_stays_on_vector_path(session_factory, db_session, caplog):
    """向量命中但 MySQL 行已删(fetch_by_ids 少返行):记 warning、返回「未找到」,**不回退关键词**。"""
    from app.db.models import Faq

    db_session.add(Faq(question="邮费是多少", answer="只有关键词路径才会给出的答案", category="售后"))
    await db_session.commit()

    r = KnowledgeRetriever(FakeEmbedder(), StubStore([(999, 0.9)]), session_factory, top_k=5, score_threshold=0.5)
    with caplog.at_level(logging.WARNING, logger="mewhelp.rag.retriever"):
        out = await r.search("邮费是多少")
    assert "未找到" in out
    assert "只有关键词路径才会给出的答案" not in out
    assert any("知识行已不存在" in m for m in caplog.messages), caplog.messages


async def test_internal_programming_error_propagates(session_factory, db_session, monkeypatch):
    """行格式化抛的编程错误(KeyError)不能被降级吞掉 —— 否则真 bug 被静默掩盖成「未找到」。"""
    from app.db import repository_knowledge as _rk
    from app.db.models import Faq

    ids = await _seed(db_session)
    db_session.add(Faq(question="邮费是多少", answer="只有关键词路径才会给出的答案", category="售后"))
    await db_session.commit()

    async def _bad_fetch(session, got_ids):  # 缺 'questions'/'answer' → 格式化时 KeyError
        return [{"id": got_ids[0]}]

    monkeypatch.setattr(_rk, "fetch_by_ids", _bad_fetch)
    r = KnowledgeRetriever(FakeEmbedder(), StubStore([(ids[0], 0.9)]), session_factory, top_k=5, score_threshold=0.5)
    with pytest.raises(KeyError):
        await r.search("邮费是多少")


# --- 进程级复用:embedder / store 不能每轮重建(否则每轮重载 2.27GB 权重) ---


def test_shared_runtime_objects_are_process_level_singletons():
    from app.rag.embedder import get_embedder
    from app.rag.runtime import get_shared_embedder, get_shared_store

    s = get_settings()
    # 旧路径:每次 get_embedder() 都是新实例 → 实例级懒加载的权重每轮重载
    assert get_embedder(s) is not get_embedder(s)
    # 新路径:进程级复用
    assert get_shared_embedder() is get_shared_embedder()
    assert get_shared_store() is get_shared_store()


def test_build_registry_reuses_shared_runtime_objects():
    from app.rag import runtime
    from app.routers.chat import build_registry

    runtime.get_shared_embedder.cache_clear()
    runtime.get_shared_store.cache_clear()
    build_registry(session_factory=object(), conversation_id=None)
    build_registry(session_factory=object(), conversation_id=None)
    assert runtime.get_shared_embedder.cache_info().misses == 1
    assert runtime.get_shared_store.cache_info().misses == 1


def test_build_registry_survives_unreachable_milvus(monkeypatch):
    """Milvus 不可达时装配本身不能抛(否则整个对话,而不只是 FAQ,都 500)。"""
    from app.rag import runtime
    from app.routers.chat import build_registry

    dead = get_settings().model_copy(update={"milvus_uri": "http://127.0.0.1:19531"})
    monkeypatch.setattr("app.rag.runtime.get_settings", lambda: dead)
    runtime.get_shared_embedder.cache_clear()
    runtime.get_shared_store.cache_clear()
    try:
        reg = build_registry(session_factory=object(), conversation_id=None)
        assert "query_faq" in reg.names()
    finally:
        runtime.get_shared_embedder.cache_clear()
        runtime.get_shared_store.cache_clear()
