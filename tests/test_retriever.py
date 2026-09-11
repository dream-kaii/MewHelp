import pytest

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
    ids = await _seed(db_session)
    r = KnowledgeRetriever(FakeEmbedder(), StubStore([(ids[0], 0.10)]), session_factory, top_k=5, score_threshold=0.5)
    out = await r.search("邮费是多少")
    assert "未找到" in out


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
