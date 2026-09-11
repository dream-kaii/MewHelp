"""/kb 录入页 API 的契约测试。

用注入的假 embedder + 测试用 Milvus 集合 + 测试库(mewhelp_test),不打真网络:
- embedder 用 `ConstantEmbedder`(所有文本同一向量)→ 让检索必然命中,断言的是**接线**而非模型质量;
- 连接池用 NullPool,避免 TestClient 每例新事件循环与池化连接冲突。
"""
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.db.base import build_database_url
from app.main import app
from app.rag.retriever import KnowledgeRetriever
from app.rag.store import VectorStore
from app.routers.kb_admin import DEFAULT_DOCS_DIR, KbServices
from tests.db_utils import truncate_all

EMBED_DIM = 1024


class ConstantEmbedder:
    """所有文本同一方向 → 余弦恒 1.0,用于让检索自测必然命中。"""

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (EMBED_DIM - 1) for _ in texts]


@pytest.fixture
def store():
    s = get_settings()
    vs = VectorStore(uri=s.milvus_uri, token=s.milvus_token, collection=s.milvus_test_collection)
    vs.drop()
    vs.ensure_collection()
    yield vs
    vs.drop()


@pytest.fixture
def kb(client, store):
    """把注入版服务挂到 app.state,测试结束后还原。"""
    s = get_settings()
    truncate_all(s.mysql_test_database)
    engine = create_async_engine(
        build_database_url(s, database=s.mysql_test_database), poolclass=NullPool
    )
    maker = async_sessionmaker(engine, expire_on_commit=False)
    app.state.kb_services = KbServices(
        session_factory=maker,
        embedder=ConstantEmbedder(),
        store=store,
        retriever=KnowledgeRetriever(ConstantEmbedder(), store, maker, top_k=5, score_threshold=0.5),
        docs_dir=DEFAULT_DOCS_DIR,
    )
    yield client
    app.state.kb_services = None


def _add(client, **over):
    body = {"doc_id": "手工录入/测试", "category": "运费", "questions": "邮费怎么算",
            "answer": "按收货地区收取", "content_type": "手工"}
    body.update(over)
    return client.post("/api/kb/chunks", json=body)


def test_kb_page_and_docs_endpoint(kb):
    assert kb.get("/kb").status_code == 200
    data = kb.get("/api/kb/docs").json()
    assert data["totals"] == {"chunks": 0, "embedded": 0, "pending": 0}
    assert any(d["doc_id"] == "运费说明.md" and d["source"] == "file" for d in data["docs"])


def test_manual_chunk_inserts_pending_then_dedupes(kb):
    r1 = _add(kb).json()
    assert r1["inserted"] is True and r1["id"] > 0
    r2 = _add(kb).json()
    assert r2["inserted"] is False and r2["id"] is None

    data = kb.get("/api/kb/docs").json()
    mine = next(d for d in data["docs"] if d["doc_id"] == "手工录入/测试")
    assert mine == {"doc_id": "手工录入/测试", "chunks": 1, "embedded": 0, "pending": 1, "source": "db"}


def test_chunks_preview_filters_by_doc(kb):
    _add(kb)
    rows = kb.get("/api/kb/chunks", params={"doc_id": "手工录入/测试"}).json()["chunks"]
    assert len(rows) == 1
    assert rows[0]["questions"] == "邮费怎么算" and rows[0]["status"] == "pending"
    assert "text" not in rows[0]  # 预览不返回整段 text


def test_embed_pending_moves_rows_to_embedded(kb, store):
    _add(kb)
    _add(kb, questions="退货运费谁出?", answer="质量问题商家承担")
    res = kb.post("/api/kb/embed_pending", json={}).json()
    assert res["pending_before"] == 2 and res["embedded"] == 2 and res["pending_after"] == 0
    assert store.count() == 2
    rows = kb.get("/api/kb/chunks", params={"doc_id": "手工录入/测试"}).json()["chunks"]
    assert {r["status"] for r in rows} == {"embedded"}


def test_embed_pending_is_idempotent(kb):
    _add(kb)
    kb.post("/api/kb/embed_pending", json={})
    again = kb.post("/api/kb/embed_pending", json={}).json()
    assert again == {"pending_before": 0, "embedded": 0, "pending_after": 0}


def test_search_returns_text_and_hit_details(kb):
    _add(kb, questions="邮费是多少", answer="下单运费按收货地区收取")
    kb.post("/api/kb/embed_pending", json={})
    data = kb.get("/api/kb/search", params={"q": "邮费是多少", "top_k": 3}).json()
    assert "按收货地区收取" in data["text"] and "未找到" not in data["text"]
    assert data["hits"] and data["hits"][0]["doc_id"] == "手工录入/测试"
    assert data["hits"][0]["score"] == pytest.approx(1.0, abs=1e-6)
    assert data["hits_degraded"] is False


def test_search_rejects_blank_query(kb):
    r = kb.get("/api/kb/search", params={"q": "   "})
    assert r.status_code == 422
