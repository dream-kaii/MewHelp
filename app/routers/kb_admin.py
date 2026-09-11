"""/kb 知识库录入页的 API:材料清单 / 切块预览 / 手工录入 / 补向量化 / 检索自测。

设计要点:
- 所有 SQL 仍只走 `app/db/repository_knowledge.py`(本模块不写裸 SQL);
- 服务对象(embedder/store/retriever/session_factory/docs_dir)通过
  `request.app.state.kb_services` 注入 —— 与 ch02 的 `app.state.chat_model` 同一模式,
  测试可注入假 embedder 与测试用 Milvus 集合,不碰真网络;
- 检索自测**额外**直接查一次 Milvus,是为了把「命中块 + 相似度」展示给运营,
  `KnowledgeRetriever.search` 仍只返回拼装文本(契约不变)。
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import repository_knowledge as rk
from app.db.base import get_sessionmaker
from app.rag.retriever import KnowledgeRetriever

logger = logging.getLogger("mewhelp.kb_admin")

router = APIRouter()

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS_DIR = ROOT / "knowledge_docs"


@dataclass
class KbServices:
    session_factory: Any
    embedder: Any
    store: Any
    retriever: Any
    docs_dir: Path = DEFAULT_DOCS_DIR


def resolve_kb(request: Request) -> KbServices:
    injected = getattr(request.app.state, "kb_services", None)
    if injected is not None:
        return injected
    from app.rag.runtime import get_shared_embedder, get_shared_store

    s = get_settings()
    sf = get_sessionmaker()
    return KbServices(
        session_factory=sf,
        embedder=get_shared_embedder(),
        store=get_shared_store(),
        retriever=KnowledgeRetriever(
            get_shared_embedder(), get_shared_store(), sf,
            top_k=s.rag_top_k, score_threshold=s.rag_score_threshold,
        ),
    )


class ManualChunk(BaseModel):
    doc_id: str = Field(min_length=1, max_length=255, description="来源标识,如 手工录入/运费补充")
    category: str = Field(default="", max_length=255)
    questions: str = Field(min_length=1, description="问法(多条用、分隔)")
    answer: str = Field(min_length=1)
    content_type: str = Field(default="手工", max_length=32)


class EmbedRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1, le=5000)


@router.get("/api/kb/docs")
async def kb_docs(services: KbServices = Depends(resolve_kb)) -> dict:
    """材料清单:knowledge_docs/ 下的文件 + 库里出现过的 doc_id,各带块数与状态计数。"""
    async with services.session_factory() as session:
        summary = await rk.doc_summary(session)
    by_doc = {d["doc_id"]: d for d in summary}
    files = sorted(p.name for p in services.docs_dir.glob("*.md")) if services.docs_dir.exists() else []
    docs = []
    for name in files:
        row = by_doc.pop(name, {"doc_id": name, "chunks": 0, "embedded": 0, "pending": 0})
        docs.append({**row, "source": "file"})
    for row in by_doc.values():  # 已删文件 / 挖矿与手工录入的 doc_id
        docs.append({**row, "source": "db"})
    totals = {
        "chunks": sum(d["chunks"] for d in docs),
        "embedded": sum(d["embedded"] for d in docs),
        "pending": sum(d["pending"] for d in docs),
    }
    return {"docs": docs, "totals": totals, "docs_dir": str(services.docs_dir)}


@router.get("/api/kb/chunks")
async def kb_chunks(
    doc_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
    services: KbServices = Depends(resolve_kb),
) -> dict:
    async with services.session_factory() as session:
        rows = await rk.list_chunks(session, doc_id=doc_id, status=status, limit=limit)
    for r in rows:  # 列表页不需要整段 text
        r.pop("text", None)
        r["answer"] = (r.get("answer") or "")[:600]
    return {"chunks": rows, "count": len(rows)}


@router.post("/api/kb/chunks")
async def kb_add_chunk(body: ManualChunk, services: KbServices = Depends(resolve_kb)) -> dict:
    """手工录入一条知识:落 MySQL 记 pending,再由「补向量化」按钮送进 Milvus。"""
    from app.rag.chunker import build_text

    text = build_text(body.category, body.questions, body.answer)
    async with services.session_factory() as session:
        ids = await rk.insert_chunks(
            session,
            [dict(doc_id=body.doc_id, category=body.category, questions=body.questions,
                  answer=body.answer, text=text, section_path=body.category,
                  content_type=body.content_type, is_key_clause=False)],
        )
        await session.commit()
    return {"inserted": bool(ids), "id": ids[0] if ids else None, "text": text}


@router.post("/api/kb/embed_pending")
async def kb_embed_pending(body: EmbedRequest | None = None, services: KbServices = Depends(resolve_kb)) -> dict:
    """把 pending(或 vector_id 为空)的块补进向量库;幂等,可反复点。"""
    from scripts.build_knowledge import embed_pending

    limit = (body.limit if body else None)
    async with services.session_factory() as session:
        before = len(await rk.list_pending(session))
    try:
        res = await embed_pending(services.session_factory, services.embedder, services.store, limit=limit)
    except Exception as exc:  # noqa: BLE001 —— Milvus/嵌入不可用要给出可读提示,不是 500 堆栈
        logger.exception("kb embed_pending failed")
        raise HTTPException(status_code=502, detail=f"向量化失败:{type(exc).__name__}: {exc}") from exc
    async with services.session_factory() as session:
        after = len(await rk.list_pending(session))
    return {"pending_before": before, "embedded": res["embedded"], "pending_after": after}


@router.get("/api/kb/search")
async def kb_search(q: str, top_k: int = 5, services: KbServices = Depends(resolve_kb)) -> dict:
    """检索自测:返回拼装文本 + 命中明细(块 id / 来源文档 / 章节 / 相似度)。"""
    if not q.strip():
        raise HTTPException(status_code=422, detail="q 不能为空")
    text = await services.retriever.search(q)
    hits: list[dict] = []
    degraded = False
    try:
        import asyncio

        vector = (await asyncio.to_thread(services.embedder.encode, [q]))[0]
        raw = await asyncio.to_thread(services.store.search, vector, top_k)
        async with services.session_factory() as session:
            rows = await rk.fetch_by_ids(session, [i for i, _ in raw])
        score_by_id = {i: s for i, s in raw}
        hits = [
            {"id": r["id"], "doc_id": r["doc_id"], "section_path": r["section_path"],
             "questions": r["questions"], "score": round(score_by_id.get(r["id"], 0.0), 4)}
            for r in rows
        ]
    except Exception:  # noqa: BLE001 —— 明细查询失败不影响自测结论(文本已由上一步给出)
        logger.exception("kb search detail failed")
        degraded = True
    return {"query": q, "text": text, "hits": hits, "hits_degraded": degraded}
