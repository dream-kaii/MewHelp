"""knowledge_chunks / knowledge_staging 的读写与状态机(唯一写这两张 SQL 的地方)。"""
import hashlib
import re

from sqlalchemy import case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import KnowledgeChunk, KnowledgeStaging


def content_hash(text: str) -> str:
    norm = re.sub(r"\s+", "", (text or "")).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


async def insert_chunks(session: AsyncSession, chunks: list[dict]) -> list[int]:
    """按 content_hash 幂等插入;返回本次真正新插入的 id。"""
    new_ids: list[int] = []
    for c in chunks:
        h = c.get("content_hash") or content_hash(c["text"])
        exists = (
            await session.execute(select(KnowledgeChunk.id).where(KnowledgeChunk.content_hash == h))
        ).scalar_one_or_none()
        if exists:
            continue
        row = KnowledgeChunk(
            doc_id=c["doc_id"], category=c.get("category", ""), questions=c["questions"],
            answer=c["answer"], text=c["text"], section_path=c.get("section_path", ""),
            content_type=c.get("content_type", "政策"), is_key_clause=bool(c.get("is_key_clause", False)),
            content_hash=h, status="pending",
        )
        session.add(row)
        await session.flush()
        new_ids.append(row.id)
    return new_ids


async def link_neighbors(session: AsyncSession, ids: list[int]) -> None:
    for prev_id, cur_id, next_id in zip([None, *ids[:-1]], ids, [*ids[1:], None]):
        row = await session.get(KnowledgeChunk, cur_id)
        if row is not None:
            row.prev_id, row.next_id = prev_id, next_id
    await session.flush()


async def list_pending(session: AsyncSession, limit: int | None = None) -> list[dict]:
    stmt = (
        select(KnowledgeChunk)
        .where(or_(KnowledgeChunk.status == "pending", KnowledgeChunk.vector_id.is_(None)))
        .order_by(KnowledgeChunk.id)
    )
    if limit:
        stmt = stmt.limit(limit)
    rows = (await session.execute(stmt)).scalars().all()
    return [_to_dict(r) for r in rows]


async def mark_embedded(session: AsyncSession, ids: list[int]) -> None:
    for i in ids:
        row = await session.get(KnowledgeChunk, i)
        if row is not None:
            row.vector_id = str(i)
            row.status = "embedded"
    await session.flush()


async def fetch_by_ids(session: AsyncSession, ids: list[int]) -> list[dict]:
    if not ids:
        return []
    rows = (await session.execute(select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids)))).scalars().all()
    by_id = {r.id: _to_dict(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


async def staging_insert(session: AsyncSession, items: list[dict]) -> int:
    n = 0
    for it in items:
        h = it.get("dedupe_hash") or content_hash(f"{it['questions']}|{it['answer']}")
        exists = (
            await session.execute(select(KnowledgeStaging.id).where(KnowledgeStaging.dedupe_hash == h))
        ).scalar_one_or_none()
        if exists:
            continue
        session.add(
            KnowledgeStaging(
                conversation_id=it["conversation_id"], source_message_ids=it.get("source_message_ids", ""),
                questions=it["questions"], answer=it["answer"], category=it.get("category", ""),
                dedupe_hash=h, status="staged",
            )
        )
        n += 1
    await session.flush()
    return n


async def staging_list(session: AsyncSession, limit: int | None = None) -> list[dict]:
    stmt = select(KnowledgeStaging).where(KnowledgeStaging.status == "staged").order_by(KnowledgeStaging.id)
    if limit:
        stmt = stmt.limit(limit)
    rows = (await session.execute(stmt)).scalars().all()
    return [
        dict(id=r.id, conversation_id=r.conversation_id, source_message_ids=r.source_message_ids,
             questions=r.questions, answer=r.answer, category=r.category, dedupe_hash=r.dedupe_hash)
        for r in rows
    ]


async def staging_mark(session: AsyncSession, ids: list[int], status: str) -> None:
    for i in ids:
        row = await session.get(KnowledgeStaging, i)
        if row is not None:
            row.status = status
    await session.flush()


def _to_dict(r: KnowledgeChunk) -> dict:
    return dict(
        id=r.id, doc_id=r.doc_id, category=r.category, questions=r.questions, answer=r.answer,
        text=r.text, section_path=r.section_path, content_type=r.content_type,
        is_key_clause=bool(r.is_key_clause), prev_id=r.prev_id, next_id=r.next_id,
        content_hash=r.content_hash, vector_id=r.vector_id, status=r.status,
    )


async def doc_summary(session: AsyncSession) -> list[dict]:
    """按 doc_id 汇总块数与状态计数(/kb 材料清单用)。"""
    rows = (
        await session.execute(
            select(
                KnowledgeChunk.doc_id,
                func.count(KnowledgeChunk.id),
                func.sum(case((KnowledgeChunk.status == "embedded", 1), else_=0)),
            )
            .group_by(KnowledgeChunk.doc_id)
            .order_by(KnowledgeChunk.doc_id)
        )
    ).all()
    return [
        {"doc_id": doc_id, "chunks": int(total or 0), "embedded": int(embedded or 0),
         "pending": int(total or 0) - int(embedded or 0)}
        for doc_id, total, embedded in rows
    ]


async def list_chunks(
    session: AsyncSession, *, doc_id: str | None = None, status: str | None = None, limit: int = 100
) -> list[dict]:
    """列出知识块(可按 doc_id / status 过滤),按 id 升序,供 /kb 切块预览。"""
    stmt = select(KnowledgeChunk).order_by(KnowledgeChunk.id).limit(max(1, limit))
    if doc_id:
        stmt = stmt.where(KnowledgeChunk.doc_id == doc_id)
    if status:
        stmt = stmt.where(KnowledgeChunk.status == status)
    rows = (await session.execute(stmt)).scalars().all()
    return [_to_dict(r) for r in rows]
