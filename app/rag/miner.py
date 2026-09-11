"""从历史客服对话挖问答对:LLM 抽取 → 暂存 → 哈希+向量近重去重 → 入库(pending)。"""
import json
import logging

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db import repository_knowledge as rk
from app.db.models import Conversation, Message

logger = logging.getLogger("mewhelp.rag.miner")


class QAPair(BaseModel):
    questions: list[str] = Field(description="用户可能的问法,1-3 条,取自对话原话或同义改写")
    answer: str = Field(description="客服给出的结论性回答")
    category: str = Field(default="", description="主题分类,如 运费/发票/退换货")


class QABatch(BaseModel):
    pairs: list[QAPair] = Field(default_factory=list)


PROMPT = """你是知识库编辑。下面是一段客服与用户的对话,请抽取其中「用户问题 → 客服结论」的知识点。
要求:只抽有明确结论的;questions 用用户原话或同义改写(1-3 条);没有知识点就返回空列表。"""


def _pair_key(p: QAPair) -> str:
    return rk.content_hash(json.dumps(sorted(p.questions), ensure_ascii=False) + "|" + p.answer)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(x * x for x in b) ** 0.5 or 1.0
    return dot / (na * nb)


def dedupe_pairs(pairs, *, embedder=None, existing_vectors=None, sim_threshold: float = 0.95):
    """先按内容哈希精确去重,再用向量余弦做近重复过滤。"""
    seen_hash: set[str] = set()
    kept: list[QAPair] = []
    kept_vecs: list[list[float]] = list(existing_vectors or [])
    for p in pairs:
        h = _pair_key(p)
        if h in seen_hash:
            continue
        seen_hash.add(h)
        if embedder is not None:
            vec = embedder.encode([p.questions[0] + "|" + p.answer])[0]
            if any(_cosine(vec, other) >= sim_threshold for other in kept_vecs):
                continue
            kept_vecs.append(vec)
        kept.append(p)
    return kept


async def extract_qa(llm, conversation_text: str) -> list[QAPair]:
    structured = llm.with_structured_output(QABatch)
    out = await structured.ainvoke([{"role": "system", "content": PROMPT},
                                    {"role": "user", "content": conversation_text}])
    return list(getattr(out, "pairs", []) or [])


async def _load_conversations(session, *, limit: int, lookback_days: int) -> list[tuple[int, list[dict], list[int]]]:
    from datetime import datetime, timedelta

    since = datetime.now() - timedelta(days=lookback_days)
    rows = (
        await session.execute(
            select(Conversation).where(Conversation.created_at >= since).order_by(Conversation.id.desc()).limit(limit)
        )
    ).scalars().all()
    out = []
    for conv in rows:
        msgs = (
            await session.execute(select(Message).where(Message.conversation_id == conv.id).order_by(Message.id))
        ).scalars().all()
        if not msgs:
            continue
        out.append((conv.id, [{"role": m.role, "content": m.content or ""} for m in msgs], [m.id for m in msgs]))
    return out


async def mine(session_factory, llm, embedder, *, limit: int = 20, lookback_days: int = 30, sim_threshold: float = 0.95) -> dict:
    async with session_factory() as session:
        conversations = await _load_conversations(session, limit=limit, lookback_days=lookback_days)
    staged = 0
    for conv_id, messages, msg_ids in conversations:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages if m["content"])
        if not text.strip():
            continue
        pairs = await extract_qa(llm, text)
        pairs = dedupe_pairs(pairs, embedder=embedder, sim_threshold=sim_threshold)
        if not pairs:
            continue
        items = [
            dict(conversation_id=conv_id, source_message_ids=",".join(map(str, msg_ids)),
                 questions=json.dumps(p.questions, ensure_ascii=False), answer=p.answer,
                 category=p.category, dedupe_hash=_pair_key(p))
            for p in pairs
        ]
        async with session_factory() as session:
            staged += await rk.staging_insert(session, items)
            await session.commit()
    return {"conversations": len(conversations), "staged": staged}


async def promote_staged(session_factory, *, limit: int | None = None) -> dict:
    """把暂存区里存活项提升进 knowledge_chunks(pending),并标记 promoted。"""
    async with session_factory() as session:
        rows = await rk.staging_list(session, limit=limit)
        promoted = 0
        for r in rows:
            questions = json.loads(r["questions"]) if r["questions"].startswith("[") else [r["questions"]]
            chunk = dict(
                doc_id=f"mine:{r['conversation_id']}", category=r["category"], questions="、".join(questions),
                answer=r["answer"], text=rk_build_text(r["category"], "、".join(questions), r["answer"]),
                section_path="", content_type="挖矿QA", is_key_clause=False,
            )
            ids = await rk.insert_chunks(session, [chunk])
            await rk.staging_mark(session, [r["id"]], "promoted")
            promoted += 1 if ids else 0
        await session.commit()
    return {"promoted": promoted}


def rk_build_text(category: str, questions: str, answer: str) -> str:
    from app.rag.chunker import build_text

    return build_text(category, questions, answer)
