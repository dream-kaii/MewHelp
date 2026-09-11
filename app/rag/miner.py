"""从历史客服对话挖问答对:LLM 抽取 → 暂存 → 哈希+向量近重去重 → 入库(pending)。"""
import json
import logging

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db import repository_knowledge as rk
from app.db.models import Conversation, Message

logger = logging.getLogger("mewhelp.rag.miner")

# 唯一键名取自 sql/schema.sql 的 uk_content_hash(与 scripts/build_knowledge.py 同一范式)
_DUP_CONTENT_HASH_MARK = "uk_content_hash"


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


def _pair_text(p: QAPair) -> str:
    """向量化用的文本:首条问法 + 结论。调用前已保证 questions 非空。"""
    return (p.questions[0] if p.questions else "") + "|" + p.answer


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(x * x for x in b) ** 0.5 or 1.0
    return dot / (na * nb)


def dedupe_pairs(pairs, *, embedder=None, existing_vectors=None, sim_threshold: float = 0.95, store=None):
    """哈希精确去重 + 向量近重,返回保留的对。

    - `questions` 为空的对直接丢弃(模型可能返回 `{"questions": []}`;否则取 questions[0] 会 IndexError
      并把无人值守的挖矿 job 卡死在同一处),并记一条 warning;
    - `embedder` 给出时批量向量化候选对,与向量池比余弦(≥ `sim_threshold` 即判为近重);
    - `existing_vectors` 既是**输入池**也是**输出池**:保留项的向量会就地追加进去,
      调用方把它跨会话/跨批次复用即可实现整体去重(同一轮内先挖到的会挡住后挖到的);
    - `store`(VectorStore)给出时,再拿候选向量去既有知识库 `search(vec, top_k=1)`,
      命中最相似块的分数 ≥ `sim_threshold` 即丢弃(与已入库内容去重)。store 依赖 embedder,缺 embedder 时跳过。
    """
    pool = existing_vectors if existing_vectors is not None else []
    seen_hash: set[str] = set()
    candidates: list[QAPair] = []
    for p in pairs:
        if not p.questions:
            logger.warning("跳过 questions 为空的问答对: answer=%r", (p.answer or "")[:60])
            continue
        h = _pair_key(p)
        if h in seen_hash:
            continue
        seen_hash.add(h)
        candidates.append(p)

    use_vectors = embedder is not None and bool(candidates)
    vectors = embedder.encode([_pair_text(p) for p in candidates]) if use_vectors else []

    kept: list[QAPair] = []
    for i, p in enumerate(candidates):
        if use_vectors:
            vec = vectors[i]
            if any(_cosine(vec, other) >= sim_threshold for other in pool):
                continue
            if store is not None:
                hits = store.search(vec, top_k=1)
                if hits and hits[0][1] >= sim_threshold:
                    continue
            pool.append(vec)
        kept.append(p)
    return kept


async def extract_qa(llm, conversation_text: str) -> list[QAPair]:
    structured = llm.with_structured_output(QABatch)
    out = await structured.ainvoke([{"role": "system", "content": PROMPT},
                                    {"role": "user", "content": conversation_text}])
    pairs = getattr(out, "pairs", None)
    if pairs is None:
        # 结构化输出没按 QABatch 返回(上游不认 schema / 返回了别的形状):按「无知识点」处理,但别静默。
        logger.warning("结构化抽取返回意外形状(%s),按无知识点处理", type(out).__name__)
        return []
    return list(pairs)


async def _load_conversations(session, *, limit: int | None, lookback_days: int) -> list[tuple[int, list[dict], list[int]]]:
    from datetime import datetime, timedelta

    since = datetime.now() - timedelta(days=lookback_days)
    stmt = select(Conversation).where(Conversation.created_at >= since).order_by(Conversation.id.desc())
    if limit is not None and limit > 0:
        # limit 为 None / <= 0 一律理解为「不限」:`.limit(0)` 会变成 LIMIT 0(静默扫 0 条),
        # 负值在 MySQL 上是语法错误 —— 两者都不是调用方想要的语义。
        stmt = stmt.limit(limit)
    rows = (await session.execute(stmt)).scalars().all()
    out = []
    for conv in rows:
        msgs = (
            await session.execute(select(Message).where(Message.conversation_id == conv.id).order_by(Message.id))
        ).scalars().all()
        if not msgs:
            continue
        out.append((conv.id, [{"role": m.role, "content": m.content or ""} for m in msgs], [m.id for m in msgs]))
    return out


async def mine(
    session_factory, llm, embedder, *,
    limit: int | None = 20, lookback_days: int = 30, sim_threshold: float = 0.95, store=None,
) -> dict:
    """一轮挖掘。返回 `{"conversations", "staged", "failed"}`。

    - 单个会话失败(抽取校验异常、网络抖动、入库报错)只记日志并跳过,不中断整轮:
      否则一个坏会话会让后面的会话永远轮不到、`promote_staged` 永远执行不到;
    - `store` 传入时,候选对先与既有知识库比对,已存在的近重内容不再入暂存。
    """
    if store is not None:
        if embedder is None:
            logger.warning("传了 store 但没传 embedder:跳过与既有知识库的去重,只做哈希/向量池去重")
        store.ensure_collection()  # 首跑/空库时也要能 search
    async with session_factory() as session:
        conversations = await _load_conversations(session, limit=limit, lookback_days=lookback_days)
    kept_vectors: list[list[float]] = []  # 本轮累积的向量池 → 跨会话近重生效
    staged = failed = 0
    for conv_id, messages, msg_ids in conversations:
        try:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages if m["content"])
            if not text.strip():
                continue
            pairs = await extract_qa(llm, text)
            pairs = dedupe_pairs(
                pairs, embedder=embedder, existing_vectors=kept_vectors,
                sim_threshold=sim_threshold, store=store,
            )
            if not pairs:
                continue
            items = [
                dict(conversation_id=conv_id, source_message_ids=",".join(map(str, msg_ids)),
                     questions=json.dumps(p.questions, ensure_ascii=False), answer=p.answer,
                     category=p.category, dedupe_hash=_pair_key(p))
                for p in pairs
            ]
            async with session_factory() as session:
                inserted = await rk.staging_insert(session, items)
                await session.commit()
            staged += inserted  # 提交成功才计数:commit 失败会走 except,不能虚报「已入暂存」
        except Exception:
            failed += 1
            logger.exception("会话 %s 挖掘失败,跳过(不影响本轮其余会话)", conv_id)
    return {"conversations": len(conversations), "staged": staged, "failed": failed}


def _is_duplicate_content_hash(exc: IntegrityError) -> bool:
    """这次 IntegrityError 是不是 content_hash 唯一键的冲突(而非其它完整性约束)。"""
    return _DUP_CONTENT_HASH_MARK in str(getattr(exc, "orig", exc))


async def _insert_tolerant(session, rows: list[dict]) -> list[int]:
    """逐行 SAVEPOINT 插入,只吸收 content_hash 唯一键的冲突(其余原样上抛)。

    `insert_chunks` 的「先 SELECT 再 INSERT」不是原子操作:并发 cron / 建库时另一个连接
    可能在我们 SELECT 之后提交同 content_hash 的行,本次 flush 就会撞 UNIQUE。
    与 `scripts/build_knowledge.py::_insert_tolerant` 同一范式:冲突只回滚这一行。
    """
    new_ids: list[int] = []
    for row in rows:
        try:
            async with session.begin_nested():
                new_ids.extend(await rk.insert_chunks(session, [row]))
        except IntegrityError as exc:
            if not _is_duplicate_content_hash(exc):
                logger.error(
                    "提升知识块失败(非重复内容,不吞掉): doc_id=%s content_hash=%s: %s",
                    row.get("doc_id"), rk.content_hash(row["text"]), exc.orig,
                )
                raise
            logger.warning("跳过重复内容块(UNIQUE 冲突): %s: %s", row.get("doc_id"), exc.orig)
    return new_ids


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
            ids = await _insert_tolerant(session, [chunk])
            await rk.staging_mark(session, [r["id"]], "promoted")
            promoted += 1 if ids else 0
        await session.commit()
    return {"promoted": promoted}


def rk_build_text(category: str, questions: str, answer: str) -> str:
    from app.rag.chunker import build_text

    return build_text(category, questions, answer)
