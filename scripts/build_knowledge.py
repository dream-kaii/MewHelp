"""离线建库:Markdown → 结构切分 → MySQL(pending)→ 向量化 → Milvus → 回填。可重跑。

用法:
  python -m scripts.build_knowledge                 # 处理 knowledge_docs/ 全部
  python -m scripts.build_knowledge --doc knowledge_docs/退货政策.md
  python -m scripts.build_knowledge --limit 50      # 本次最多向量化 50 块(分段执行)
  python -m scripts.build_knowledge --skip-embed    # 只写 MySQL 不向量化
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

from sqlalchemy.exc import IntegrityError

from app.config import get_settings
from app.db.base import dispose_engine, get_sessionmaker
from app.db import repository_knowledge as rk
from app.rag.chunker import chunk_markdown
from app.rag.embedder import get_embedder
from app.rag.store import VectorStore

logger = logging.getLogger("mewhelp.rag.build")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "knowledge_docs"

# 唯一键名取自 sql/schema.sql 的 uk_content_hash
_DUP_CONTENT_HASH_MARK = "uk_content_hash"


def _is_duplicate_content_hash(exc: IntegrityError) -> bool:
    """这次 IntegrityError 是不是 content_hash 唯一键的冲突(而非其它完整性约束)。"""
    return _DUP_CONTENT_HASH_MARK in str(getattr(exc, "orig", exc))


async def _insert_tolerant(session, payload: list[dict]) -> list[int]:
    """逐行插入,只吸收 content_hash 唯一键的冲突。

    `insert_chunks` 的「先 SELECT 再 INSERT」不是原子操作:并发建库或重跑时,
    另一个连接可能在我们 SELECT 之后提交同 content_hash 的行 —— 本次 flush 就会
    撞 UNIQUE 而后抛 IntegrityError。这里给每行套一个 SAVEPOINT:冲突时只回滚这一行
    并跳过,其余行照常插入,建库脚本因此可重跑、并发也不会崩。

    **只认 `uk_content_hash` 这一种冲突**:NOT NULL / 其它唯一键 / 真正的数据缺陷
    必须原样上抛 —— 否则会无声丢块、`inserted` 少计,日志还会把原因误报成「重复内容」。
    """
    new_ids: list[int] = []
    for item in payload:
        try:
            async with session.begin_nested():
                new_ids.extend(await rk.insert_chunks(session, [item]))
        except IntegrityError as exc:
            if not _is_duplicate_content_hash(exc):
                logger.error(
                    "插入知识块失败(非重复内容,不吞掉): doc_id=%s content_hash=%s: %s",
                    item.get("doc_id"), rk.content_hash(item["text"]), exc.orig,
                )
                raise
            logger.warning("跳过重复内容块(UNIQUE 冲突): %s: %s", item.get("doc_id"), exc.orig)
    return new_ids


async def ingest_markdown(
    session_factory, embedder, store, doc_id: str, markdown: str,
    *, content_type: str = "政策", max_chars: int = 800, overlap: int = 120, skip_embed: bool = False,
) -> dict:
    chunks = chunk_markdown(doc_id, markdown, content_type=content_type, max_chars=max_chars, overlap=overlap)
    payload = [
        dict(doc_id=c.doc_id, category=c.category, questions=c.questions, answer=c.answer, text=c.text,
             section_path=c.section_path, content_type=c.content_type, is_key_clause=c.is_key_clause)
        for c in chunks
    ]
    async with session_factory() as session:
        new_ids = await _insert_tolerant(session, payload)
        await rk.link_neighbors(session, new_ids)
        await session.commit()
    embedded = 0
    if not skip_embed:
        res = await embed_pending(session_factory, embedder, store)
        embedded = res["embedded"]
    return {"inserted": len(new_ids), "embedded": embedded}


async def embed_pending(session_factory, embedder, store, *, limit: int | None = None, batch: int = 32) -> dict:
    if limit is not None and limit <= 0:
        # --limit 0 / 负数 = 本次不向量化;不能因为 `if limit:` 为假而退化成「无限制」。
        return {"embedded": 0}
    store.ensure_collection()
    embedded = 0
    while True:
        async with session_factory() as session:
            rows = await rk.list_pending(session, limit=limit)
        if not rows:
            break
        take = rows if limit is None else rows[:limit]
        for i in range(0, len(take), batch):
            group = take[i : i + batch]
            vectors = embedder.encode([r["text"] for r in group])
            ids = [r["id"] for r in group]
            store.upsert(ids, vectors)
            async with session_factory() as session:
                await rk.mark_embedded(session, ids)
                await session.commit()
            embedded += len(ids)
        if limit is not None:
            break
    return {"embedded": embedded}


async def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    p = argparse.ArgumentParser(description="离线建库:文档 → 切分 → 双写")
    p.add_argument("--doc", help="单个文件路径;缺省处理 --dir 下全部 .md")
    p.add_argument("--dir", default=str(DEFAULT_DIR))
    p.add_argument("--limit", type=int, default=None,
                   help="本次最多向量化 N 块(分段执行);不能与 --skip-embed 同用")
    p.add_argument("--content-type", default="政策")
    p.add_argument("--skip-embed", action="store_true",
                   help="只写 MySQL 不向量化(便于演练断点续跑);不能与 --limit 同用")
    args = p.parse_args(argv)
    if args.skip_embed and args.limit is not None:
        # 二者语义矛盾(--skip-embed 要求本次完全不向量化,--limit 要求本次向量化 N 块)。
        # 旧实现让 --skip-embed 静默胜出,`--limit` 被无声吞掉 —— 用户以为限了量,实际一块没向量化。
        p.error("--skip-embed 与 --limit 不能同时使用:前者要求本次不向量化,后者要求本次向量化 N 块")

    sf = get_sessionmaker()
    store = VectorStore(uri=s.milvus_uri, token=s.milvus_token, collection=s.knowledge_collection)
    embedder = get_embedder(s)

    try:
        docs = [Path(args.doc)] if args.doc else sorted(Path(args.dir).glob("*.md"))
        if not docs:
            print(f"没有找到文档:{args.dir}", file=sys.stderr)
            return 2
        # --limit 有值:落库与向量化解耦 —— 先把文档全部写入 MySQL(pending),
        # 最后统一限量向量化,便于分段执行(不会一次把积压全量向量化)。
        deferred = args.limit is not None
        total = {"inserted": 0, "embedded": 0}
        for path in docs:
            stats = await ingest_markdown(
                sf, embedder, store, doc_id=path.name, markdown=path.read_text(encoding="utf-8"),
                content_type=args.content_type, max_chars=s.chunk_max_chars, overlap=s.chunk_overlap,
                skip_embed=args.skip_embed or deferred,
            )
            total["inserted"] += stats["inserted"]
            total["embedded"] += stats["embedded"]
            tail = "" if (args.skip_embed or deferred) else f",已向量化 {stats['embedded']}"
            print(f"[{path.name}] 新增 {stats['inserted']} 块{tail}")
        if args.skip_embed:
            print("(已跳过向量化;跑 embed_pending 或重跑本脚本会补齐)")
        elif deferred:
            total["embedded"] = (await embed_pending(sf, embedder, store, limit=args.limit))["embedded"]
            print(f"(--limit {args.limit}:本次最多向量化 {args.limit} 块,其余 pending 重跑本脚本补齐)")
        print(f"合计:新增 {total['inserted']} 块,向量化 {total['embedded']} 块")
        return 0
    finally:
        # 不 dispose 的话,连接会在事件循环关闭后才被 GC 回收,
        # 退出时刷一屏 "Event loop is closed" 噪音。
        await dispose_engine()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
