"""离线建库:Markdown → 结构切分 → MySQL(pending)→ 向量化 → Milvus → 回填。可重跑。

用法:
  python -m scripts.build_knowledge                 # 处理 knowledge_docs/ 全部
  python -m scripts.build_knowledge --doc knowledge_docs/退货政策.md
  python -m scripts.build_knowledge --limit 50 --skip-embed
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


async def _insert_tolerant(session, payload: list[dict]) -> list[int]:
    """逐行插入并吸收 content_hash 的 UNIQUE 冲突。

    `insert_chunks` 的「先 SELECT 再 INSERT」不是原子操作:并发建库或重跑时,
    另一个连接可能在我们 SELECT 之后提交同 content_hash 的行 —— 本次 flush 就会
    撞 UNIQUE 而后抛 IntegrityError。这里给每行套一个 SAVEPOINT:冲突时只回滚这一行
    并跳过,其余行照常插入,建库脚本因此可重跑、并发也不会崩。
    """
    new_ids: list[int] = []
    for item in payload:
        try:
            async with session.begin_nested():
                new_ids.extend(await rk.insert_chunks(session, [item]))
        except IntegrityError:
            logger.warning("跳过重复内容块(UNIQUE 冲突): %s", item.get("doc_id"))
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
    store.ensure_collection()
    embedded = 0
    while True:
        async with session_factory() as session:
            rows = await rk.list_pending(session, limit=limit)
        if not rows:
            break
        take = rows[: (limit or len(rows))]
        for i in range(0, len(take), batch):
            group = take[i : i + batch]
            vectors = embedder.encode([r["text"] for r in group])
            ids = [r["id"] for r in group]
            store.upsert(ids, vectors)
            async with session_factory() as session:
                await rk.mark_embedded(session, ids)
                await session.commit()
            embedded += len(ids)
        if limit:
            break
    return {"embedded": embedded}


async def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    p = argparse.ArgumentParser(description="离线建库:文档 → 切分 → 双写")
    p.add_argument("--doc", help="单个文件路径;缺省处理 --dir 下全部 .md")
    p.add_argument("--dir", default=str(DEFAULT_DIR))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--content-type", default="政策")
    p.add_argument("--skip-embed", action="store_true", help="只写 MySQL 不向量化(便于演练断点续跑)")
    args = p.parse_args(argv)

    sf = get_sessionmaker()
    store = VectorStore(uri=s.milvus_uri, token=s.milvus_token, collection=s.knowledge_collection)
    embedder = get_embedder(s)

    try:
        docs = [Path(args.doc)] if args.doc else sorted(Path(args.dir).glob("*.md"))
        if not docs:
            print(f"没有找到文档:{args.dir}", file=sys.stderr)
            return 2
        total = {"inserted": 0, "embedded": 0}
        for path in docs:
            stats = await ingest_markdown(
                sf, embedder, store, doc_id=path.name, markdown=path.read_text(encoding="utf-8"),
                content_type=args.content_type, max_chars=s.chunk_max_chars, overlap=s.chunk_overlap,
                skip_embed=args.skip_embed,
            )
            total["inserted"] += stats["inserted"]
            total["embedded"] += stats["embedded"]
            print(f"[{path.name}] 新增 {stats['inserted']} 块,已向量化 {stats['embedded']}")
        if args.skip_embed:
            print("(已跳过向量化;跑 embed_pending 或重跑本脚本会补齐)")
        print(f"合计:新增 {total['inserted']} 块,向量化 {total['embedded']} 块")
        return 0
    finally:
        # 不 dispose 的话,连接会在事件循环关闭后才被 GC 回收,
        # 退出时刷一屏 "Event loop is closed" 噪音。
        await dispose_engine()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
