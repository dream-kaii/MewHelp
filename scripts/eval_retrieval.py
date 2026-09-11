# scripts/eval_retrieval.py
"""向量检索质量评测。用法:python -m scripts.eval_retrieval"""
import asyncio
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.db.base import get_sessionmaker
from app.db import repository_knowledge as rk
from app.rag.embedder import get_embedder
from app.rag.retriever import KnowledgeRetriever
from app.rag.store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "eval_data" / "retrieval_samples.json"


async def main() -> int:
    s = get_settings()
    sf = get_sessionmaker()
    store = VectorStore(uri=s.milvus_uri, token=s.milvus_token, collection=s.knowledge_collection)
    retriever = KnowledgeRetriever(get_embedder(s), store, sf, top_k=s.rag_top_k, score_threshold=s.rag_score_threshold)
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    hits = 0
    for i, sm in enumerate(samples, 1):
        out = await retriever.search(sm["question"])
        kw_ok = all(k in out for k in sm["expect_keywords"])
        doc_ok = sm["expect_doc"].replace(".md", "") in out
        ok = kw_ok and doc_ok
        hits += ok
        print(f"[{'PASS' if ok else 'FAIL'}] #{i} {sm['question']} -> {out[:80].replace(chr(10),' ')}")
    rate = hits / len(samples)
    print(f"\n检索命中率: {hits}/{len(samples)} = {rate:.0%}")
    return 0 if rate >= 0.75 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
