"""在线检索:query → 向量 → Milvus Top-K → MySQL 取答案 → 拼文本。"""
import logging

logger = logging.getLogger("mewhelp.rag.retriever")


class KnowledgeRetriever:
    def __init__(self, embedder, store, session_factory, top_k: int = 5, score_threshold: float = 0.5):
        self._embedder = embedder
        self._store = store
        self._session_factory = session_factory
        self._top_k = top_k
        self._threshold = score_threshold

    async def search(self, query: str) -> str:
        from app.db import repository_knowledge as rk

        vector = self._embedder.encode([query])[0]
        hits = self._store.search(vector, top_k=self._top_k)
        kept = [(i, s) for i, s in hits if s >= self._threshold]
        if not kept:
            return f"FAQ 未找到与「{query}」相关的条目(向量检索无命中)"
        async with self._session_factory() as session:
            rows = await rk.fetch_by_ids(session, [i for i, _ in kept])
        score_by_id = {i: s for i, s in kept}
        parts = []
        for r in rows:
            head = f"问:{r['questions']}" if r["questions"] else ""
            cat = f"[{r['category']}] " if r["category"] else ""
            parts.append(f"{cat}{head}\n答:{r['answer']}\n(相似度 {score_by_id.get(r['id'], 0):.3f})")
        if not parts:
            # 向量命中但 MySQL 侧行已被删/未落库(fetch_by_ids 会少返行):仍须给模型
            # 「未找到」的明确信号,而不是空串 —— 保持 query_faq 的对外契约。
            return f"FAQ 未找到与「{query}」相关的条目(向量命中但知识行已不存在)"
        return "\n".join(parts)
