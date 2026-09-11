"""在线检索:query → 向量 → Milvus Top-K → MySQL 取答案 → 拼文本。

向量路径**不设「必须成功」的前提**:Milvus 连不上/集合没建、embedding 抛错、命中行
已被删、无命中过阈值 —— 任何一环失败都只记 warning 并回退 ch02 的关键词 LIKE 检索,
**两路都无命中才返回含「未找到」的字符串**(且不向上抛)。FAQ 能力因此不会因为
Milvus 缺席而整体不可用。
"""
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
        text = None
        try:
            text = await self._vector_search(query)
        except Exception:
            logger.warning("向量检索失败,回退关键词检索(query=%r)", query, exc_info=True)
        if text is None:
            try:
                text = await self._keyword_search(query)
            except Exception:
                logger.warning("关键词检索也失败(query=%r)", query, exc_info=True)
        if text is None:
            return f"FAQ 未找到与「{query}」相关的条目(向量与关键词检索均无命中)"
        return text

    async def _vector_search(self, query: str) -> str | None:
        """命中返回组织好的文本;无命中或命中行已删返回 None(交给关键词路径兜底)。"""
        from app.db import repository_knowledge as rk

        vector = self._embedder.encode([query])[0]
        hits = self._store.search(vector, top_k=self._top_k)
        kept = [(i, s) for i, s in hits if s >= self._threshold]
        if not kept:
            logger.warning("向量检索无命中(query=%r, hits=%d, threshold=%.3f)", query, len(hits), self._threshold)
            return None
        async with self._session_factory() as session:
            rows = await rk.fetch_by_ids(session, [i for i, _ in kept])
        score_by_id = {i: s for i, s in kept}
        parts = []
        for r in rows:
            head = f"问:{r['questions']}" if r["questions"] else ""
            cat = f"[{r['category']}] " if r["category"] else ""
            parts.append(f"{cat}{head}\n答:{r['answer']}\n(相似度 {score_by_id.get(r['id'], 0):.3f})")
        if not parts:
            # fetch_by_ids 少返行(行被删/未落库):不能返空串 —— 交给关键词路径兜底。
            logger.warning("向量命中但知识行已不存在(query=%r, ids=%r)", query, [i for i, _ in kept])
            return None
        return "\n".join(parts)

    async def _keyword_search(self, query: str) -> str | None:
        """ch02 回退:关键词 LIKE 查 faq 表;无命中返回 None。"""
        from app.db import repository as repo

        async with self._session_factory() as session:
            hits = await repo.search_faq(session, query)
        if not hits:
            return None
        return "\n".join(f"问:{h['question']}\n答:{h['answer']}" for h in hits)
