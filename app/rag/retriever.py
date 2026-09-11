"""在线检索:query → 向量 → Milvus Top-K → MySQL 取答案 → 拼文本。

**本章只跑 dense 单路**(spec §12):阈值挡下的「干净未命中」就是未命中,不改走关键词召回 ——
关键词 LIKE(ch02)只是**向量路径因服务不可用而抛异常**时的兜底,保证 Milvus 缺席不整章失效。
"""
import logging

from pymilvus.exceptions import MilvusException
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger("mewhelp.rag.retriever")

# 只有「依赖服务/环境不可用」才降级到关键词检索:
#   MilvusException —— Milvus 连不上 / 集合不存在
#   SQLAlchemyError —— MySQL 不可用(取答案那一步)
#   OSError         —— 嵌入模型权重缺失 / 网络不可达(含 ConnectionError/TimeoutError)
#   ImportError     —— 嵌入依赖未安装(FlagEmbedding 缺席)
# 其余异常(KeyError/AttributeError/TypeError…)是本模块的**编程错误**,直接上抛,
# 不能静默降级成「未找到」把真 bug 藏起来。
_DEGRADE_ERRORS = (MilvusException, SQLAlchemyError, OSError, ImportError)


def _not_found(query: str, reason: str) -> str:
    return f"FAQ 未找到与「{query}」相关的条目({reason})"


class KnowledgeRetriever:
    def __init__(self, embedder, store, session_factory, top_k: int = 5, score_threshold: float = 0.5):
        self._embedder = embedder
        self._store = store
        self._session_factory = session_factory
        self._top_k = top_k
        self._threshold = score_threshold

    async def search(self, query: str) -> str:
        try:
            return await self._vector_search(query)
        except _DEGRADE_ERRORS:
            logger.exception("向量检索不可用,降级关键词检索(query=%r)", query)
        try:
            return await self._keyword_search(query)
        except _DEGRADE_ERRORS:
            logger.exception("关键词检索也不可用(query=%r)", query)
            return _not_found(query, "向量与关键词检索均不可用")

    async def _vector_search(self, query: str) -> str:
        """向量单路检索:命中返回组织好的文本;**干净未命中直接返回含「未找到」的字符串**。

        这里不做任何关键词召回 —— 唯一的关键词兜底在 `search` 的 except 分支。
        """
        from app.db import repository_knowledge as rk

        vector = self._embedder.encode([query])[0]
        hits = self._store.search(vector, top_k=self._top_k)
        kept = [(i, s) for i, s in hits if s >= self._threshold]
        if not kept:
            logger.warning("向量检索无命中(query=%r, hits=%d, threshold=%.3f)", query, len(hits), self._threshold)
            return _not_found(query, "向量检索无命中")
        async with self._session_factory() as session:
            rows = await rk.fetch_by_ids(session, [i for i, _ in kept])
        score_by_id = {i: s for i, s in kept}
        parts = []
        for r in rows:
            head = f"问:{r['questions']}" if r["questions"] else ""
            cat = f"[{r['category']}] " if r["category"] else ""
            parts.append(f"{cat}{head}\n答:{r['answer']}\n(相似度 {score_by_id.get(r['id'], 0):.3f})")
        if not parts:
            # fetch_by_ids 少返行(行被删/未落库):仍是"干净未命中",不能返空串,也不改走关键词。
            logger.warning("向量命中但知识行已不存在(query=%r, ids=%r)", query, [i for i, _ in kept])
            return _not_found(query, "向量命中但知识行已不存在")
        return "\n".join(parts)

    async def _keyword_search(self, query: str) -> str:
        """ch02 兜底:关键词 LIKE 查 faq 表。**只应由 `search` 的 except 分支调用。**"""
        from app.db import repository as repo

        async with self._session_factory() as session:
            hits = await repo.search_faq(session, query)
        if not hits:
            return _not_found(query, "关键词检索无命中")
        return "\n".join(f"问:{h['question']}\n答:{h['answer']}" for h in hits)
