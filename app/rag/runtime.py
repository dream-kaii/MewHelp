"""进程级复用的 RAG 运行时对象:embedder 与 Milvus store 都只建一次。

为什么必须复用 —— `build_registry` 在每轮对话里都会被调(`ChatService` 的
`registry_factory(cid)`):
- **embedder**:`BgeM3Embedder._model` 是实例级懒加载,每轮新建实例就等于每轮重载
  2.27GB 权重(实测冷载约 19s CPU),用户可见的灾难性延迟;
- **store**:`MilvusClient` 每次构造注册一个连接且从不关闭,长驻服务里持续泄漏。

store 额外包一层惰性代理:`Milvus 不可达时 MilvusClient 构造即抛`
(MilvusException: server unavailable),若在装配期就连,整个对话(而不只是 FAQ)
都会崩 —— 延后到第一次检索,异常交给 `KnowledgeRetriever` 的 try/except 降级到
关键词检索。
"""
from functools import lru_cache

from app.config import get_settings


class LazyStore:
    """`VectorStore` 的惰性代理:首次真正用到时才构造并连接 Milvus。"""

    def __init__(self, factory):
        self._factory = factory
        self._inner = None

    def _get(self):
        # 不缓存失败:Milvus 抖动恢复后下一轮应能自愈,而不是本进程永久降级。
        if self._inner is None:
            self._inner = self._factory()
        return self._inner

    def ensure_collection(self):
        return self._get().ensure_collection()

    def upsert(self, ids, vectors):
        return self._get().upsert(ids, vectors)

    def search(self, vector, top_k: int = 5):
        return self._get().search(vector, top_k=top_k)

    def delete(self, ids):
        return self._get().delete(ids)

    def count(self):
        return self._get().count()

    def drop(self):
        return self._get().drop()


@lru_cache
def get_shared_embedder():
    from app.rag.embedder import get_embedder

    return get_embedder(get_settings())


@lru_cache
def get_shared_store():
    from app.rag.store import VectorStore

    def _build():
        s = get_settings()
        return VectorStore(uri=s.milvus_uri, token=s.milvus_token, collection=s.knowledge_collection)

    return LazyStore(_build)
