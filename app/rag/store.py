"""Milvus 封装:集合只存 id + vector(1024, COSINE)。"""
import logging

from pymilvus import MilvusClient

logger = logging.getLogger("mewhelp.rag.store")


class VectorStore:
    def __init__(self, uri: str, token: str = "", collection: str = "knowledge", dim: int = 1024):
        self._client = MilvusClient(uri=uri, token=token) if token else MilvusClient(uri=uri)
        self._collection = collection
        self._dim = dim

    def ensure_collection(self) -> None:
        if not self._client.has_collection(self._collection):
            # consistency_level="Strong":默认的 Bounded 会让 upsert/delete 之后的
            # count/query 读到旧值(实测有毫秒级可见性延迟),下游「写完立刻检索」
            # 会踩坑;单机 standalone 下 Strong 开销可忽略。
            self._client.create_collection(
                collection_name=self._collection,
                dimension=self._dim,
                metric_type="COSINE",
                consistency_level="Strong",
            )

    def upsert(self, ids: list[int], vectors: list[list[float]]) -> int:
        if not ids:
            return 0
        rows = [{"id": int(i), "vector": v} for i, v in zip(ids, vectors)]
        res = self._client.upsert(collection_name=self._collection, data=rows)
        return int(res.get("upsert_count", 0))

    def search(self, vector: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        res = self._client.search(
            collection_name=self._collection, data=[vector], limit=top_k, output_fields=["id"]
        )
        hits = res[0] if res else []
        out = [(int(h["id"]), float(h["distance"])) for h in hits]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def delete(self, ids: list[int]) -> int:
        if not ids:
            return 0
        res = self._client.delete(collection_name=self._collection, ids=[int(i) for i in ids])
        return int(res.get("delete_count", 0))

    def count(self) -> int:
        # 不能用 get_collection_stats()["row_count"]:它只统计已 flush 落盘的
        # segment,刚落库的行会一直读到 0(实测超过 5s 仍为 0)。
        # query 的 count(*) 走查询路径,配合 Strong 一致性能读到最新数据。
        res = self._client.query(
            collection_name=self._collection, filter="", output_fields=["count(*)"]
        )
        return int(res[0]["count(*)"]) if res else 0

    def drop(self) -> None:
        if self._client.has_collection(self._collection):
            self._client.drop_collection(self._collection)
