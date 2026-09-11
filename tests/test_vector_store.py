import pytest

from app.config import get_settings
from app.rag.store import VectorStore

pytestmark = pytest.mark.anyio


@pytest.fixture
def store():
    s = get_settings()
    vs = VectorStore(uri=s.milvus_uri, token=s.milvus_token, collection=s.milvus_test_collection)
    vs.drop()
    vs.ensure_collection()
    yield vs
    vs.drop()


def _vec(seed: float, dim: int = 1024, at: int = 0):
    """把 seed 放在第 at 维上。COSINE 只看方向,所以必须用不同的 at 造出不同的相似度,
    否则 [1,0,...] 与 [0.5,0,...] 的余弦相似度同为 1.0,无法区分远近。"""
    v = [0.0] * dim
    v[at] = seed
    return v


def test_ensure_is_idempotent_and_count_starts_zero(store):
    store.ensure_collection()
    assert store.count() == 0


def test_upsert_then_search_returns_nearest(store):
    store.upsert([1, 2, 3], [_vec(1.0), _vec(0.5, at=1), _vec(-1.0)])
    hits = store.search(_vec(1.0), top_k=2)
    assert hits[0][0] == 1
    assert hits[0][1] > hits[1][1]


def test_upsert_is_idempotent_by_primary_key(store):
    store.upsert([7], [_vec(1.0)])
    store.upsert([7], [_vec(1.0)])  # 同主键再写
    assert store.count() == 1


def test_delete_removes_rows(store):
    store.upsert([1, 2], [_vec(1.0), _vec(0.5)])
    assert store.delete([1]) == 1
    assert store.count() == 1
