from app.rag.embedder import EMBED_DIM, FakeEmbedder, get_embedder


def test_fake_embedder_shape_and_determinism():
    e = FakeEmbedder()
    v1 = e.encode(["邮费是多少"])[0]
    v2 = e.encode(["邮费是多少"])[0]
    assert len(v1) == EMBED_DIM and v1 == v2
    assert e.encode(["邮费是多少"])[0] != e.encode(["运费怎么算"])[0]


def test_fake_embedder_batch():
    e = FakeEmbedder()
    out = e.encode(["a", "b", "c"])
    assert len(out) == 3 and all(len(v) == EMBED_DIM for v in out)


def test_get_embedder_returns_real_impl_without_loading(monkeypatch):
    """get_embedder 只构造对象,不在构造期下载/加载模型。"""
    from app.config import get_settings

    e = get_embedder(get_settings())
    assert e.__class__.__name__ == "BgeM3Embedder"
    assert getattr(e, "_model", None) is None  # 懒加载
