"""BGE-M3 向量化封装。真实模型懒加载;测试用 FakeEmbedder 免得下载权重。"""
import hashlib
from typing import Protocol

EMBED_DIM = 1024


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """确定性伪向量:同一文本恒得同一向量,不同文本不同(测试替身)。"""

    def __init__(self, dim: int = EMBED_DIM):
        self._dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [((h[i % len(h)] / 255.0) - 0.5) for i in range(self._dim)]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out


class BgeM3Embedder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 12, max_length: int = 8192):
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None  # 懒加载

    def _load(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            # CPU 上不能用 fp16
            self._model = BGEM3FlagModel(self._model_name, use_fp16=(self._device != "cpu"))
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        dense = model.encode(
            texts, batch_size=self._batch_size, max_length=self._max_length, return_dense=True
        )["dense_vecs"]
        return [list(map(float, v)) for v in dense]


def get_embedder(settings) -> Embedder:
    return BgeM3Embedder(
        model_name=settings.embed_model, device=settings.embed_device, batch_size=settings.embed_batch_size
    )
