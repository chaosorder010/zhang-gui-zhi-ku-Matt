#: Embedding — dense + sparse 向量产出(Milvus hybrid)。
from __future__ import annotations

import hashlib
import logging
import math
import re
from functools import lru_cache
from typing import Iterable

from app.core.config import settings

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[\w一-鿿]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _SparseCoder:
    """BM25-style 稀疏编码器。用文档集合的 IDF;无解码器亦可运行。"""

    def __init__(self, dfs: dict[str, int] | None = None, n: int = 0) -> None:
        self._dfs: dict[str, int] = dict(dfs or {})
        self._n = n

    def _idf(self, term: str) -> float:
        df = self._dfs.get(term, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1.0) if self._n else 1.0

    def encode(self, text: str) -> dict[int, float]:
        toks = _tokens(text)
        if not toks:
            return {}
        out: dict[int, float] = {}
        for t in set(toks):
            tf = toks.count(t) / len(toks)
            out[abs(hash(t)) % (10**9)] = tf * self._idf(t)
        return out


class _DenseModel:
    """尝试 sentence_transformers bge-m3。"""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        self.model = SentenceTransformer(settings.EMBED_MODEL, device=settings.EMBED_DEVICE)
        dim = self.model.get_sentence_embedding_dimension()
        if settings.EMBED_DENSE_DIM != dim:
            log.warning("EMBED_DENSE_DIM tuned %d → model dim %d", settings.EMBED_DENSE_DIM, dim)

    def encode(self, texts: list[str]) -> list[list[float]]:
        import numpy as np  # type: ignore[import-untyped]

        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() if isinstance(v, np.ndarray) else list(v) for v in vecs]


class _DenseFallback:
    """确定性伪 dense。测试用,维度对齐、同一输入同 output。"""

    def __init__(self) -> None:
        self.dim = settings.EMBED_DENSE_DIM

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            for tok in _tokens(t):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            z = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / z for v in vec])
        return out


class Embedder:
    """单例封装 dense + sparse。fit_sparse 索引期先吃 corpus 建 IDF。"""

    def __init__(self) -> None:
        self.dim = settings.EMBED_DENSE_DIM
        self._dense = self._load_dense()
        self._sparse_dfs: dict[str, int] = {}
        self._sparse_n: int = 0

    def _load_dense(self):
        if settings.EMBED_FALLBACK:
            return _DenseFallback()
        try:
            return _DenseModel()
        except ImportError:
            log.warning("sentence_transformers missing, dense fallback on")
            return _DenseFallback()

    def fit_sparse(self, corpus: Iterable[str]) -> None:
        """索引期先吃 corpus 建 IDF。"""
        for doc in corpus:
            for t in set(_tokens(doc)):
                self._sparse_dfs[t] = self._sparse_dfs.get(t, 0) + 1
            self._sparse_n += 1

    def dense(self, texts: list[str]) -> list[list[float]]:
        return self._dense.encode(texts)

    def sparse(self, texts: list[str]) -> list[dict[int, float]]:
        coder = _SparseCoder(self._sparse_dfs, self._sparse_n)
        return [coder.encode(t) for t in texts]

    def encode(self, texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
        return self.dense(texts), self.sparse(texts)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


def embed_dense(texts: list[str]) -> list[list[float]]:
    return get_embedder().dense(texts)


def embed_sparse(texts: list[str]) -> list[dict[int, float]]:
    return get_embedder().sparse(texts)
