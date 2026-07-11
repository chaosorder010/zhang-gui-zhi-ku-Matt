#: Embedding — dense + sparse 向量产出(Milvus hybrid)。
from __future__ import annotations


def embed_dense(texts: list[str]) -> list[list[float]]:
    """dense embed。MVP:占位 — 等 #3 接 bge-m3。"""
    raise NotImplementedError("dense embed — wire up bge-m3 in #3")


def embed_sparse(texts: list[str]) -> list[dict[int, float]]:
    """sparse embed。MVP:占位 — 等 #3 接 bge-m3 IDF。"""
    raise NotImplementedError("sparse embed — wire up bge-m3 IDF in #3")
