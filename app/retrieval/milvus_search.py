#: Milvus hybrid 搜索 — 把 dense/sparse embed 合并成单路混合召回。
#: 已建 MilvusVectorStore 后此模块做 outward-facing 包装。
from __future__ import annotations

import logging

from app.ingestion import embed as embed_mod
from app.storage.vector_store import ChunkRecord, VectorStore, get_vector_store
from app.core.config import settings

log = logging.getLogger(__name__)


def hybrid_search(query: str, store: VectorStore | None = None,
                  top_k: int = settings.TOP_K_RETRIEVE,
                  alpha: float = settings.ALPHA) -> list[tuple[ChunkRecord, float]]:
    """单 query Milvus hybrid top-K。返回 [(ChunkRecord, score), ...]"""
    if not query.strip():
        return []
    embedder = embed_mod.get_embedder()
    dense_vecs, sparse_vecs = embedder.encode([query])
    dense_vec, sparse_vec = dense_vecs[0], sparse_vecs[0]
    vv = store or get_vector_store()
    return vv.search(dense_vec, sparse_vec, top_k, alpha)
