#: Milvus hybrid 搜索 — score = α·dense + (1-α)·sparse。
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candidate:
    """单召回候选。"""

    chunk_id: str
    text: str
    score: float
    item_name: str = ""
    source: str = "local"


def hybrid_search(query: str, top_k: int) -> list[Candidate]:
    """Milvus hybrid top-K。MVP:占位 — 等 #3 接。"""
    raise NotImplementedError("hybrid_search — wire up in #3")
