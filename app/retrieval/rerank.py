#: Rerank 二阶精排 — BGE-Reranker-Large / Qwen3-Reranker。
from __future__ import annotations

from app.retrieval.milvus_search import Candidate


def rerank(query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
    """精排 top-N 送 LLM。MVP:占位 — 等 #3 接。"""
    raise NotImplementedError("rerank — wire up model in #3")
