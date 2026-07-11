#: RRF 融合 — score(d) = Σ 1/(k + rank_i(d))。
from __future__ import annotations

from app.retrieval.milvus_search import Candidate


def reciprocal_rank_fuse(
    routes: list[list[Candidate]], k: int = 60
) -> list[Candidate]:
    """多路 RRF 融合,等 #3 接。MVP:占位 — NotImplementedError。"""
    raise NotImplementedError("RRF — implement arithmetic in #3")
