#: RRF 融合 — score(d) = Σ_routes 1/(k + rank_i(d))。
from __future__ import annotations

from app.storage.vector_store import ChunkRecord


def reciprocal_rank_fuse(
    routes: list[list[tuple[ChunkRecord, float]]], k: int = 60,
) -> list[tuple[ChunkRecord, float]]:
    """多路 RRF 融合。

    ``routes`` 每路是 ``[(ChunkRecord, score), ...]`` 已按 score 降序。
    返回按 RRF 分降序的 ``[(ChunkRecord, score), ...]``,chunk_id 去重。
    """
    if not routes:
        return []
    scores: dict[str, float] = {}
    meta: dict[str, ChunkRecord] = {}
    for route in routes:
        for rank, (rec, _) in enumerate(route):
            key = rec.chunk_id
            if key == "":
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            meta[key] = rec
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(meta[cid], score) for cid, score in ranked if cid in meta]
