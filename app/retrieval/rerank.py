#: Rerank 二阶精排 — BGE-Reranker / Qwen3-Reranker;无模型时回退 RRF 序。
from __future__ import annotations

import logging

from app.core.config import settings
from app.storage.vector_store import ChunkRecord

log = logging.getLogger(__name__)


class _RerankFallback:
    """无 sentence_transformers 时保留 RRF 序(不破坏相关性)。"""

    def score(self, query: str, cands: list[ChunkRecord]) -> list[float]:
        # 用 query 与 chunk 的字符重叠做轻量保序
        qt = set(query)
        out: list[float] = []
        for c in cands:
            ct = set(c.text)
            inter = len(qt & ct)
            union = len(qt | ct) or 1
            out.append(inter / union)
        return out


class _RerankModel:
    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

        self.model = CrossEncoder(settings.RERANK_MODEL, device=settings.RERANK_DEVICE)

    def score(self, query: str, cands: list[ChunkRecord]) -> list[float]:
        pairs = [(query, c.text) for c in cands]
        return [float(s) for s in self.model.predict(pairs)]


class Reranker:
    def __init__(self) -> None:
        self._backend = self._load()

    def _load(self):
        if settings.RERANK_FALLBACK:
            return _RerankFallback()
        try:
            return _RerankModel()
        except ImportError:
            log.warning("sentence_transformers 未装, rerank 回退保序")
            return _RerankFallback()

    def score(self, query: str, cands: list[ChunkRecord]) -> list[float]:
        if not cands:
            return []
        return self._backend.score(query, cands)


def rerank(query: str, candidates: list[tuple[ChunkRecord, float]],
           top_n: int = settings.TOP_N_RERANK) -> list[tuple[ChunkRecord, float]]:
    """精排 top-N 送 LLM。输入 RRF 融合结果,输出按 rerank 分降序。"""
    if not candidates:
        return []
    recs = [c for c, _ in candidates]
    rr = Reranker()
    new_scores = rr.score(query, recs)
    paired = sorted(zip(recs, new_scores), key=lambda kv: kv[1], reverse=True)
    return paired[:top_n]
