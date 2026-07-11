#: Rerank 二阶精排 — BGE-Reranker / Qwen3-Reranker;无模型时回退 RRF 序。
from __future__ import annotations

import logging

from app.core.config import settings
from app.storage.vector_store import ChunkRecord

log = logging.getLogger(__name__)


class _RerankFallback:
    """无 sentence_transformers 时,用字符 bigram Jaccard 做轻量保序。

    字符 bigram 对中英文都 OK(中文 2-gram 组词、英文 2-gram 分词),
    跨域 query/文本若共享 <2 个 bigram 则 Jaccard ≈ 0(过阈值后不触发),
    同域 query(如"功能")则 bigram「功能」命中 → Jaccard > 0.3。
    """

    def score(self, query: str, cands: list[ChunkRecord]) -> list[float]:
        """Overlap coefficient = |A∩B|/min(|A|,|B|)。

        短 query 命中长 chunk 时仍可得高分(因为分母 min).比 Jaccard 更贴
        合"query bigram 是 chunk 子集"的语义匹配,且跨域 query 与长 chunk 的
        少量 bigram 重合也会被 min 分母压低。
        """
        def _bigrams(t: str) -> set[str]:
            s = t.strip()
            return {s[i:i + 2] for i in range(len(s) - 1)} or {s}
        qg = _bigrams(query)
        if not qg:
            return [0.0] * len(cands)
        out: list[float] = []
        for c in cands:
            cg = _bigrams(c.text)
            inter = len(qg & cg)
            denom = min(len(qg), len(cg)) or 1
            out.append(inter / denom)
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
