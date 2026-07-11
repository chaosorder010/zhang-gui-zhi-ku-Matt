#: RAG 状态机共享状态。
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """单条引用。"""

    doc_id: str
    chunk_id: str
    preview: str


class RAGState(BaseModel):
    """LangGraph 状态。MVP:占位 schema,等 #3 接 LangGraph。"""

    question: str
    history: list[dict] = Field(default_factory=list)
    route: Literal["local", "web", "reject"] = "local"
    candidates: list[dict] = Field(default_factory=list)
    reranked: list[dict] = Field(default_factory=list)
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    retrieval_trace: dict = Field(default_factory=dict)
