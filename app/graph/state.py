#: RAG 状态机共享状态(LangGraph TypedDict 版 — partial return OK)。
from __future__ import annotations

from typing import Annotated, Any, Literal

from langgraph.graph.message import add_messages  # type: ignore[import-untyped]
from pydantic import BaseModel
from typing_extensions import TypedDict

from app.storage.vector_store import ChunkRecord


class Citation(BaseModel):
    """单条引用。"""

    doc_id: str
    chunk_id: str
    preview: str
    item_name: str = ""
    score: float = 0.0


class RAGState(TypedDict, total=False):
    """LangGraph 状态机 — total=False 让各节点 return partial dict。"""

    question: str
    history: Annotated[list[Any], add_messages]
    route: Literal["local", "web", "reject"]
    candidates: list[dict[str, Any]]
    reranked: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    retrieval_trace: dict[str, Any]
    _top_records: list[ChunkRecord]
