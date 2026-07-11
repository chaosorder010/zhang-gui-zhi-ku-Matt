#: RAG 管线装配 — LangGraph 状态机:retrieve → (generate|respond)。
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal

from langgraph.graph import StateGraph, END, START  # type: ignore[import-untyped]

from app.core.config import settings
from app.graph.state import Citation, RAGState
from app.retrieval.hyde import generate_hypothetical_doc
from app.retrieval.milvus_search import hybrid_search
from app.retrieval.rerank import rerank
from app.retrieval.rrf import reciprocal_rank_fuse
from app.services.llm import get_chat_llm
from app.storage.vector_store import ChunkRecord

log = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 6000


# --------------------------------------------------------------------------- #
# 检索
# --------------------------------------------------------------------------- #
def _do_retrieve(question: str) -> tuple[
    list[tuple[ChunkRecord, float]],
    list[tuple[ChunkRecord, float]],
    list[tuple[ChunkRecord, float]],
]:
    """单 query 出三路原始召回(路1 原 query,路2 HyDE,路3 MCP 占位空)。"""
    from app.ingestion.index import get_vector_store  # noqa: PLC0415

    store = get_vector_store()
    route1 = hybrid_search(question, store=store,
                           top_k=settings.TOP_K_RETRIEVE, alpha=settings.ALPHA)
    hypo = generate_hypothetical_doc(question)
    route2 = (hybrid_search(hypo, store=store,
                            top_k=settings.TOP_K_RETRIEVE, alpha=settings.ALPHA)
              if hypo else [])
    route3: list[tuple[ChunkRecord, float]] = []  # TODO: MCP 网搜
    return route1, route2, route3


def node_retrieve(state: RAGState) -> RAGState:
    """单 query 三路召回 + RRF + 二阶 rerank + 路由决策。

    整个图只此节点做 embed/hybrid_search;route 在 retrieve 内部决策,
    generate 节点不调 LLM(由 API handler 流式时调)。
    """
    route1, route2, route3 = _do_retrieve(state["question"])
    fused = reciprocal_rank_fuse([route1, route2, route3], k=settings.RRF_K)
    reranked = rerank(state["question"], fused,
                      top_n=settings.TOP_N_RERANK)
    top = [r for r, _ in reranked]
    scores = {r.chunk_id: s for r, s in reranked}
    max_score = max(scores.values()) if scores else 0.0
    route: Literal["local", "web", "reject"] = (
        "reject" if not top or max_score <= settings.REJECT_THRESHOLD
        else "local")
    return {
        "candidates": [_to_dict(r, s) for r, s in route1 + route2 + route3],
        "reranked": [_to_dict(r, scores.get(r.chunk_id, 0.0)) for r in top],
        "_top_records": top,
        "route": route,
        "retrieval_trace": {
            "route1": len(route1), "route2": len(route2), "route3": len(route3),
            "rrf": [(r.chunk_id, s) for r, s in fused[:10]],
            "rerank_top": [(r.chunk_id, scores.get(r.chunk_id, 0.0)) for r in top],
            "route": route,
        },
    }


def node_generate(state: RAGState) -> RAGState:
    """拼 context + 历史 → LLM 生成带 citation 答。同步 invoke,供非 stream 调用。"""
    top = state.get("_top_records", [])
    history = state.get("history") or []
    lc = _build_messages(history, top, state["question"])
    llm = get_chat_llm()
    try:
        result = llm.invoke(lc)
        answer = (result.content if isinstance(result.content, str)
                  else str(result.content))
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM generate 失败: %s", exc)
        answer = "(LLM 调用失败,请检查 LLM_API_KEY 与 provider)"
    cited = [_citation(r) for r in top]
    return {"answer": answer,
            "citations": [c.model_dump() for c in cited],
            "route": "local"}


def node_reject(state: RAGState) -> RAGState:
    return {
        "answer": "文档未覆盖此提问。请换关键词或上传更多文档。",
        "citations": [],
        "route": "reject",
    }


# --------------------------------------------------------------------------- #
# 装配
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def build_graph():  # type: ignore[no-any-return]
    g = StateGraph(RAGState)  # type: ignore[arg-type]
    g.add_node("retrieve", node_retrieve)
    g.add_node("generate", node_generate)
    g.add_node("reject", node_reject)

    g.add_edge(START, "retrieve")
    g.add_conditional_edges(
        "retrieve",
        lambda s: s.get("route", "local"),
        {"local": "generate", "reject": "reject", "web": "reject"})
    g.add_edge("generate", END)
    g.add_edge("reject", END)
    return g.compile()


@lru_cache(maxsize=1)
def build_retrieval_graph():  # type: ignore[no-any-return]
    """仅 retrieve 节点,不调 LLM。供 /api/query 流式 handler。"""
    g = StateGraph(RAGState)  # type: ignore[arg-type]
    g.add_node("retrieve", node_retrieve)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", END)
    return g.compile()


async def _run(graph, question: str, history: list[dict] | None = None) -> dict:
    init: RAGState = {
        "question": question,
        "history": history or [],
        "route": "local",
        "candidates": [],
        "reranked": [],
        "answer": "",
        "citations": [],
        "retrieval_trace": {},
        "_top_records": [],
    }
    return await graph.ainvoke(init)  # type: ignore[arg-type]


async def ainvoke_retrieval(question: str, history: list[dict] | None = None) -> dict:
    """仅检索 + rerank,不调 LLM。返回 {reranked, route, retrieval_trace}。"""
    out = await _run(build_retrieval_graph(), question, history)
    return {
        "reranked": out.get("reranked", []),
        "route": out.get("route", "local"),
        "retrieval_trace": out.get("retrieval_trace", {}),
    }


async def ainvoke(question: str, history: list[dict] | None = None) -> dict:
    """单轮问答,返回 {answer, citations, route, retrieval_trace}。向后兼容。"""
    out = await _run(build_graph(), question, history)
    return {
        "answer": out.get("answer", ""),
        "citations": out.get("citations", []),
        "route": out.get("route", "local"),
        "retrieval_trace": out.get("retrieval_trace", {}),
    }


# --------------------------------------------------------------------------- #
# 公共 helpers(query.py 也 import,消除重复)
# --------------------------------------------------------------------------- #
def _to_dict(rec: ChunkRecord, score: float) -> dict:
    return {"chunk_id": rec.chunk_id, "doc_id": rec.doc_id,
            "item_name": rec.item_name, "text": rec.text,
            "section": rec.section, "score": score}


def build_messages(history: list[dict], top_records: list[dict],
                   question: str, context_size: int = MAX_CONTEXT_CHARS) -> list:
    """拼 system prompt(含 context) + 历史 + question → LangChain 消息列表。

    ``top_records`` 是 reranked 后的 chunk dict 列表(每 dict 含 text/chunk_id/...)。
    """
    context = format_context(top_records, context_size)
    system = system_prompt(context)
    return _to_lc_messages(history, system, question)


def format_context(top: list[Any], context_size: int = MAX_CONTEXT_CHARS) -> str:
    """chunk dict 列表 → 拼好的 context string,总长不超过 context_size 字符。"""
    parts: list[str] = []
    n = 0
    for r in top:
        text = r["text"] if isinstance(r, dict) else r.text
        chunk_id = r.get("chunk_id", "") if isinstance(r, dict) else r.chunk_id
        block = f"[{chunk_id}] {text}"
        if n + len(block) > context_size:
            break
        parts.append(block)
        n += len(block) + 2
    return "\n\n".join(parts)


def system_prompt(context: str) -> str:
    return (
        "你是「掌柜智库」垂直领域 RAG 助手,中文回答。只依据下述文档片段。"
        "若依据不足,请在答案末尾写「(文档未覆盖)」。"
        "\n\n--- DOCS ---\n" + (context or "(EMPTY)") + "\n--- END DOCS ---\n"
    )


def make_citation(rec: Any) -> Citation:
    """chunk dict 或 ChunkRecord → Citation。"""
    if isinstance(rec, dict):
        preview = rec.get("text", "")[:120]
        if len(rec.get("text", "")) > 120:
            preview += "…"
        return Citation(
            doc_id=rec.get("doc_id", ""), chunk_id=rec.get("chunk_id", ""),
            preview=preview, item_name=rec.get("item_name", ""),
            score=float(rec.get("score", 0.0)))
    preview = (rec.text[:120] + "…") if len(rec.text) > 120 else rec.text
    return Citation(doc_id=rec.doc_id, chunk_id=rec.chunk_id,
                    preview=preview, item_name=rec.item_name, score=0.0)


def _build_messages(history: list[dict], top: list[ChunkRecord],
                    question: str) -> list:
    """内部:ChunkRecord 列表 → LangChain 消息列表(用于非 stream 同步 invoke)。"""
    context = format_context(top)
    system = system_prompt(context)
    return _to_lc_messages(history, system, question)


def _to_lc_messages(history: list[dict], system_prompt_str: str,
                    question: str) -> list:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    lc: list = [SystemMessage(content=system_prompt_str)]
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant":
            lc.append(AIMessage(content=content))
        elif role == "system":
            lc.append(SystemMessage(content=content))
        else:
            lc.append(HumanMessage(content=content))
    lc.append(HumanMessage(content=question))
    return lc


def _citation(rec: ChunkRecord) -> Citation:
    preview = (rec.text[:120] + "…") if len(rec.text) > 120 else rec.text
    return Citation(doc_id=rec.doc_id, chunk_id=rec.chunk_id,
                    preview=preview, item_name=rec.item_name)


#: 兼容旧调用方
_rec_from_dict = None  # type: ignore[assignment]
