#: RAG 管线装配 — LangGraph 状态机:retrieve → fuse → rerank → generate → respond。
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal

from langgraph.graph import StateGraph, END, START  # type: ignore[import-untyped]

from app.core.config import settings
from app.graph.state import RAGState
from app.retrieval.hyde import generate_hypothetical_doc
from app.retrieval.milvus_search import hybrid_search
from app.retrieval.rerank import rerank
from app.retrieval.rrf import reciprocal_rank_fuse
from app.services.llm import get_chat_llm
from app.storage.vector_store import ChunkRecord

log = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 6000


# --------------------------------------------------------------------------- #
# 节点
# --------------------------------------------------------------------------- #
def _do_retrieve(question: str) -> tuple[
    list[tuple[ChunkRecord, float]],
    list[tuple[ChunkRecord, float]],
    list[tuple[ChunkRecord, float]],
]:
    """单 query 出三路原始召回(路1 原 query,路2 HyDE,路3 MCP 占位空)。"""
    from app.ingestion.index import get_vector_store

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
    """单 query 一次 embed 出三路原始召回 + RRF + 二阶 rerank + 路由决策。

    整个图只此节点做一次 embed/hybrid_search,下游 fuse/rerank 做无 ops 透传(保拓扑可拓展)。
    """
    route1, route2, route3 = _do_retrieve(state["question"])
    fused = reciprocal_rank_fuse([route1, route2, route3], k=settings.RRF_K)
    reranked = rerank(state["question"], fused,
                      top_n=settings.TOP_N_RERANK)
    top = [r for r, _ in reranked]
    scores = {r.chunk_id: s for r, s in reranked}
    max_score = max(scores.values()) if scores else 0.0
    route: Literal["local", "web", "reject"] = (
        "reject" if not top or max_score <= settings.REJECT_THRESHOLD else "local")
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


# fuse / rerank 节点是占位无 ops 透传,预留后续多级扩展。
def node_fuse(state: RAGState) -> RAGState:
    return {"retrieval_trace": state.get("retrieval_trace", {})}


def node_rerank(state: RAGState) -> RAGState:
    new_trace = dict(state.get("retrieval_trace", {}))
    new_trace["rerank_top"] = [(d.get("chunk_id"), d.get("score", 0.0))
                               for d in state.get("reranked", [])]
    return {"retrieval_trace": new_trace}


def _decide(state: RAGState) -> str:
    top = state.get("_top_records", [])
    if not top:
        return "reject"
    scores = [float(d.get("score", 0.0)) for d in state.get("reranked", [])]
    if scores and max(scores) <= settings.REJECT_THRESHOLD:
        return "reject"
    return "generate"


def node_generate(state: RAGState) -> RAGState:
    """拼 context + 历史 → LLM 生成带 citation 答(流式在路由外)。"""
    top = state.get("_top_records", [])
    context = _format_context(top)
    history = state.get("history") or []
    msgs = list(history)
    msgs.append({"role": "system", "content": _system_prompt(context)})
    msgs.append({"role": "user", "content": state["question"]})
    llm = get_chat_llm()
    try:
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        lc_msgs: list[Any] = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lc_msgs.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_msgs.append(AIMessage(content=content))
            else:
                lc_msgs.append(HumanMessage(content=content))
        result = llm.invoke(lc_msgs)
        answer = (result.content if isinstance(result.content, str)
                  else str(result.content))
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM generate 失败: %s", exc)
        answer = "(LLM 调用失败,请检查 LLM_API_KEY 与 provider)"
    cited: list[Any] = [_citation(r) for r in top]
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
    g.add_node("fuse", node_fuse)
    g.add_node("rerank", node_rerank)
    g.add_node("generate", node_generate)
    g.add_node("reject", node_reject)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "fuse")
    g.add_edge("fuse", "rerank")
    g.add_conditional_edges(
        "rerank", _decide,
        {"generate": "generate", "reject": "reject"})
    g.add_edge("generate", END)
    g.add_edge("reject", END)
    return g.compile()


def build_retrieval_graph():  # type: ignore[no-any-return]
    """仅检索 + rerank + 路由决策,不调 LLM。供 /api/query 流式 handler 用。"""
    g = StateGraph(RAGState)  # type: ignore[arg-type]
    g.add_node("retrieve", node_retrieve)
    g.add_node("fuse", node_fuse)
    g.add_node("rerank", node_rerank)

    def _decide_end(state: RAGState) -> str:
        r = _decide(state)
        return "end" if r == "reject" else "end"

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "fuse")
    g.add_edge("fuse", "rerank")
    g.add_edge("rerank", END)
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
# helpers
# --------------------------------------------------------------------------- #
def _to_dict(rec: ChunkRecord, score: float) -> dict:
    return {"chunk_id": rec.chunk_id, "doc_id": rec.doc_id,
            "item_name": rec.item_name, "text": rec.text,
            "section": rec.section, "score": score}


def _collect_records(state: RAGState) -> list[ChunkRecord]:
    """从 reranked 列表拼记录(同 chunk_id 优先取 _top_records)。"""
    out: dict[str, ChunkRecord] = {}
    for r in state.get("_top_records", []):
        out[r.chunk_id] = r
    for d in state.get("reranked", []):
        cid = d.get("chunk_id")
        if cid and cid not in out:
            out[cid] = _rec_from_dict(d)
    order = [d.get("chunk_id") for d in state.get("reranked", [])]
    return [out[c] for c in order if c in out]


def _rec_from_dict(d: dict) -> ChunkRecord:
    return ChunkRecord(text=d.get("text", ""), doc_id=d.get("doc_id", ""),
                       chunk_id=d.get("chunk_id", ""),
                       item_name=d.get("item_name", ""),
                       section=d.get("section", ""))


def _citation(rec: ChunkRecord) -> object:
    from app.graph.state import Citation

    preview = (rec.text[:120] + "…") if len(rec.text) > 120 else rec.text
    return Citation(doc_id=rec.doc_id, chunk_id=rec.chunk_id,
                    preview=preview, item_name=rec.item_name)


def _format_context(top: list[ChunkRecord]) -> str:
    parts: list[str] = []
    n = 0
    for r in top:
        block = f"[{r.chunk_id}] {r.text}"
        if n + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        n += len(block) + 2
    return "\n\n".join(parts)


def _system_prompt(context: str) -> str:
    return (
        "你是「掌柜智库」垂直领域 RAG 助手,中文回答。只依据下述文档片段。"
        "若依据不足,请在答案末尾写“(文档未覆盖)”。"
        "\n\n--- DOCS ---\n" + (context or "(EMPTY)")
        + "\n--- END DOCS ---\n")
