#: RAG 管线装配 — LangGraph 状态机:retrieve → fuse → rerank → generate → respond。
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

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
def node_retrieve(state: RAGState) -> RAGState:
    """路1 原始 query hybrid + 路2 HyDE hybrid;路3 MCP 占位空。"""
    from app.ingestion.index import get_vector_store

    question = state["question"]
    store = get_vector_store()
    route1 = hybrid_search(question, store=store,
                           top_k=settings.TOP_K_RETRIEVE, alpha=settings.ALPHA)
    hypo = generate_hypothetical_doc(question)
    route2 = hybrid_search(hypo, store=store,
                           top_k=settings.TOP_K_RETRIEVE, alpha=settings.ALPHA) if hypo else []
    route3: list[tuple[ChunkRecord, float]] = []  # TODO: MCP 网搜

    return {
        "candidates": [_to_dict(r, s) for r, s in route1 + route2 + route3],
        "retrieval_trace": {
            "route1": len(route1), "route2": len(route2), "route3": len(route3),
        },
        "route": "local",
    }


def node_fuse(state: RAGState) -> RAGState:
    """RRF 三路融合(兼容路3 占位空)。"""
    from app.ingestion.index import get_vector_store

    store = get_vector_store()
    question = state["question"]
    route1 = hybrid_search(question, store=store,
                           top_k=settings.TOP_K_RETRIEVE, alpha=settings.ALPHA)
    hypo = generate_hypothetical_doc(question)
    route2 = hybrid_search(hypo, store=store,
                           top_k=settings.TOP_K_RETRIEVE, alpha=settings.ALPHA) if hypo else []
    route3: list[tuple[ChunkRecord, float]] = []
    fused = reciprocal_rank_fuse([route1, route2, route3], k=settings.RRF_K)
    return {
        "reranked": [_to_dict(r, s) for r, s in fused],
        "retrieval_trace": {**state.get("retrieval_trace", {}),
                            "rrf": [(r.chunk_id, s) for r, s in fused[:10]]},
    }


def node_rerank(state: RAGState) -> RAGState:
    """二阶精排 top-N 送 LLM。"""
    recs = _collect_records(state)
    reranked = rerank(state["question"], [(r, 0.0) for r in recs],
                      top_n=settings.TOP_N_RERANK)
    top = [r for r, _ in reranked]
    scores = {r.chunk_id: s for r, s in reranked}
    new_trace = dict(state.get("retrieval_trace", {}))
    new_trace["rerank_top"] = [(r.chunk_id, scores.get(r.chunk_id, 0.0))
                               for r in top]
    return {
        "reranked": [_to_dict(r, scores.get(r.chunk_id, 0.0)) for r in top],
        "retrieval_trace": new_trace,
        "_top_records": top,
    }


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


async def ainvoke(question: str, history: list[dict] | None = None) -> dict:
    """单轮问答,返回 {answer, citations, route, retrieval_trace}。"""
    graph = build_graph()
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
    out: dict[str, Any] = await graph.ainvoke(init)
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
            out[cid] = ChunkRecord(text=d.get("text", ""),
                                   doc_id=d.get("doc_id", ""),
                                   chunk_id=cid,
                                   item_name=d.get("item_name", ""),
                                   section=d.get("section", ""))
    # 按 reranked 序
    order = [d.get("chunk_id") for d in state.get("reranked", [])]
    return [out[c] for c in order if c in out]


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
