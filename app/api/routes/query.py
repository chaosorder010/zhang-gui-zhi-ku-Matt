#: `/api/query` — RAG 问答,支持 SSE 流式真答 + citation + trace。
#: 单次 LLM 调用:streamer 自己做 LLM 流式,graph 仅跑 retrieve(无 node_generate)。
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.graph.pipeline import (
    build_messages, ainvoke_retrieval, make_citation,
)
from app.services.llm import get_chat_llm

log = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    session_id: str | None = Field(default=None, description="会话 ID,多轮")
    trace: bool = Field(default=False, description="是否返回检索 trace")
    stream: bool = Field(default=True, description="是否 SSE 流式")


# 内存 session 历史(进程内,多轮用;MVP 非持久)
_SESSIONS: dict[str, list[dict]] = {}


def _history_for(session_id: str | None) -> list[dict]:
    if not session_id:
        return []
    return _SESSIONS.get(session_id, [])


def _append_history(session_id: str | None, q: str, a: str) -> None:
    if not session_id:
        return
    lst = _SESSIONS.setdefault(session_id, [])
    lst.append({"role": "user", "content": q})
    lst.append({"role": "assistant", "content": a})
    if len(lst) > 24:
        _SESSIONS[session_id] = lst[-24:]


@router.post("")
async def query(req: QueryRequest):
    """同步或流式问答。stream=True 走 SSE,False 走一次性 JSON。"""
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=422, detail="question 不能为空")
    history = _history_for(req.session_id)
    if not req.stream:
        # 一次性 JSON:检索 + LLM 一次 invoke
        out = await ainvoke_retrieval(q, history)
        rows = out["reranked"] if out["route"] != "reject" else []
        if out["route"] == "reject":
            ans: str = "文档未覆盖此提问。请换关键词或上传更多文档。"
            cited = []
        else:
            ans = await _generate_sync(q, history, rows)
            cited = [make_citation(d) for d in rows]
        _append_history(req.session_id, q, ans)
        return {"answer": ans,
                "citations": [c.model_dump() for c in cited],
                "route": out["route"],
                **({"retrieval_trace": out["retrieval_trace"]} if req.trace else {})}
    return StreamingResponse(
        _stream(q, history, req.session_id, req.trace), media_type="text/event-stream")


async def _generate_sync(question: str, history: list[dict],
                         reranked: list[dict]) -> str:
    """单次 LLM invoke 出答。
    修复 🔴 /code-review 双 LLM 问题:此路径为唯一 LLM 调用。
    """
    lc = build_messages(history, reranked, question)
    llm = get_chat_llm()
    try:
        result = llm.invoke(lc)
        return (result.content if isinstance(result.content, str)
                else str(result.content))
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM invoke 失败: %s", exc)
        return "(LLM 调用失败,请检查 LLM_API_KEY 与 provider)"


async def _stream(q: str, history: list[dict], sid: str | None, trace: bool):
    """SSE 流式:retrieve(仅)→ 单次 LLM astream → 推 token/done。

    修复 🔴 /code-review:*无 node_generate* — ainvoke_retrieval 不调 LLM,
    handler 自己调一次 astream(或 reject 路径 0 次 LLM)。
    """
    # 1) 仅检索 + rerank + route(无 node_generate,不调 LLM)
    out = await ainvoke_retrieval(q, history)
    reranked = out["reranked"]
    route = out["route"]
    trace_obj = out.get("retrieval_trace") if trace else None

    # 2) reject 路径:直接推 reject 答,不调 LLM
    if route == "reject":
        ans = "文档未覆盖此提问。请换关键词或上传更多文档。"
        yield f"event: meta\ndata: {json.dumps({'route': route, 'citations': [], 'trace': trace_ok(trace_obj)}, ensure_ascii=False)}\n\n"
        for piece in _chunk_text(ans):
            yield f"event: token\ndata: {json.dumps({'t': piece}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)
        yield f"event: done\ndata: {json.dumps({'ok': True}, ensure_ascii=False)}\n\n"
        _append_history(sid, q, ans)
        return

    # 3) 推 citation 元数据
    cited = [make_citation(d) for d in reranked]
    yield f"event: meta\ndata: {json.dumps({'route': route, 'citations': [c.model_dump() for c in cited], 'trace': trace_ok(trace_obj)}, ensure_ascii=False)}\n\n"

    # 4) 单次 LLM astream — 整个请求生命周期的唯一 LLM 调用
    lc = build_messages(history, reranked, q)
    llm = get_chat_llm()
    ans_acc: list[str] = []
    try:
        async for chunk in llm.astream(lc):
            piece = getattr(chunk, "content", None)
            if isinstance(piece, str) and piece:
                ans_acc.append(piece)
                yield f"event: token\ndata: {json.dumps({'t': piece}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM stream 失败,回退简讯: %s", exc)
        fallback = "(流式生成失败)"
        yield f"event: token\ndata: {json.dumps({'t': fallback}, ensure_ascii=False)}\n\n"
        ans_acc.append(fallback)

    yield f"event: done\ndata: {json.dumps({'ok': True}, ensure_ascii=False)}\n\n"
    _append_history(sid, q, "".join(ans_acc))


def trace_ok(trace_obj):
    """trace 序列化,None 返 {}。"""
    return trace_obj if trace_obj is not None else {}


def _chunk_text(text: str, size: int = 8) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]
