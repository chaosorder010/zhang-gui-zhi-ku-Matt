#: `/api/query` — RAG 问答,支持 SSE 流式真答 + citation + trace。
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.graph.pipeline import ainvoke
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
    """同步或流式问答。"""
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=422, detail="question 不能为空")
    history = _history_for(req.session_id)
    if not req.stream:
        out = await ainvoke(q, history)
        _append_history(req.session_id, q, out["answer"])
        return {"answer": out["answer"], "citations": out["citations"],
                "route": out["route"],
                **({"retrieval_trace": out["retrieval_trace"]} if req.trace else {})}
    return StreamingResponse(_stream(q, history, req.session_id, req.trace),
                             media_type="text/event-stream")


async def _stream(q: str, history: list[dict], sid: str | None, trace: bool):
    """SSE 流式:先发 citation 元数据,再流式 LLM 真答。"""
    # 1) 检索 + rerank(同步,快)
    out = await ainvoke(q, history)
    citations = out["citations"]
    route = out["route"]
    trace_obj = out["retrieval_trace"] if trace else None

    # 2) 推 citation 元数据
    yield f"event: meta\ndata: {json.dumps({'route': route, 'citations': citations, 'trace': trace_obj}, ensure_ascii=False)}\n\n"

    # 3) 流式 LLM 真答
    answer = out["answer"]
    try:
        llm = get_chat_llm()
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        lc_msgs = []
        for m in history:
            role = m.get("role", "user")
            content = m.get("content", "")
            msgs_to_append: list
            if role == "system":
                msgs_to_append = [SystemMessage(content=content)]
            elif role == "assistant":
                msgs_to_append = [AIMessage(content=content)]
            else:
                msgs_to_append = [HumanMessage(content=content)]
            lc_msgs += msgs_to_append
        lc_msgs.append(HumanMessage(content=q))
        async for chunk in llm.astream(lc_msgs):
            piece = getattr(chunk, "content", None)
            if piece:
                yield f"event: token\ndata: {json.dumps({'t': piece}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM stream 失败,回退一次性答: %s", exc)
        for piece in _chunk_text(answer):
            yield f"event: token\ndata: {json.dumps({'t': piece}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)

    yield f"event: done\ndata: {json.dumps({'ok': True}, ensure_ascii=False)}\n\n"
    _append_history(sid, q, answer)


def _chunk_text(text: str, size: int = 8) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]
