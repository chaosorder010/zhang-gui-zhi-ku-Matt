#: `/api/query` — RAG 问答,支持 SSE 流式真答 + citation + trace。
#: 单次 LLM 调用:streamer 自己做 LLM 流式,graph 仅跑检索 + rerank(无 node_generate)。
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.graph.pipeline import ainvoke_retrieval
from app.graph.state import Citation
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


def _build_prompt(question: str, top: list[dict], context_size: int = 6000) -> str:
    """拼 context + system prompt;top 是 reranked 后的 chunk dict 列表。"""
    parts: list[str] = []
    n = 0
    for rec in top:
        block = f"[{rec.get('chunk_id', '')}] {rec.get('text', '')}"
        if n + len(block) > context_size:
            break
        parts.append(block)
        n += len(block) + 2
    context = "\n\n".join(parts)
    return (
        "你是「掌柜智库」垂直领域 RAG 助手,中文回答。只依据下述文档片段。"
        "若依据不足,请在答案末尾写「(文档未覆盖)」。"
        "\n\n--- DOCS ---\n" + (context or "(EMPTY)") + "\n--- END DOCS ---\n"
    )


def _to_lc_messages(history: list[dict], system_prompt: str, question: str) -> list:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    lc: list = [SystemMessage(content=system_prompt)]
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
        else:
            ans, rows = await _generate_sync(q, history, rows)
        cited: list[Citation] = [_citation(d) for d in rows]
        _append_history(req.session_id, q, ans)
        return {"answer": ans, "citations": [c.model_dump() for c in cited],
                "route": out["route"],
                **({"retrieval_trace": out["retrieval_trace"]} if req.trace else {})}
    return StreamingResponse(
        _stream(q, history, req.session_id, req.trace), media_type="text/event-stream")


async def _generate_sync(question: str, history: list[dict],
                         reranked: list[dict]) -> tuple[str, list[dict]]:
    """单次 LLM invoke 出答。返回 (answer, reranked top rows)。"""
    prompt = _build_prompt(question, reranked)
    lc = _to_lc_messages(history, prompt, question)
    llm = get_chat_llm()
    try:
        result = llm.invoke(lc)
        ans = (result.content if isinstance(result.content, str)
               else str(result.content))
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM invoke 失败: %s", exc)
        ans = "(LLM 调用失败,请检查 LLM_API_KEY 与 provider)"
    return ans, reranked


async def _stream(q: str, history: list[dict], sid: str | None, trace: bool):
    """SSE 流式:检索 + rerank → 一次 LLM astream → 推 token/done。

    单次 LLM 调用(修复 🔴 review 双 LLM 问题)。
    """
    # 1) 检索 + rerank + route(无 node_generate,不调 LLM)
    out = await ainvoke_retrieval(q, history)
    reranked = out["reranked"]
    route = out["route"]
    trace_obj = out.get("retrieval_trace") if trace else None

    # 2) 路由决策:reject → 直接推 reject 答,不调 LLM
    if route == "reject":
        ans = "文档未覆盖此提问。请换关键词或上传更多文档。"
        yield f"event: meta\ndata: {json.dumps({'route': route, 'citations': [], 'trace': trace_obj}, ensure_ascii=False)}\n\n"
        for piece in _chunk_text(ans):
            yield f"event: token\ndata: {json.dumps({'t': piece}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)
        yield f"event: done\ndata: {json.dumps({'ok': True}, ensure_ascii=False)}\n\n"
        _append_history(sid, q, ans)
        return

    # 3) 推 citation 元数据(含供前端展示的引用预览)
    cited = [_citation(d) for d in reranked]
    yield f"event: meta\ndata: {json.dumps({'route': route, 'citations': [c.model_dump() for c in cited], 'trace': trace_obj}, ensure_ascii=False)}\n\n"

    # 4) 单次 LLM astream
    prompt = _build_prompt(q, reranked)
    lc = _to_lc_messages(history, prompt, q)
    llm = get_chat_llm()
    ans_acc: list[str] = []
    try:
        async for chunk in llm.astream(lc):
            piece = getattr(chunk, "content", None)
            if piece:
                text = str(piece)
                ans_acc.append(text)  # type: ignore[arg-type]
                yield f"event: token\ndata: {json.dumps({'t': text}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM stream 失败,回退预拼答: %s", exc)
        fallback = "(流式生成失败)"
        yield f"event: token\ndata: {json.dumps({'t': fallback}, ensure_ascii=False)}\n\n"
        ans_acc.append(fallback)

    yield f"event: done\ndata: {json.dumps({'ok': True}, ensure_ascii=False)}\n\n"
    _append_history(sid, q, "".join(ans_acc))  # type: ignore[arg-type]


def _citation(rec: dict) -> Citation:
    preview = (rec.get("text", "")[:120] + "…") if len(rec.get("text", "")) > 120 else rec.get("text", "")
    return Citation(doc_id=rec.get("doc_id", ""), chunk_id=rec.get("chunk_id", ""),
                    preview=preview, item_name=rec.get("item_name", ""),
                    score=float(rec.get("score", 0.0)))


def _chunk_text(text: str, size: int = 8) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]
