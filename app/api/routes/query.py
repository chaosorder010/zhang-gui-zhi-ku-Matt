#: `/api/query` — RAG 问答(MVP 占位)。
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    session_id: str | None = Field(default=None, description="会话 ID,多轮")
    trace: bool = Field(default=False, description="是否返回检索 trace")


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict] = []
    retrieval_trace: dict | None = None


@router.post("")
async def query(req: QueryRequest) -> QueryResponse:
    """提交问题,流式/同步返回答案 + citation。MVP:占位 501。"""
    raise HTTPException(status_code=501, detail="RAG answering — coming in #3/#4")
