#: `/api/documents` — 文档入库 + 列表 + 删除(MVP 占位)。
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """上传 PDF/MD 触发入库管线。MVP:占位,返回 501。"""
    raise HTTPException(status_code=501, detail="ingestion pipeline — coming in #3")


@router.get("")
async def list_documents() -> list[dict]:
    """列已入库文档。MVP:占位,返回空列表。"""
    return []


@router.delete("/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    """删文档 + 关联 chunk。MVP:占位,返回 501。"""
    raise HTTPException(status_code=501, detail="delete pipeline — coming in #3")
