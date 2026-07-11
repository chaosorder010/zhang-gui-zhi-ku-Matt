#: `/api/documents` — 文档入库 + 列表 + 删除。
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.ingestion.pipeline import run_ingestion
from app.ingestion.index import list_documents as _list, delete_by_doc

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_BYTES = 50 * 1024 * 1024


@router.post("")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """上传 PDF/MD 触发入库管线。坏文件 → 400/415 明报。"""
    ctype = file.content_type
    filename = file.filename or "upload.bin"
    try:
        data = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"读取失败: {exc}") from exc
    if not data:
        raise HTTPException(status_code=400, detail=f"[{filename}] 空文件")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"[{filename}] 超 50MB 限制")
    try:
        result = await run_ingestion(data, filename, ctype)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"入库失败: {type(exc).__name__}") from exc
    if not result.get("chunk_count"):
        raise HTTPException(status_code=400,
                            detail=f"[{filename}] 未产出任何 chunk(可能空文件/扫描版)")
    return result


@router.get("")
async def list_documents() -> list[dict]:
    """列已入库文档 metadata。"""
    return await _list()


@router.delete("/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    """删文档 + 关联 chunk。"""
    try:
        n = await delete_by_doc(doc_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"删除失败: {type(exc).__name__}") from exc
    if n == 0:
        raise HTTPException(status_code=404, detail=f"文档 {doc_id} 不存在")
    return {"deleted_chunks": n, "doc_id": doc_id}
