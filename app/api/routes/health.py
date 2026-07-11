#: `/api/health` — 后端 + Milvus/Mongo/MinIO 共答健康探测。
from fastapi import APIRouter

from app.core.events import _probe_milvus, _probe_mongo, _probe_minio

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """后端 + 三依赖健康共答。"""
    milvus_ok, milvus_msg = await _probe_milvus()
    mongo_ok, mongo_msg = await _probe_mongo()
    minio_ok, minio_msg = await _probe_minio()

    return {
        "backend": {"status": "ok"},
        "milvus": {"status": "ok" if milvus_ok else "unhealthy", "detail": milvus_msg},
        "mongo": {"status": "ok" if mongo_ok else "unhealthy", "detail": mongo_msg},
        "minio": {"status": "ok" if minio_ok else "unhealthy", "detail": minio_msg},
    }
