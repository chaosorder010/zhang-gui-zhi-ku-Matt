#: FastAPI lifespan — 启停装配 + 三依赖健康探测(pymilvus 3.x)。
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import get_logger, configure_logging

log = get_logger(__name__)


async def _probe_milvus() -> tuple[bool, str]:
    """Milvus 探活。"""
    try:
        from pymilvus import MilvusClient  # type: ignore[import-untyped]

        client = MilvusClient(uri=settings.MILVUS_URI)
        # 触发一次轻量 RPC
        client.list_collections()
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 — 探测不抛
        return False, f"{type(exc).__name__}: {exc}"


async def _probe_mongo() -> tuple[bool, str]:
    """MongoDB 探活。"""
    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        client: AsyncIOMotorClient = AsyncIOMotorClient(
            settings.MONGO_URI, serverSelectionTimeoutMS=2000
        )
        await client.admin.command("ping")
        client.close()
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


async def _probe_minio() -> tuple[bool, str]:
    """MinIO 探活。"""
    try:
        from minio import Minio

        client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        client.bucket_exists(settings.MINIO_BUCKET)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def health_status() -> dict:
    """同步版供非 async 场景,MVP 留空占位。"""
    return {"backend": {"status": "ok"}, "milvus": "deferred",
            "mongo": "deferred", "minio": "deferred"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启:装配日志 + 验必填;停:打一行。"""
    configure_logging()
    log.info("starting %s env=%s", settings.APP_NAME, settings.APP_ENV)
    yield
    log.info("shutting down %s", settings.APP_NAME)
