#: Milvus + Mongo 元数据装配。
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from app.core.config import settings
from app.storage.vector_store import ChunkRecord  # noqa: F401
from app.storage.vector_store import InMemoryVectorStore  # noqa: F401
from app.storage.vector_store import MilvusVectorStore  # noqa: F401
from app.storage.vector_store import VectorStore, get_vector_store  # noqa: F401

log = logging.getLogger(__name__)


def _new_doc_id(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0] or "doc"
    return f"{stem}-{uuid.uuid4().hex[:8]}"


class _MongoStub:
    """mongo 未起或 motor 未装时的进程内兜底。"""

    def __init__(self) -> None:
        self._meta: dict[str, dict] = {}

    def upsert(self, doc_id: str, meta: dict) -> None:
        self._meta[doc_id] = meta

    def list_all(self) -> list[dict]:
        return sorted(self._meta.values(), key=lambda m: m.get("uploaded_at", ""))

    def delete(self, doc_id: str) -> bool:
        return self._meta.pop(doc_id, None) is not None

    def reload(self, metas: list[dict]) -> None:
        self._meta = {m["doc_id"]: m for m in metas}


class _MongoDocs:
    """motor 真实现,失败降级 _MongoStub。"""

    def __init__(self) -> None:
        self._stub = _MongoStub()
        self._coll = None
        self._client = None  # type: ignore[var-annotated]
        try:
            from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore

            client: Any = AsyncIOMotorClient(  # type: ignore[assignment]
                settings.MONGO_URI, serverSelectionTimeoutMS=800)
            self._client = client
            self._coll = client[settings.MONGO_DB]["documents"]
        except Exception as exc:  # noqa: BLE001
            log.warning("motor 连不上,元数据走内存存根: %s", exc)

    async def upsert(self, doc_id: str, meta: dict) -> None:
        if self._coll is None:
            self._stub.upsert(doc_id, meta)
            return
        try:
            await self._coll.update_one(
                {"doc_id": doc_id}, {"$set": meta}, upsert=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("mongo upsert 失败,降级存根: %s", exc)
            self._stub.upsert(doc_id, meta)

    async def list_all(self) -> list[dict]:
        if self._coll is None:
            return self._stub.list_all()
        try:
            return [d async for d in self._coll.find({}, {"_id": 0})]
        except Exception as exc:  # noqa: BLE001
            log.warning("mongo list 失败,降级存根: %s", exc)
            return self._stub.list_all()

    async def delete(self, doc_id: str) -> bool:
        if self._coll is None:
            return self._stub.delete(doc_id)
        try:
            r = await self._coll.delete_one({"doc_id": doc_id})
            return r.deleted_count > 0
        except Exception as exc:  # noqa: BLE001
            log.warning("mongo delete 失败,降级存根: %s", exc)
            return self._stub.delete(doc_id)


_DOCS_STORE: _MongoDocs | None = None


def get_doc_meta_store() -> _MongoDocs:
    """文档元数据存储。进程级单例。"""
    global _DOCS_STORE
    if _DOCS_STORE is None:
        _DOCS_STORE = _MongoDocs()
    return _DOCS_STORE


# 向后兼容:让外部使用者继续 from app.ingestion.index import get_vector_store
# 成立 — 直接委托给 storage 模块的进程内单例。
def rebuild_vector_store(store: VectorStore) -> None:
    """测试期允许替换后端。"""
    from app.storage.vector_store import rebuild_vector_store as _r

    _r(store)



def ensure_collection() -> None:
    get_vector_store().ensure()


def make_records(chunks: list[dict], dense_vecs: list[list[float]],
                 sparse_vecs: list[dict[int, float]]) -> list[ChunkRecord]:
    return [
        ChunkRecord(
            text=c["text"], doc_id=c["doc_id"], chunk_id=c["chunk_id"],
            item_name=c["item_name"], section=c.get("section", ""),
            seq=c.get("seq", 0), token_len=c.get("token_len", 0),
            dense=d, sparse=s,
        )
        for c, d, s in zip(chunks, dense_vecs, sparse_vecs)
    ]


async def upsert_chunks(chunks: list[dict], dense_vecs: list[list[float]],
                        sparse_vecs: list[dict[int, float]]) -> dict:
    """chunks 入 Milvus + Mongo 元数据。返回 {doc_id, item_name, chunk_count}。"""
    if not chunks:
        return {"doc_id": "", "item_name": "", "chunk_count": 0}
    records = make_records(chunks, dense_vecs, sparse_vecs)
    n = get_vector_store().upsert(records)
    doc_id = chunks[0]["doc_id"]
    item_name = chunks[0]["item_name"]
    store = get_doc_meta_store()
    await store.upsert(doc_id, {
        "doc_id": doc_id,
        "title": item_name,
        "uploaded_at": int(time.time()),
        "chunk_count": n,
        "status": "ready",
    })
    return {"doc_id": doc_id, "item_name": item_name, "chunk_count": n}


async def list_documents() -> list[dict]:
    return await get_doc_meta_store().list_all()


async def delete_by_doc(doc_id: str) -> int:
    n = get_vector_store().delete(doc_id)
    await get_doc_meta_store().delete(doc_id)
    return n


__all__ = [
    "ChunkRecord", "VectorStore", "InMemoryVectorStore", "MilvusVectorStore",
    "get_vector_store", "get_doc_meta_store", "rebuild_vector_store",
    "ensure_collection", "upsert_chunks", "list_documents",
    "delete_by_doc", "_new_doc_id",
]
