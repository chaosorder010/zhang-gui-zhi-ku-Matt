#: VectorStore 抽象 + 双实现。
from __future__ import annotations

import abc
import logging
import os
from dataclasses import dataclass, field

from app.core.config import settings

log = logging.getLogger(__name__)

_STORE: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """进程内 VectorStore 单例。MILVUS_FALLBACK=1 走 InMemory。"""
    global _STORE
    if _STORE is not None:
        return _STORE
    if os.getenv("MILVUS_FALLBACK") == "1":
        _STORE = InMemoryVectorStore()
    else:
        v = MilvusVectorStore()
        v.ensure()
        _STORE = v
    return _STORE


def rebuild_vector_store(store: VectorStore) -> None:
    """测试期替换后端。"""
    global _STORE
    _STORE = store


@dataclass
class ChunkRecord:
    """入/出库的统一记录。"""

    text: str
    doc_id: str
    chunk_id: str
    item_name: str
    section: str = ""
    seq: int = 0
    token_len: int = 0
    dense: list[float] = field(default_factory=list)
    sparse: dict[int, float] = field(default_factory=dict)


class VectorStore(abc.ABC):
    """后端抽象:Milvus 实 / InMemory 测试。"""

    @abc.abstractmethod
    def ensure(self) -> None:
        ...

    @abc.abstractmethod
    def upsert(self, chunks: list[ChunkRecord]) -> int:
        ...

    @abc.abstractmethod
    def delete(self, doc_id: str) -> int:
        ...

    @abc.abstractmethod
    def search(self, dense_vec: list[float] | None, sparse_vec: dict[int, float] | None,
               top_k: int, alpha: float = 0.7) -> list[tuple[ChunkRecord, float]]:
        """hybrid score = α·dense + (1-α)·sparse。"""
        ...

    @abc.abstractmethod
    def list_doc_ids(self) -> list[str]:
        ...


# --------------------------------------------------------------------------- #
# 相似度
# --------------------------------------------------------------------------- #
def _cos_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(y * y for y in b) ** 0.5 or 1.0
    return dot / (na * nb)


def _sparse_sim(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    inter = set(a) & set(b)
    if not inter:
        return 0.0
    dot = sum(a[k] * b[k] for k in inter)
    na = sum(v * v for v in a.values()) ** 0.5 or 1.0
    nb = sum(v * v for v in b.values()) ** 0.5 or 1.0
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# InMemoryVectorStore
# --------------------------------------------------------------------------- #
class InMemoryVectorStore(VectorStore):
    """无外依赖实现,供冒烟/单元。"""

    def __init__(self) -> None:
        self._data: dict[str, ChunkRecord] = {}  # chunk_id → rec

    def ensure(self) -> None:
        return

    def upsert(self, chunks: list[ChunkRecord]) -> int:
        for c in chunks:
            self._data[c.chunk_id] = c
        return len(chunks)

    def delete(self, doc_id: str) -> int:
        victim = [cid for cid, c in self._data.items() if c.doc_id == doc_id]
        for cid in victim:
            self._data.pop(cid, None)
        return len(victim)

    def search(self, dense_vec, sparse_vec, top_k, alpha: float = 0.7):
        scored: list[tuple[ChunkRecord, float]] = []
        for rec in self._data.values():
            d = _cos_sim(dense_vec, rec.dense) if dense_vec and rec.dense else 0.0
            s = _sparse_sim(sparse_vec, rec.sparse) if sparse_vec and rec.sparse else 0.0
            score = alpha * d + (1.0 - alpha) * s
            scored.append((rec, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def list_doc_ids(self) -> list[str]:
        return sorted({c.doc_id for c in self._data.values()})


# --------------------------------------------------------------------------- #
# MilvusVectorStore  (pymilvus 3.x MilvusClient)
# --------------------------------------------------------------------------- #
class MilvusVectorStore(VectorStore):
    """pymilvus 3.x hybrid collection。连不上时降级 InMemoryVectorStore。"""

    def __init__(self, uri: str = settings.MILVUS_URI,
                 collection: str = settings.MILVUS_COLLECTION) -> None:
        self._uri = uri
        self._collection = collection
        self._client = None  # type: ignore[var-annotated]
        self._in_memory = InMemoryVectorStore()

    def _connect(self) -> bool:
        if self._client is not None:
            return True
        try:
            from pymilvus import MilvusClient  # type: ignore[import-untyped]

            self._client = MilvusClient(uri=self._uri)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Milvus 连不上(%s),降级 InMemory: %s", self._uri, exc)
            self._client = None  # type: ignore[assignment]
            return False

    def _active(self) -> VectorStore:
        if self._connect() and self._client is not None:
            return self
        return self._in_memory

    def ensure(self) -> None:
        store = self._active()
        if store is not self:
            return store.ensure()
        client = self._client
        if client is None:
            return
        try:
            if client.has_collection(self._collection):
                return
            self._ensure_schema()
            self._ensure_indexes()
        except Exception as exc:  # noqa: BLE001
            log.warning("ensure schema/index 失败: %s", exc)

    def _ensure_schema(self) -> None:
        from pymilvus import DataType  # type: ignore[import-untyped]

        client = self._client
        if client is None or not client.has_collection(self._collection):
            return
        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=128)
        schema.add_field("item_name", DataType.VARCHAR, max_length=128)
        schema.add_field("section", DataType.VARCHAR, max_length=256)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=settings.EMBED_DENSE_DIM)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        try:
            client.create_collection(collection_name=self._collection, schema=schema)
        except Exception as exc:  # noqa: BLE001
            log.warning("create_collection 失败: %s", exc)

    def _ensure_indexes(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            client.create_index(self._collection, "dense",
                                {"index_type": "HNSW", "metric_type": "IP",
                                 "params": {"M": 32, "efConstruction": 200}})
            client.create_index(self._collection, "sparse",
                                {"index_type": "SPARSE_INVERTED_INDEX",
                                 "metric_type": "IP",
                                 "params": {"drop_ratio_build": 0.2}})
            client.load_collection(self._collection)
        except Exception as exc:  # noqa: BLE001
            log.warning("建索引/装载失败,降级: %s", exc)

    def upsert(self, chunks: list[ChunkRecord]) -> int:
        store = self._active()
        if store is not self:
            return store.upsert(chunks)
        client = self._client
        if client is None:
            return self._in_memory.upsert(chunks)
        rows = [
            {"doc_id": c.doc_id, "chunk_id": c.chunk_id,
             "item_name": c.item_name, "section": c.section, "text": c.text,
             "dense": c.dense, "sparse": c.sparse} for c in chunks]
        try:
            client.upsert(self._collection, rows)
            return len(rows)
        except Exception as exc:  # noqa: BLE001
            log.warning("Milvus upsert 失败,降级: %s", exc)
            self._client = None  # type: ignore[assignment]
            return self._in_memory.upsert(chunks)

    def delete(self, doc_id: str) -> int:
        store = self._active()
        if store is not self:
            return store.delete(doc_id)
        client = self._client
        if client is None:
            return self._in_memory.delete(doc_id)
        try:
            res = client.delete(self._collection, filter=f'doc_id == "{doc_id}"')
            cnt = int(res.get("delete_count", 0)) if isinstance(res, dict) else len(res or [])
            return cnt
        except Exception as exc:  # noqa: BLE001
            log.warning("Milvus delete 失败,降级: %s", exc)
            self._client = None  # type: ignore[assignment]
            return self._in_memory.delete(doc_id)

    def search(self, dense_vec, sparse_vec, top_k, alpha: float = 0.7):
        """Milvus 路径用内置 RRF 重排,alpha 仅作用于 InMemoryVectorStore。

        Milvus 内建 RRF 不受 alpha 控制。若 caller 显式传 alpha != settings.ALPHA,
        抛 AssertionError 提示契约(避免静默忽略导致调参假象)。
        """
        if alpha != settings.ALPHA:
            raise AssertionError(
                f"MilvusVectorStore 路径不使用 alpha(内置 RRF)。"
                f"如需 α={alpha} 控制,改用 InMemoryVectorStore 或在外部后处理。")
        store = self._active()
        if store is not self:
            return store.search(dense_vec, sparse_vec, top_k, alpha)
        if not dense_vec and not sparse_vec:
            return []
        client = self._client
        if client is None:
            return self._in_memory.search(dense_vec, sparse_vec, top_k, alpha)
        try:
            return self._hybrid_search(dense_vec, sparse_vec, top_k)
        except Exception as exc:  # noqa: BLE001
            log.warning("Milvus hybrid_search 失败,降级: %s", exc)
            self._client = None  # type: ignore[assignment]
            return self._in_memory.search(dense_vec, sparse_vec, top_k, alpha)

    def _hybrid_search(self, dense_vec, sparse_vec, top_k):
        from pymilvus import AnnSearchRequest  # type: ignore[import-untyped]

        client = self._client
        assert client is not None  # 调用方已窄化
        reqs: list = []
        if dense_vec:
            reqs.append(AnnSearchRequest(
                data=[dense_vec], anns_field="dense",
                param={"metric_type": "IP", "params": {"ef": 128}},
                limit=top_k))
        if sparse_vec:
            reqs.append(AnnSearchRequest(
                data=[sparse_vec], anns_field="sparse",
                param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
                limit=top_k))
        if not reqs:
            return []
        hits = client.hybrid_search(
            self._collection, reqs,
            rerank={"strategy": "rrf", "params": {"k": 60}},
            limit=top_k,
            output_fields=["doc_id", "chunk_id", "item_name", "text", "section"])
        # pymilvus 3.x 对多 AnnSearchRequest 返回 list[list[Hit]],需展平。
        # 单请求时也可能返回 [Hit],故统一展平一层。
        flat: list = []
        for item in hits or []:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        results: list[tuple[ChunkRecord, float]] = []
        seen: set[str] = set()
        for hit in flat:
            ent = hit.get("entity", {})
            cid = ent.get("chunk_id", "")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            rec = ChunkRecord(
                text=ent.get("text", ""),
                doc_id=ent.get("doc_id", ""),
                chunk_id=cid,
                item_name=ent.get("item_name", ""),
                section=ent.get("section", ""),
            )
            results.append((rec, float(hit.get("distance", 0.0))))
        return results

    def list_doc_ids(self) -> list[str]:
        store = self._active()
        if store is not self:
            return store.list_doc_ids()
        client = self._client
        if client is None:
            return self._in_memory.list_doc_ids()
        try:
            out = client.query(self._collection, filter="",
                               output_fields=["doc_id"])
            ids = sorted({row.get("doc_id", "") for row in out if row.get("doc_id")})
            return ids
        except Exception as exc:  # noqa: BLE001
            log.warning("Milvus query doc_ids 失败,降级: %s", exc)
            self._client = None  # type: ignore[assignment]
            return self._in_memory.list_doc_ids()
