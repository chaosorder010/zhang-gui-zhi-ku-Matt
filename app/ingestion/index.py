#: Milvus collection 建 +  chunk 入库 + Mongo 元数据同步。
from __future__]


def ensure_collection() -> None:
    """建 hybrid collection(id, doc_id, chunk_id, item_name, dense, sparse, text)。MVP:占位。"""
    raise NotImplementedError("Milvus ensure_collection — wire up in #3")


def upsert_chunks(chunks: list[object]) -> str:
    """chunk 入 Milvus + Mongo 元数据。MVP:占位。"""
    raise NotImplementedError("upsert_chunks — wire up in #3")


def delete_by_doc(doc_id: str) -> int:
    """按 doc_id 删全部 chunk。MVP:占位。"""
    raise NotImplementedError("delete_by_doc — wire up in #3")
