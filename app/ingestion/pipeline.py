#: 入库管线:解析 → item_name 抽 → 分块 → embed → Milvus+Mongo 落。
from __future__ import annotations

import logging
import os

from app.ingestion import embed as embed_mod
from app.ingestion import chunker, parsers
from app.ingestion.index import ensure_collection, upsert_chunks, _new_doc_id
from app.ingestion.item_name import extract_item_name

log = logging.getLogger(__name__)


def _is_pdf(filename: str, ctype: str | None) -> bool:
    low = filename.lower()
    return low.endswith(".pdf") or (ctype or "").find("pdf") >= 0


def _is_markdown(filename: str, ctype: str | None) -> bool:
    low = filename.lower()
    return low.endswith((".md", ".markdown")) or (ctype or "").find("markdown") >= 0


def parse_document(data: bytes, filename: str, content_type: str | None) -> parsers.ParsedDoc:
    """按后缀/ctype 选解析器。坏文件 → ValueError(路由转 4xx)。"""
    if _is_pdf(filename, content_type):
        return parsers.parse_pdf(data, filename)
    if _is_markdown(filename, content_type) or (content_type or "").startswith("text/"):
        text = data.decode("utf-8", errors="replace")
        return parsers.parse_markdown(text, filename)
    raise ValueError(
        f"[{filename}] 不受支持格式(ctype={content_type})。MVP 仅 PDF/MD")


def build_chunks(parsed: parsers.ParsedDoc, doc_id: str, item_name: str) -> list[dict]:
    """三段分块。分块前 item_name 由 chunker 拼到每 text 头。"""
    return chunker.chunk_document(parsed.pages_md, doc_id, item_name)


def embed_chunks(chunks: list[dict]) -> tuple[list[list[float]], list[dict[int, float]]]:
    """embedder 先 fit_sparse 建 IDF,再批量编码。"""
    embedder = embed_mod.get_embedder()
    embedder.fit_sparse([c["text"] for c in chunks])
    return embedder.encode([c["text"] for c in chunks])


async def run_ingestion(data: bytes, filename: str, content_type: str | None) -> dict:
    """入口:单文档入库。返回 {doc_id, item_name, chunk_count}。"""
    ensure_collection()
    parsed = parse_document(data, filename, content_type)
    head_md = parsed.pages_md[0] if parsed.pages_md else filename
    item_name = extract_item_name(head_md[:800])
    doc_id = _new_doc_id(filename)
    chunks = build_chunks(parsed, doc_id, item_name)
    if not chunks:
        log.warning("[%s] 零 chunk", filename)
    dense_vecs, sparse_vecs = embed_chunks(chunks)
    result = await upsert_chunks(chunks, dense_vecs, sparse_vecs)
    log.info("ingested %s → %s chunks", filename, result["chunk_count"])
    return result


def reset_backend_for_tests() -> None:
    """测试期清理:清掉 embedder 缓存与 InMemory。"""
    embed_mod.get_embedder.cache_clear()
    from app.ingestion.index import rebuild_vector_store, InMemoryVectorStore
    rebuild_vector_store(InMemoryVectorStore())


if os.getenv("MILVUS_FALLBACK") != "1" and os.getenv("CI") != "":
    os.environ["MILVUS_FALLBACK"] = "1"
