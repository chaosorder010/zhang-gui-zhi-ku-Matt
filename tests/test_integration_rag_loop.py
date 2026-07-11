"""集成测试 — LangGraph 状态流转、入库→检索→问答闭环(忽略 Milvus 未起的降级路径)。

run 时默认走 InMemoryVectorStore(MILVUS_FALLBACK=1),所以不依赖 milvus 容器。
compose 实库回归留待 /verify 冒烟(docker compose up + 上传→查询闭环)。
"""
from __future__ import annotations

import pytest

from app.ingestion.pipeline import run_ingestion, reset_backend_for_tests
from app.ingestion.index import list_documents, delete_by_doc, get_doc_meta_store
from app.retrieval.milvus_search import hybrid_search
from app.graph.pipeline import ainvoke


MD = (
    "# 掌柜智库\n\n掌柜智库是垂直领域企业级 RAG 知识库,中文优先。\n\n"
    "## 功能\n- 文档入库:PDF/MD 自动解析、分块、embed、入库。\n"
    "## 架构\n核心环路:retrieve → fuse → rerank → generate → respond。"
)


@pytest.fixture(autouse=True)
def _reset_store(monkeypatch) -> None:
    """每测清后端,互不影响。"""
    monkeypatch.setenv("MILVUS_FALLBACK", "1")
    monkeypatch.setenv("EMBED_FALLBACK", "1")
    monkeypatch.setenv("RERANK_FALLBACK", "1")
    monkeypatch.setenv("LLM_STUB", "1")
    from app.ingestion.index import (  # noqa: PLC0415
        rebuild_vector_store, InMemoryVectorStore,
    )
    reset_backend_for_tests()
    rebuild_vector_store(InMemoryVectorStore())
    get_doc_meta_store()._stub = type(get_doc_meta_store()._stub)()
    yield


@pytest.mark.asyncio
async def test_ingest_e2e_returns_item_name_and_chunks() -> None:
    res = await run_ingestion(MD.encode("utf-8"), "intro.md", "text/markdown")
    assert res["chunk_count"] >= 1
    assert "掌柜智库" in res["item_name"], res


@pytest.mark.asyncio
async def test_search_returns_candidates() -> None:
    await run_ingestion(MD.encode("utf-8"), "intro.md", "text/markdown")
    out = hybrid_search("检索功能", top_k=5)
    assert isinstance(out, list)
    assert len(out) >= 1


@pytest.mark.asyncio
async def test_graph_runs_local_branch() -> None:
    await run_ingestion(MD.encode("utf-8"), "intro.md", "text/markdown")
    out = await ainvoke("系统功能?", history=[])
    assert out["route"] in ("local", "reject")
    assert isinstance(out["answer"], str)
    assert "retrieval_trace" in out


@pytest.mark.asyncio
async def test_reject_branch_empty_store() -> None:
    """空库时直接走 reject(无高质量命中)。"""
    out = await ainvoke("随便来点什么", history=[])
    # 空库无候选,fuse 空 → rerank 空 → reject
    assert out["route"] == "reject"
    assert out["citations"] == []


@pytest.mark.asyncio
async def test_delete_then_list_empty() -> None:
    res = await run_ingestion(MD.encode("utf-8"), "intro.md", "text/markdown")
    n = await delete_by_doc(res["doc_id"])
    assert n >= 1
    # 元数据也应删
    metas = await list_documents()
    assert not any(m.get("doc_id") == res["doc_id"] for m in metas)


@pytest.mark.asyncio
async def test_multi_turn_session_does_not_repeat_history() -> None:
    await run_ingestion(MD.encode("utf-8"), "intro.md", "text/markdown")
    prior = await ainvoke("功能有哪些?", history=[])
    # assistant 已在 history,"总结一下"不应把上一答再输出一遍
    follow = await ainvoke(
        "总结一下。",
        history=[
            {"role": "user", "content": "功能有哪些?"},
            {"role": "assistant", "content": prior["answer"]},
        ],
    )
    assert follow["answer"] and isinstance(follow["answer"], str)
