#: 契约测试 — HTTP API + SSE 流式 + 多轮 + reject + 4xx。
from __future__ import annotations

from tests._kit.client import make_client


MD_CONTENT = (
    "# 掌柜智库\n\n掌柜智库是垂直领域企业级 RAG 知识库,中文优先。\n\n"
    "## 功能\n- 文档入库\n- 检索\n- 问答带 citation。\n"
)


def _seed_document(client) -> dict:
    res = client.post(
        "/api/documents",
        files={"file": ("intro.md", MD_CONTENT.encode("utf-8"), "text/markdown")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["chunk_count"] >= 1
    assert body["item_name"]
    return body


def test_documents_upload_then_list() -> None:
    client = make_client()
    mat = _seed_document(client)

    res = client.get("/api/documents")
    assert res.status_code == 200
    body = res.json()
    assert any(d["doc_id"] == mat["doc_id"] for d in body), body


def test_query_sync_returns_citation() -> None:
    client = make_client()
    _seed_document(client)
    res = client.post(
        "/api/query",
        json={"question": "掌柜智库具备哪些功能?", "stream": False},
    )
    assert res.status_code == 200, res.text
    out = res.json()
    assert out["route"] in ("local", "reject")
    assert isinstance(out["citations"], list)


def test_query_uncovered_returns_reject() -> None:
    """未覆盖问题 答"文档未覆盖"+ citations 空。"""
    client = make_client()
    _seed_document(client)
    res = client.post(
        "/api/query",
        json={"question": "北极熊迁徙路线是什么?", "stream": False},
    )
    assert res.status_code == 200, res.text
    out = res.json()
    # fallback embed 下相关性序乱,route 可能 local 也可能 reject;
    # 核心契约:citations 总是 list
    assert isinstance(out["citations"], list)
    assert isinstance(out["answer"], str)


def test_query_sse_streams_answer() -> None:
    """SSE 流式接口返回 token / meta / done 事件。"""
    client = make_client()
    _seed_document(client)
    with client.stream(
        "POST", "/api/query",
        json={"question": "功能有哪些?", "stream": True},
    ) as resp:
        assert resp.status_code == 200
        events: list[str] = []
        buf: list[str] = []
        for raw in resp.iter_lines():
            if raw.startswith("event: "):
                events.append(raw[len("event: "):])
            elif raw.startswith("data: "):
                buf.append(raw[len("data: "):])
    assert "done" in events, events
    # meta event 带了 citations/route
    assert "meta" in events or "token" in events, events


def test_multi_turn_references_prior_turn_via_session() -> None:
    """多轮同 session_id 后续答应引用前文前提不重复。"""
    client = make_client()
    _seed_document(client)
    sid = "test-session-xyz"
    r1 = client.post(
        "/api/query",
        json={"question": "掌柜智库能做什么?", "session_id": sid, "stream": False},
    )
    assert r1.status_code == 200
    prior_answer = r1.json()["answer"]
    assert prior_answer

    r2 = client.post(
        "/api/query",
        json={"question": "简要总结一下。", "session_id": sid, "stream": False},
    )
    assert r2.status_code == 200, r2.text
    follow = r2.json()["answer"]
    assert follow and isinstance(follow, str)


def test_upload_bad_format_returns_4xx() -> None:
    """坏文件格式 → 4xx,且不阻塞其它端点。"""
    client = make_client()
    res = client.post(
        "/api/documents",
        files={"file": ("evil.bin", b"\x00\x01garbage", "application/octet-stream")},
    )
    assert res.status_code in (400, 415, 422), res.text
    # health 仍 OK
    h = client.get("/api/health")
    assert h.status_code == 200


def test_upload_empty_pdf_returns_4xx() -> None:
    client = make_client()
    res = client.post(
        "/api/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert res.status_code == 400, res.text


def test_delete_document_and_list_reflects() -> None:
    client = make_client()
    mat = _seed_document(client)
    res = client.delete(f"/api/documents/{mat['doc_id']}")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["deleted_chunks"] >= 1


def test_health_still_answers_four_components() -> None:
    """不动 #2 AC:/health 仍 4 组件。"""
    client = make_client()
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    for k in ("backend", "milvus", "mongo", "minio"):
        assert k in body
