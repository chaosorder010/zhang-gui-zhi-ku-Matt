#: 冒烟 — 路由注册 + 健康端点响应 + 文档/query 占位 501。
from __future__ import annotations

from tests._kit.client import make_client


def test_health_returns_four_components() -> None:
    """/api/health 共答 backend + 三依赖。"""
    client = make_client()
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    for key in ("backend", "milvus", "mongo", "minio"):
        assert key in body, f"missing {key}"


def test_documents_list_empty() -> None:
    """GET /api/documents 路由注册,返回空列表。"""
    client = make_client()
    res = client.get("/api/documents")
    assert res.status_code == 200
    assert res.json() == []


def test_documents_upload_placeholder_501() -> None:
    """POST /api/documents 占位返回 501。"""
    client = make_client()
    res = client.post("/api/documents", files={"file": ("a.txt", b"x", "text/plain")})
    assert res.status_code == 501


def test_query_placeholder_501() -> None:
    """POST /api/query 占位返回 501。"""
    client = make_client()
    res = client.post("/api/query", json={"question": "hello"})
    assert res.status_code == 501


def test_app_instantiates_clean() -> None:
    """FastAPI 实例装配不报错。"""
    from app.api.app import create_app

    app = create_app()
    assert app is not None
