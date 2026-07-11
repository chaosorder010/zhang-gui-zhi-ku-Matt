#: 冒烟 — 路由注册 + 健康端点响应 + FastAPI 装配。
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


def test_app_instantiates_clean() -> None:
    """FastAPI 实例装配不报错。"""
    from app.api.app import create_app

    app = create_app()
    assert app is not None
