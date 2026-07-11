#: 共用 FastAPI TestClient 工厂。
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.app import create_app


def make_client() -> TestClient:
    """装配测试客户端。"""
    return TestClient(create_app())
