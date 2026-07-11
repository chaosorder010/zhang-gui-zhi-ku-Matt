#: pytest fixtures — 注入启动期必填 env,让 Settings 在测试期可装配。
#: NOTE: os.environ 必须在 collection 前写入(app.core.config 在 import 时即实例化 Settings)。
from __future__ import annotations

import os

_REQUIRED_ENV = {
    "LLM_API_KEY": "test-key",
    "MINIO_ACCESS_KEY": "minioadmin",
    "MINIO_SECRET_KEY": "minioadmin",
}

for _k, _v in _REQUIRED_ENV.items():
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402 — env must be set before app import


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    """自动注入启动期必填 env 变量(双保险)。"""
    for k, v in _REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
