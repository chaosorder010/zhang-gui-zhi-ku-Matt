#: FastAPI 装配 — 注册路由 + lifespan。
from fastapi import FastAPI

from app.api.routes import health, documents, query
from app.core.config import settings
from app.core.events import lifespan


def create_app() -> FastAPI:
    """工厂:装配 FastAPI 实例。"""
    app = FastAPI(title=settings.APP_NAME, version=settings.__module__, lifespan=lifespan)
    app.include_router(health.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(query.router, prefix="/api")
    return app
