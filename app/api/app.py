#: FastAPI 装配 — 注册路由 + lifespan + 统一错误 envelope。
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import health, documents, query
from app import __version__
from app.core.config import settings
from app.core.events import lifespan


def _env(detail: str, code: str) -> dict:
    return {"detail": detail, "code": code}


def create_app() -> FastAPI:
    """工厂:装配 FastAPI 实例。"""
    app = FastAPI(title=settings.APP_NAME, version=__version__, lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException) -> JSONResponse:
        status = exc.status_code
        code = {400: "bad_request", 404: "not_found", 413: "too_large",
                422: "validation", 500: "internal"}.get(status, "error")
        return JSONResponse(status_code=status,
                            content=_env(str(exc.detail), code))

    @app.exception_handler(RequestValidationError)
    async def _val(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422,
                            content=_env(str(exc.errors()), "validation"))

    app.include_router(health.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(query.router, prefix="/api")
    return app
