#: uvicorn 启动入口。
import uvicorn

from app.core.config import settings

if __name__ == "__main__":
    uvicorn.run("app.api.app:create_app", host="0.0.0.0", port=8000,
                factory=True, reload=settings.APP_ENV == "development")
