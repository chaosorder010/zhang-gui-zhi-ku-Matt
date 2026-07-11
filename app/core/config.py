#: pydantic-settings 启动期配置。缺必填键直接raise,明报缺啥。
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全启动期配置,收 `.env`,启动期校验。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    #: 服务
    APP_NAME: str = "zhang-gui-zhi-ku"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    #: Embedding(本机 GPU)
    EMBED_MODEL: str = "bge-m3"
    EMBED_DENSE_DIM: int = 1024

    #: Rerank(本机 GPU)
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"

    #: LLM(云端)
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_API_KEY: str = Field(default=..., description="云端 LLM API key,必填")
    LLM_BASE_URL: str = "https://api.openai.com/v1"

    #: Milvus
    MILVUS_URI: str = "http://milvus:19530"
    MILVUS_COLLECTION: str = "kb_chunks"

    #: MongoDB
    MONGO_URI: str = "mongodb://mongo:27017"
    MONGO_DB: str = "zhang_gui_zhi_ku"

    #: MinIO
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "kb-images"
    MINIO_SECURE: bool = False

    #: 检索超参
    ALPHA: float = Field(default=0.7, ge=0.0, le=1.0, description="dense/sparse 混合权重")
    RRF_K: int = Field(default=60, gt=0, description="RRF 平滑常数")
    TOP_K_RETRIEVE: int = Field(default=20, gt=0, description="每路召回 top-K")
    TOP_N_RERANK: int = Field(default=5, gt=0, description="rerank 后送 LLM top-N")

    #: 分块超参
    MAX_TOKENS: int = Field(default=512, gt=0)
    MIN_TOKENS: int = Field(default=64, gt=0)


settings = Settings()
