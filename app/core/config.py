#: pydantic-settings 启动期配置。缺必填键直接 raise,明报缺啥。
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
    EMBED_MODEL: str = "BAAI/bge-m3"
    EMBED_DENSE_DIM: int = 1024
    EMBED_DEVICE: str = "cpu"
    #: 本机无 GPU/无 sentence_transformers 时启用确定性 fallback(测试/冒烟)
    EMBED_FALLBACK: bool = False

    #: Rerank(本机 GPU)
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_DEVICE: str = "cpu"
    RERANK_FALLBACK: bool = False

    #: LLM(云端)
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_API_KEY: str = Field(default="", description="云端 LLM API key")
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    #: LLM / item_name 抽 / HyDE:测试期 stub 真调用
    LLM_STUB: bool = False

    #: Milvus
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_COLLECTION: str = "kb_chunks"

    #: MongoDB
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "zhang_gui_zhi_ku"

    #: MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
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
    OVERLAP_TOKENS: int = Field(default=80, ge=0, description="滑动窗口重叠")

    #: 护栏
    #: fallback Jaccard 相似度通常 0.05-0.25,真 cosine 0.3+ 才算相关;
    #: 默认 0.3 让 fallback 路径下"几乎不相关"的 query 走 reject。
    REJECT_THRESHOLD: float = Field(default=0.3, ge=0.0, description="rerank 分低于此 → reject")


settings = Settings()
