#: 结构化日志装配。
import logging
import sys

from app.core.config import settings


def configure_logging() -> None:
    """装配根 logger,LOG_LEVEL 从配置读。"""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    """取命名 logger。"""
    return logging.getLogger(name)
