#: LangChain ChatModel 装配 — OpenAI(默认)兼容接口 + Stub 双轨。
from __future__ import annotations

import logging
from functools import lru_cache

from app.core.config import settings

log = logging.getLogger(__name__)


class _StubLLM:
    """无 API key / LLM_STUB 时返回稳字串,跑通管线。"""

    def invoke(self, messages):  # type: ignore[no-uniform-def]
        from langchain_core.messages import AIMessage

        return AIMessage(content="(stub) 请检查后端密钥配置。")

    def stream(self, messages):  # type: ignore[no-uniform-def]
        yield "(stub)"

    async def astream(self, messages):  # type: ignore[no-uniform-def]
        from langchain_core.messages import AIMessage

        # 拆字符流式,行为更接近真 LLM
        text = "(stub) 请检查后端密钥配置。"
        for ch in text:
            yield AIMessage(content=ch, additional_kwargs={})


def _build_llm():
    if not settings.LLM_API_KEY or settings.LLM_STUB:
        log.warning("LLM 用 Stub 模式(LLM_STUB 或 key 空)")
        return _StubLLM()
    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import-untyped]

        return ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=0.2,
        )
    except ImportError:
        log.warning("langchain_openai 未装,回退 Stub")
        return _StubLLM()


@lru_cache(maxsize=1)
def get_chat_llm():
    """取 ChatModel 实例(单例)。"""
    return _build_llm()
