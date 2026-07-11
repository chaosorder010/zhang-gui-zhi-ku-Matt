#: HyDE — 生成假设文档辅助检索。LLM_STUB 时返回提问本身。
from __future__ import annotations

import logging

from app.core.config import settings
from app.services.llm import get_chat_llm

log = logging.getLogger(__name__)

HYDE_PROMPT = """用户提问: {question}
请写出一段 3-8 句的假设性中文回答,供给检索系统使用。只输出回答,无其它。"""


def _stub(question: str) -> str:
    return f"关于“{question}”的假设说明。"


def generate_hypothetical_doc(question: str) -> str:
    """LLM 生成假设答,写安全时退化为提问本身。"""
    if not question.strip():
        return ""
    if settings.LLM_STUB:
        return _stub(question)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        msg = get_chat_llm().invoke([
            SystemMessage(content="你是检索增强的助手,会写假设性中文回答。"),
            HumanMessage(content=HYDE_PROMPT.format(question=question)),
        ])
        out = (msg.content if isinstance(msg.content, str) else str(msg.content)).strip()
        return out or _stub(question)
    except Exception as exc:  # noqa: BLE001
        log.warning("HyDE LLM 失败,回退提问: %s", exc)
        return _stub(question)
