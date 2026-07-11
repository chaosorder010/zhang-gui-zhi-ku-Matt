#: 主体名(item_name)抽取 — 文档级一次调用,LLM 优先;LLM_STUB 兜底取首非空行。
from __future__ import annotations

import logging

from app.core.config import settings

log = logging.getLogger(__name__)

ITEM_NAME_PROMPT = """从以下文档首段推断主体名称(item name,如"格力 KFR-35GW 空调")。
只输出名称,无其他字。若推断不出,输出"未识别"。

--- 文档首段 ---
{head}
"""


def _fallback(head: str) -> str:
    line = ""
    for ln in head.splitlines():
        ln = ln.strip().lstrip("#-_*=").strip()
        if ln and len(ln) >= 2:
            line = ln
            break
    return (line or "未识别")[:40] or "未识别"


def _call_llm(prompt: str) -> str:
    from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore[import-untyped]
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.exceptions import LangChainException

    llm = _get_llm()
    if llm is None:
        return ""
    assert isinstance(llm, BaseChatModel)
    try:
        msg = llm.invoke(
            [SystemMessage(content="你是中文实体抽取助手。"), HumanMessage(content=prompt)]
        )
        return (msg.content if isinstance(msg.content, str) else str(msg.content)).strip()
    except (LangChainException, Exception) as exc:  # noqa: BLE001 — 抽失败回退
        log.warning("item_name LLM 失败: %s", exc)
        return ""


def extract_item_name(head: str) -> str:
    """抽取主体名。LLM_STUB → 兜底启发式;否则走 LangChain。"""
    if not head or not head.strip():
        return "未识别"
    if settings.LLM_STUB:
        return _fallback(head)
    name = _call_llm(ITEM_NAME_PROMPT.format(head=head[:600]))
    return (name or "未识别")[:40] or "未识别"


def _get_llm():  # 局部避免循环
    from app.services.llm import get_chat_llm

    return get_chat_llm()
