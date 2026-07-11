#: LLM 抽取主体名(item_name) — 文档级一次调用。
from __future__ import annotations


ITEM_NAME_PROMPT = """从以下文档首段推断主体名称(item name,如"格力 KFR-35GW 空调")。
只输出名称,无其他字。若推断不出,输出"未识别"。

--- 文档首段 ---
{head}
"""


def extract_item_name(head: str) -> str:
    """抽取主体名。MVP:占位 — 截前 30 字,等 #3 接 LLM。"""
    return head.strip()[:30] or "未识别"
