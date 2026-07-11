#: RAG 管线装配 — LangGraph 状态机入口。MVP:占位。
from __future__ import annotations


def build_graph() -> object:
    """装配 LangGraph 状态机。MVP:占位 — 等 #3 接 LangGraph。"""
    raise NotImplementedError("build_graph — wire up LangGraph in #3")


async def run(graph: object, question: str, history: list[dict]) -> dict:
    """单轮问答。MVP:占位。"""
    raise NotImplementedError("run — wire up in #3")
