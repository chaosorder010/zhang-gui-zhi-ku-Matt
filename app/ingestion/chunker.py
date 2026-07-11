#: 三段分块:标题切 → 超长再切 → 过短合并。
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """检索单元。"""

    text: str
    doc_id: str
    item_name: str
    section: str
    seq: int
    token_len: int


def chunk_page(md: str, doc_id: str, item_name: str, seq: int = 0) -> list[Chunk]:
    """分块入口。MVP:占位 — 整页单 chunk,等 #3 接三段。"""
    text = md.strip()
    if not text:
        return []
    return [
        Chunk(
            text=text,
            doc_id=doc_id,
            item_name=item_name,
            section="",
            seq=seq,
            token_len=len(text),
        )
    ]
