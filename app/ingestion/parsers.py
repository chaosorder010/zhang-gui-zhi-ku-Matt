#: 文档解析器 — PDF(MinerU 占位) / MD 统一产出 ParsedDoc。
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ImageRef:
    """图片占位引用。"""

    page: int
    index: int
    alt_text: str = ""


@dataclass
class ParsedDoc:
    """解析统一产物。"""

    pages_md: list[str]
    images: list[ImageRef]


def parse_pdf(data: bytes, filename: str) -> ParsedDoc:
    """PDF 解析。MVP:占位 — NotImplementedError,等 MinerU 本机 pipeline。"""
    raise NotImplementedError("PDF parsing via MinerU — wire up in #3")


def parse_markdown(text: str, filename: str) -> ParsedDoc:
    """MD 解析:按换页符拆 pages_md。"""
    pages = [p for p in text.split("\f") if p.strip()]
    pages = pages or [text]
    return ParsedDoc(pages_md=pages, images=[])
