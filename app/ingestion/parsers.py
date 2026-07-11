#: 文档解析器 — PDF(PyMuPDF) / Markdown 统一产出 ParsedDoc。
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


def _slug_to_text(filename: str) -> str:
    name = filename.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return stem.replace("_", " ").replace("-", " ") or "未命名文档"


def parse_pdf(data: bytes, filename: str) -> ParsedDoc:
    """PDF 解析。MVP 用 PyMuPDF 逐页取文本(明文 PDF);失败明报。"""
    if not data:
        raise ValueError(f"[{filename}] 空文件")
    try:
        import fitz  # type: ignore[import-untyped]  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(f"[{filename}] PyMuPDF 未装,无法解析 PDF") from exc
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"[{filename}] 非有效 PDF: {exc}") from exc
    try:
        pages_md: list[str] = []
        images: list[ImageRef] = []
        for pi, page in enumerate(doc):
            # html 能 retained 空格布局,比 "text" 稳
            txt = page.get_text("html")
            if not txt:
                txt = page.get_text("text")
            # html 直接 passthrough;去行首连续占位符噪声
            import re as _re
            cleaned = _re.sub(r"[·•　\s]+\n", "\n", txt).strip()
            if cleaned and _re.search(r"[\w一-鿿]", cleaned):
                pages_md.append(cleaned)
            for ii, img in enumerate(page.get_images(full=False)):
                xref = img[0] if img else -1
                images.append(ImageRef(page=pi, index=ii, alt_text=f"xref={xref}"))
        if not pages_md:
            # 扫描版兜底:用文件名 stem 占位一页,避免全库空
            pages_md = [f"# {_slug_to_text(filename)}\n(扫描版 PDF,未提取到文本)"]
        return ParsedDoc(pages_md=pages_md, images=images)
    finally:
        doc.close()


def parse_markdown(text: str, filename: str) -> ParsedDoc:
    """MD 解析:按换页符拆 pages_md;无换页即单页。"""
    if not text or not text.strip():
        raise ValueError(f"[{filename}] 空 Markdown")
    pages = [p.strip() for p in text.split("\f") if p.strip()]
    pages = pages or [text.strip()]
    return ParsedDoc(pages_md=pages, images=[])
