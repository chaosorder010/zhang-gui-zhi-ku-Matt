#: 三段分块:标题切 → 超长段落滑动窗口切 → 过短合并。
#: 每个产出 Chunk 文本头自动拼 "<item_name> | "。
from __future__ import annotations

import re

from app.core.config import settings

_HEADING_RE = re.compile(
    r"""^[ \t]*(?:
       第[一二三四五六七八九十百千零〇\d]+[章节篇条款]  |  # 第一章 / 第三节
       \d+(?:\.\d+){0,3}[、.\)\s]                     |  # 1. / 1.1 / 1、
       [#]{1,6}\s+                                       # markdown #
    )\S*""",
    re.MULTILINE | re.UNICODE | re.VERBOSE,
)


def _estimate_tokens(text: str) -> int:
    """粗 token 估计:EN ~4char/tok,CN ~1.5char/tok,取均值。"""
    cn = sum(1 for ch in text if "一" <= ch <= "鿿")
    en = len(text) - cn
    return max(1, int(cn * 0.9 + en * 0.33))


def _split_paragraphs(md: str) -> list[str]:
    """按一个以上换行切段落,去空过短。"""
    out: list[str] = []
    for blk in re.split(r"\n\s*\n", md):
        b = blk.strip()
        if b:
            out.append(b)
    return out or [md.strip()]


def _slice_long(para: str, item_name: str, doc_id: str, section: str, seq: int) -> list[dict]:
    """超长段滑动窗口切片。返回部分 chunk 字典(未拼 seq)。"""
    maxc = max(1, settings.MAX_TOKENS)
    over = min(settings.OVERLAP_TOKENS, maxc // 2)
    body_prefix = f"{item_name} | "
    unit = 3  # 粗 char 分片;按估算 token 收敛较稳
    slices: list[str] = []
    i = 0
    while i < len(para):
        chunk = para[i : i + maxc * unit]
        slices.append(chunk)
        if i + maxc * unit >= len(para):
            break
        i += (maxc - over) * unit
    out: list[dict] = []
    for s in slices:
        txt = (body_prefix + s).strip()
        out.append({"text": txt, "doc_id": doc_id, "item_name": item_name,
                    "section": section, "token_len": _estimate_tokens(txt)})
    return out


def _section_split(md: str) -> list[tuple[str, str]]:
    """标题正则切章节。每章节 (heading, body)。"""
    matches = list(_HEADING_RE.finditer(md))
    if not matches:
        return [("", md.strip())]
    sections: list[tuple[str, str]] = []
    for idx, m in enumerate(matches):
        heading = m.group(0).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(md)
        body = md[start:end].strip()
        sections.append((heading, body))
    # heading 之前的前置段
    first_start = matches[0].start()
    if first_start > 0:
        prefix = md[:first_start].strip()
        if prefix:
            sections.insert(0, ("", prefix))
    return sections


def _merge_short(chunks: list[dict]) -> list[dict]:
    """相邻过短 chunk 合并入邻居(优先向前合)。"""
    merged: list[dict] = []
    minc = max(1, settings.MIN_TOKENS)
    for c in chunks:
        if c["token_len"] < minc and merged:
            prev = merged[-1]
            cand = {
                "text": prev["text"] + "\n" + c["text"],
                "doc_id": prev["doc_id"],
                "item_name": prev["item_name"],
                "section": prev["section"] or c["section"],
                "token_len": prev["token_len"] + c["token_len"],
            }
            merged[-1] = cand
            continue
        merged.append(c)
    # 兜底:末 chunk 过短→合入前一个
    if len(merged) >= 2 and merged[-1]["token_len"] < minc:
        last = merged.pop()
        prev = merged[-1]
        merged[-1] = {
            "text": prev["text"] + "\n" + last["text"],
            "doc_id": prev["doc_id"],
            "item_name": prev["item_name"],
            "section": prev["section"] or last["section"],
            "token_len": prev["token_len"] + last["token_len"],
        }
    return merged


def _to_chunks(raw: list[dict], doc_id: str, base_seq: int) -> list[dict]:
    """拼 seq + chunk_id。"""
    out: list[dict] = []
    for i, c in enumerate(raw):
        out.append({
            "text": c["text"],
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_{base_seq + i}",
            "item_name": c["item_name"],
            "section": c["section"],
            "seq": base_seq + i,
            "token_len": c["token_len"],
        })
    return out


def chunk_document(
    pages_md: list[str], doc_id: str, item_name: str
) -> list[dict]:
    """三段分块入口,返回字典列表(含 text/chunk_id/doc_id/item_name/section/seq/token_len)。"""
    body_prefix = f"{item_name} | "
    buf: list[dict] = []
    seq = 0
    for pi, md in enumerate(pages_md):
        md = md.strip()
        if not md:
            continue
        if _estimate_tokens(md) <= settings.MAX_TOKENS:
            txt = (body_prefix + md).strip()
            buf.append({"text": txt, "doc_id": doc_id, "item_name": item_name,
                        "section": "", "token_len": _estimate_tokens(txt)})
            continue
        for heading, body in _section_split(md):
            if not body:
                continue
            if _estimate_tokens(body) <= settings.MAX_TOKENS:
                txt = (body_prefix + body).strip()
                buf.append({"text": txt, "doc_id": doc_id, "item_name": item_name,
                            "section": heading, "token_len": _estimate_tokens(txt)})
            else:
                for part in _slice_long(body, item_name, doc_id, heading, seq):
                    buf.append(part)
    merged = _merge_short(buf)
    return _to_chunks(merged, doc_id, 0)


# 保持向后兼容
def chunk_page(md: str, doc_id: str, item_name: str, seq: int = 0) -> list[dict]:
    """兼容旧占位入口。"""
    return chunk_document([md], doc_id, item_name) or []
