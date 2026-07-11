#: 单元测试 — RRF 算术、α 混合、chunk 切点、parser 形状、item_name 启发。
from __future__ import annotations

import pytest

from app.storage.vector_store import ChunkRecord
from app.ingestion import chunker, parsers
from app.retrieval.rrf import reciprocal_rank_fuse


def _rec(cid: str, text: str = "x") -> ChunkRecord:
    return ChunkRecord(text=text, doc_id="d", chunk_id=cid, item_name="n")


# --- chunker -------------------------------------------------------------- #
def test_chunker_short_merge() -> None:
    """过短段(< MIN_TOKENS)应合入邻段。"""
    pages_md = ["很小的片段 A。", "很小的片段 B。"]
    # 大幅降低 MIN_TOKENS 让短段被合并
    import unittest.mock as mock
    with mock.patch("app.ingestion.chunker.settings") as s:
        s.MAX_TOKENS = 512
        s.MIN_TOKENS = 1000  # 强制全部过短
        s.OVERLAP_TOKENS = 50
        chunks = chunker.chunk_document(pages_md, "d1", "主体 X")
    assert len(chunks) == 1, "short 应合入 1 个 chunk"
    assert "主体 X" in chunks[0]["text"], "item_name 头必须拼入"


def test_chunker_longen_split() -> None:
    """超长文本应被切成多个 chunk。"""
    import unittest.mock as mock
    big = "数据入库是企业 RAG 必备环节。" * 200
    with mock.patch("app.ingestion.chunker.settings") as s:
        s.MAX_TOKENS = 30
        s.MIN_TOKENS = 5
        s.OVERLAP_TOKENS = 5
        chunks = chunker.chunk_document([big], "d1", "主体 X")
    assert len(chunks) >= 2, "长文本应切多块"
    for c in chunks:
        assert c["chunk_id"].startswith("d1_")
        assert "主体 X" in c["text"]


def test_chunker_item_name_prefix() -> None:
    chunks = chunker.chunk_document(["有效段。"], "d1", "格力 KFR-35GW")
    assert chunks, "有效段应产出 chunk"
    assert chunks[0]["text"].startswith("格力 KFR-35GW | "), \
        f"chunk 头未拼 item_name: {chunks[0]['text'][:20]}"


# --- RRF ------------------------------------------------------------------ #
def test_rrf_k60_matches_hand_calc() -> None:
    """k=60 时 RRF 分数与手算一致。公式 score(d)=Σ 1/(k+rank+1),rank 0-based。"""
    r1 = [(_rec("a"), 0.9), (_rec("b"), 0.8), (_rec("c"), 0.6)]
    r2 = [(_rec("b"), 1.0), (_rec("a"), 0.5)]
    fused = reciprocal_rank_fuse([r1, r2], k=60)
    assert fused, "RRF 结果不应空"
    got = {cid: sc for cid, sc in zip(
        [r.chunk_id for r, _ in fused], [s for _, s in fused])}
    # 手工核:
    #   b 在 r1 rank=1 → 1/(60+1+1)=1/62;在 r2 rank=0 → 1/(60+0+1)=1/61
    #   a 在 r1 rank=0 → 1/61;在 r2 rank=1 → 1/62
    #   c 在 r1 rank=2 → 1/(60+2+1)=1/63
    kb = 1 / 61 + 1 / 62
    ka = 1 / 61 + 1 / 62
    kc = 1 / 63
    assert abs(got["b"] - kb) < 1e-9, got
    assert abs(got["a"] - ka) < 1e-9, got
    assert abs(got["c"] - kc) < 1e-9, got
    # b 与 a 分相等,b 在 r2 rrf 累积顺序先入 meta,排前
    ids = [r.chunk_id for r, _ in fused]
    assert ids[0] in ("a", "b")
    assert ids[-1] == "c"


def test_rrf_dedup_by_chunk_id() -> None:
    r1 = [(_rec("a"), 0.9)]
    r2 = [(_rec("a"), 1.0)]
    fused = reciprocal_rank_fuse([r1, r2], k=60)
    assert len(fused) == 1, "相同 chunk_id 应去重"
    assert fused[0][0].chunk_id == "a"


def test_rrf_empty_route_ok() -> None:
    """路3 占位空仍应工作。"""
    r1 = [(_rec("x"), 0.5)]
    fused = reciprocal_rank_fuse([r1, [], []], k=60)
    assert len(fused) == 1 and fused[0][0].chunk_id == "x"


def test_rrf_alpha_edge() -> None:
    """α=1 全 dense、α=0 全 sparse,分数边界在 0..1。"""
    from app.storage.vector_store import InMemoryVectorStore
    s = InMemoryVectorStore()
    vec = [1.0, 0.0]
    sp = {0: 1.0}
    s.upsert([
        ChunkRecord(text="da", doc_id="d", chunk_id="c1", item_name="",
                    dense=[1.0, 0.0], sparse={0: 1.0}),
        ChunkRecord(text="db", doc_id="d", chunk_id="c2", item_name="",
                    dense=[0.5, 0.0], sparse={0: 0.5}),
    ])
    out1 = s.search(vec, sp, top_k=2, alpha=1.0)
    out0 = s.search(vec, sp, top_k=2, alpha=0.0)
    assert out1[0][0].chunk_id == "c1"  # α=1 全 dense 取最相似
    assert out0[0][0].chunk_id == "c1"  # α=0 全 sparse 仍最高
    assert all(0.0 <= sc <= 1.0 for _, sc in out1 + out0)


# --- parser --------------------------------------------------------------- #
def test_parse_markdown_split_pages() -> None:
    doc = parsers.parse_markdown("A\n\n\f\nB\n", "a.md")
    assert doc.pages_md == ["A", "B"]
    assert doc.images == []


def test_parse_markdown_empty_raises(tmp_path=None) -> None:
    with pytest.raises(ValueError):
        parsers.parse_markdown("   \n  ", "empty.md")


def test_parse_pdf_empty_raises() -> None:
    with pytest.raises(ValueError):
        parsers.parse_pdf(b"", "e.pdf")


def test_parse_pdf_invalid_raises() -> None:
    with pytest.raises(ValueError):
        parsers.parse_pdf(b"not a pdf content at all", "broken.pdf")


def test_parse_pdf_ok_returns_pages(tmp_path) -> None:
    """PyMuPDF 真产 PDF,断言带文本页(默认 font 用 ASCII,"cjk" 需要系统字体)。"""
    import fitz
    pdf = tmp_path / "sample.pdf"
    d = fitz.open()
    page = d.new_page()
    page.insert_text((72, 72), "Hello World.掌柜智库介绍。")
    d.save(str(pdf))
    d.close()
    doc = parsers.parse_pdf(pdf.read_bytes(), pdf.name)
    assert any("Hello" in p for p in doc.pages_md), \
        f"PD 解析未取到文本,pages_md={doc.pages_md!r}"
    assert doc.pages_md, "pages_md 不应空"


# --- item_name 启发 ------------------------------------------------------- #
def test_item_name_stub_fallback_uses_first_heading() -> None:
    from app.ingestion.item_name import extract_item_name
    head = "## 海尔 EG10014HB39GU1 洗烘一体机\n正文……"
    got = extract_item_name(head)
    assert got not in ("", "未识别") and len(got) >= 2


def test_item_name_stub_empty_head() -> None:
    from app.ingestion.item_name import extract_item_name
    assert extract_item_name("    \n  ") == "未识别"


# --- embed fallback 确定性 ---------------------------------------------- #
def test_embed_fallback_same_input_same_output() -> None:
    """fallback dense 同一 input → 同一 output(流水线确定)。"""
    try:
        import sentence_transformers  # noqa: F401
        pytest.skip("sentence_transformers 已装,跳过 fallback 路径测")
    except ImportError:
        pass
    from app.ingestion.embed import get_embedder
    get_embedder.cache_clear()
    e = get_embedder()
    a = e.dense(["hello 世界"])[0]
    b = e.dense(["hello 世界"])[0]
    assert a == b, "fallback dense 应确定性"
    sa = e.sparse(["hello"])[0]
    sb = e.sparse(["hello"])[0]
    assert sa == sb
