# tests/test_chunker.py
from app.rag.chunker import build_text, chunk_markdown

DOC = """# 售后政策

## 退货政策

签收后 7 天内可无理由退货。质量问题 30 天内可退换。定制商品不支持。

## 运费说明

下单运费按地区收取。退货寄回的运费:质量问题由商家承担,非质量问题由买家承担。上门取件时无需垫付。
"""

TABLE_DOC = """# 商品 FAQ

## 热门商品

| 商品 | 价格 | 保修 |
|---|---|---|
| 猫粮 5kg | 129 | 7天无理由 |
| 自动喂食器 | 299 | 一年质保 |
| 猫抓板 | 49 | 15天包换 |
"""


def test_build_text_skips_empty_parts():
    assert build_text("售后/退货", "怎么退", "7天无理由") == "售后/退货\n怎么退\n7天无理由"
    assert build_text("", "", "只有答案") == "只有答案"


def test_chunk_by_headings_sets_section_path_and_category():
    chunks = chunk_markdown("policy.md", DOC)
    titles = [c.section_path for c in chunks]
    assert "售后政策 > 退货政策" in titles and "售后政策 > 运费说明" in titles
    assert all(c.content_type == "政策" for c in chunks)
    # 政策类:questions 用章节标题,category 用上级路径
    refund = next(c for c in chunks if c.section_path.endswith("退货政策"))
    assert refund.questions == "退货政策"
    assert refund.category == "售后政策"
    assert "7 天" in refund.answer


def test_chunk_text_is_assembled_for_embedding():
    chunks = chunk_markdown("policy.md", DOC)
    c = chunks[0]
    assert c.text == build_text(c.category, c.questions, c.answer)


def test_table_rows_split_with_header_copied():
    chunks = chunk_markdown("faq.md", TABLE_DOC, content_type="FAQ", max_chars=60)
    table_chunks = [c for c in chunks if "|" in c.answer]
    assert len(table_chunks) >= 2  # 大表格被按行切开
    for c in table_chunks:
        # 每块都带表头(表头行 + 分隔行)
        assert "| 商品 | 价格 | 保修 |" in c.answer
        assert "|---|---|---|" in c.answer.replace(" ", "")


def test_overlap_trimmed_to_sentence_boundary():
    long_text = "# 手册\n\n## 长章节\n\n" + "这是一句话。 " * 200
    chunks = chunk_markdown("manual.md", long_text, max_chars=200, overlap=60)
    assert len(chunks) >= 2
    for c in chunks[:-1]:
        assert c.answer.rstrip().endswith(("。", "!", "?", "！", "？", "。"[-1]))


def test_no_chunk_exceeds_max_chars_badly():
    long_text = "# 手册\n\n## 长章节\n\n" + "这是一句话。 " * 200
    chunks = chunk_markdown("manual.md", long_text, max_chars=200, overlap=60)
    assert all(len(c.answer) <= 240 for c in chunks)  # 允许少量超出行长


# --- 重叠(overlap)必须真的发生 ---

SENT_END = "。!?！？;；"


def _shared_prefix_in_suffix(a: str, b: str) -> int:
    """返回使 a 以 b[:k] 结尾的最大 k,即相邻块 a→b 实际共享的文本长度。"""
    for k in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:k]):
            return k
    return 0


def _numbered_doc(n: int = 200) -> str:
    return "# 手册\n\n## 长章节\n\n" + " ".join(f"第{i}句。" for i in range(1, n + 1))


def test_adjacent_chunks_share_overlap_on_sentence_boundary():
    chunks = chunk_markdown("manual.md", _numbered_doc(), max_chars=200, overlap=60)
    assert len(chunks) >= 2
    for a, b in zip(chunks, chunks[1:]):
        shared = _shared_prefix_in_suffix(a.answer, b.answer)
        assert 0 < shared <= 60, (a.answer[-30:], b.answer[:30])
        # 共享的那段文字原样出现在两块里(a 的结尾 == b 的开头)
        assert a.answer.endswith(b.answer[:shared])
        assert b.answer.startswith(a.answer[-shared:])
        # 重叠段是"整句"后缀,不以半截话开头/结尾
        assert b.answer[:shared].endswith(tuple(SENT_END))
        before = a.answer[: len(a.answer) - shared].rstrip()
        assert not before or before[-1] in SENT_END


def test_zero_overlap_shares_nothing():
    chunks = chunk_markdown("manual.md", _numbered_doc(), max_chars=200, overlap=0)
    assert len(chunks) >= 2
    for a, b in zip(chunks, chunks[1:]):
        assert _shared_prefix_in_suffix(a.answer, b.answer) == 0


def test_overlap_not_smaller_than_max_chars_does_not_explode():
    # 旧实现里 stride = max(1, max_chars - overlap) → overlap >= max_chars 时步长为 1,
    # 1000 字符会被切成近千块。无终止符的正文强制走定宽兜底路径。
    chunks = chunk_markdown("x.md", "# T\n\n## S\n\n" + "x" * 1000, max_chars=50, overlap=100)
    assert 2 <= len(chunks) < 100


# --- 表格回退必须是"按行"的,且每块都带表头 ---


def test_wide_table_row_keeps_header_on_every_chunk():
    wide = "# T\n\n## S\n\n| A | B |\n|---|---|\n| " + "x" * 100 + " | y |\n"
    chunks = chunk_markdown("t.md", wide, content_type="FAQ", max_chars=40)
    assert len(chunks) >= 2
    for c in chunks:
        # 一行超宽时按该行子切分,每个子块仍带完整的表头行 + 分隔行
        assert "| A | B |" in c.answer
        assert "|---|---|" in c.answer.replace(" ", "")


def test_leading_prose_not_glued_to_table_header():
    doc = (
        "# 商品 FAQ\n\n## 热门商品\n\n"
        "以下是本月热门商品的价格与保修说明,请以页面实时信息为准。\n\n"
        "| 商品 | 价格 | 保修 |\n"
        "|---|---|---|\n"
        "| 猫粮 5kg | 129 | 7天无理由 |\n"
        "| 自动喂食器 | 299 | 一年质保 |\n"
    )
    chunks = chunk_markdown("faq.md", doc, content_type="FAQ", max_chars=60)
    table_chunks = [c for c in chunks if "|" in c.answer]
    prose_chunks = [c for c in chunks if "|" not in c.answer]
    assert len(table_chunks) >= 2
    for c in table_chunks:
        # 不允许出现无表头的表格块,且正文不得混进表格块
        assert "| 商品 | 价格 | 保修 |" in c.answer
        assert "|---|---|---|" in c.answer.replace(" ", "")
        assert "以下是本月热门商品" not in c.answer
    # 正文行自成一款,不与表头混拼
    assert len(prose_chunks) == 1
    assert prose_chunks[0].answer.startswith("以下是本月热门商品")
