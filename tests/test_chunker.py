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
