# ch03 知识库向量检索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `query_faq` 的内部实现从 SQL 关键词查表换成 BGE-M3 + Milvus 的 dense 向量语义检索,工具入参/出参契约不变;离线侧新增"文档结构感知切分 + 历史对话挖知识 + MySQL/Milvus 双写"建库链路。

**Architecture:** 新增 `app/rag/`(chunker / embedder / store / retriever)与 `app/db/repository_knowledge.py`;MySQL 存原文与元数据(权威源,`knowledge_chunks` + `knowledge_staging`),Milvus 集合 `knowledge` 只存 `id + vector(1024, COSINE)`;建库脚本先写 MySQL 记 `pending`、再向量化 upsert 到 Milvus 并回填状态,断点重跑只补未完成块。`query_faq` 改为调 retriever,签名与「未找到」语义不变。

**Tech Stack:** Python 3.12、pymilvus 2.6.x(服务端 Milvus v2.6.10 standalone,Docker)、FlagEmbedding BGE-M3(本地权重)、SQLAlchemy 2.0 async + MySQL、pytest。

**Spec:** `docs/superpowers/specs/2026-09-11-rag-knowledge-base-design.md`

## Global Constraints

- **工具契约不变**:`query_faq(keyword: str) -> str` 签名与返回风格不变;无命中仍返回含「未找到」的提示。
- **MySQL 是原文权威源**:`knowledge_chunks` 存全部文本与元数据;Milvus 集合 `knowledge` **只存 `id`(INT64 主键)+ `vector`(FLOAT_VECTOR 1024,COSINE)**。
- **只跑 dense 单路**:不做关键词召回 / 混合检索 / 重排。
- **双写幂等**:先写 MySQL `status='pending'` → 向量化 → Milvus `upsert`(按主键幂等)→ 回填 `vector_id` 且 `status='embedded'`;**重跑只扫 `pending` 或 `vector_id IS NULL`**。
- **版本对齐**:Milvus 服务端 **v2.6.10**;客户端 `pymilvus>=2.6,<3`。BGE-M3 用**本地权重**(`FlagEmbedding.BGEM3FlagModel`),CPU 上 `use_fp16=False`。
- **四类元数据只进 MySQL、不进向量**:`section_path` / `content_type` / `is_key_clause` / `prev_id`+`next_id`。
- **不改 ch02 既有四张表**(conversations/messages/faq/tickets);只新增两张知识表。
- **测试隔离**:MySQL 用 `mewhelp_test`(ch02 已有 fixtures);Milvus 用独立集合 `knowledge_test`,测试结束 drop。
- **文档先行**:凡 pymilvus / FlagEmbedding / SQLAlchemy 具体 API,动手前先用 Context7 查官方文档核对(已核:create_collection 快建、search 返回 distance、BGEM3FlagModel.encode 返回 `dense_vecs`)。
- **每 Task 结束**向 `dev-notes/ch03.md` 追加一段(四样:用户关键原话 / 我的关键产出 / 用户拒绝或纠偏 / 翻车与返工)。禁止收尾一次性补记。
- **执行模式**:Subagent-Driven(每 Task 独立子代理 + 任务间评审)。
- 命令环境:解释器 `./.venv/Scripts/python.exe`;Milvus 用 `bash scripts/milvus.sh up|status`。

---

### Task 1: 依赖与配置(pymilvus / FlagEmbedding / RAG 参数)

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/config.py`
- Modify: `.env.example`
- Test: `tests/test_rag_config.py`

**Interfaces:**
- Produces(Settings 新增字段): `milvus_uri: str="http://127.0.0.1:19530"`, `milvus_token: str=""`, `knowledge_collection: str="knowledge"`, `milvus_test_collection: str="knowledge_test"`, `embed_model: str="BAAI/bge-m3"`, `embed_device: str="cpu"`, `embed_batch_size: int=12`, `rag_top_k: int=5`, `rag_score_threshold: float=0.5`, `chunk_max_chars: int=800`, `chunk_overlap: int=120`, `mine_batch_size: int=20`, `mine_lookback_days: int=30`, `dedupe_sim_threshold: float=0.95`。

- [ ] **Step 1: 文档核对**

Context7 `/milvus-io/pymilvus`:确认 `MilvusClient(uri=...)`、`create_collection(collection_name, dimension, metric_type="COSINE")` 快建会自带索引;`/flagopen/flagembedding`:确认 `BGEM3FlagModel(model, use_fp16=...)` 与 `.encode(list, batch_size=...)['dense_vecs']`。记录到 dev-notes。

- [ ] **Step 2: 写失败测试**

```python
# tests/test_rag_config.py
from app.config import get_settings


def test_rag_settings_defaults():
    s = get_settings()
    assert s.milvus_uri.startswith("http")
    assert s.knowledge_collection == "knowledge"
    assert s.milvus_test_collection == "knowledge_test"
    assert s.embed_model == "BAAI/bge-m3"
    assert s.rag_top_k == 5 and 0 < s.rag_score_threshold < 1
    assert s.chunk_max_chars == 800 and s.chunk_overlap == 120
    assert 0 < s.dedupe_sim_threshold <= 1
```

- [ ] **Step 3: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_rag_config.py -q`
Expected: FAIL(`AttributeError: 'Settings' object has no attribute 'milvus_uri'`)

- [ ] **Step 4: 实现**

`app/config.py` 在 `mock_seed_salt` 之后追加:

```python
    # RAG / 知识库
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    knowledge_collection: str = "knowledge"
    milvus_test_collection: str = "knowledge_test"
    embed_model: str = "BAAI/bge-m3"
    embed_device: str = "cpu"
    embed_batch_size: int = 12
    rag_top_k: int = 5
    rag_score_threshold: float = 0.5
    chunk_max_chars: int = 800
    chunk_overlap: int = 120
    mine_batch_size: int = 20
    mine_lookback_days: int = 30
    dedupe_sim_threshold: float = 0.95
```

`pyproject.toml` 的 `[project].dependencies` 追加:

```toml
    "pymilvus>=2.6,<3",
    "FlagEmbedding>=1.3",
```

`.env.example` 追加(ch03 段):

```
# --- ch03 RAG ---
MILVUS_URI=http://127.0.0.1:19530
MILVUS_TOKEN=
KNOWLEDGE_COLLECTION=knowledge
MILVUS_TEST_COLLECTION=knowledge_test
EMBED_MODEL=BAAI/bge-m3
EMBED_DEVICE=cpu
EMBED_BATCH_SIZE=12
RAG_TOP_K=5
RAG_SCORE_THRESHOLD=0.5
CHUNK_MAX_CHARS=800
CHUNK_OVERLAP=120
MINE_BATCH_SIZE=20
MINE_LOOKBACK_DAYS=30
DEDUPE_SIM_THRESHOLD=0.95
```

安装(耗时较长,FlagEmbedding 会拉 torch):

```bash
./.venv/Scripts/python.exe -m pip install -q "pymilvus>=2.6,<3" "FlagEmbedding>=1.3"
```

- [ ] **Step 5: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_rag_config.py -q`
Expected: PASS

- [ ] **Step 6: 真实连通性冒烟(一次性)**

```bash
./.venv/Scripts/python.exe -c "
from pymilvus import MilvusClient
from app.config import get_settings
c = MilvusClient(uri=get_settings().milvus_uri)
print('milvus collections:', c.list_collections())
"
```
Expected: 打印 collections 列表(可为空),证明 19530 可达。

- [ ] **Step 7: Commit + dev-notes**

```bash
git add pyproject.toml app/config.py .env.example tests/test_rag_config.py
git commit -m "feat(rag): pymilvus/FlagEmbedding deps and RAG settings"
```
并向 `dev-notes/ch03.md` 追加 Task 1 段。

---

### Task 2: 知识表 DDL + SQLAlchemy 模型

**Files:**
- Modify: `sql/schema.sql`
- Modify: `app/db/models.py`
- Create: `tests/test_knowledge_models.py`

**Interfaces:**
- Produces: `KnowledgeChunk`(表 `knowledge_chunks`)、`KnowledgeStaging`(表 `knowledge_staging`)两个 ORM 类;字段见下。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_knowledge_models.py
from sqlalchemy import inspect

from app.db.models import KnowledgeChunk, KnowledgeStaging


def test_knowledge_chunk_columns():
    cols = {c.name for c in inspect(KnowledgeChunk).columns}
    assert {
        "id", "doc_id", "category", "questions", "answer", "text",
        "section_path", "content_type", "is_key_clause", "prev_id", "next_id",
        "content_hash", "vector_id", "status", "created_at", "updated_at",
    } <= cols
    assert KnowledgeChunk.__table__.c.status.type.enums == ["pending", "embedded"]


def test_knowledge_staging_columns():
    cols = {c.name for c in inspect(KnowledgeStaging).columns}
    assert {"id", "conversation_id", "source_message_ids", "questions", "answer",
            "category", "dedupe_hash", "status", "created_at", "updated_at"} <= cols
    assert KnowledgeStaging.__table__.c.status.type.enums == [
        "staged", "promoted", "dropped"
    ]
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_knowledge_models.py -q`
Expected: FAIL(`ImportError: cannot import name 'KnowledgeChunk'`)

- [ ] **Step 3: 实现**

`sql/schema.sql` 末尾追加(与既有四表同风格;幂等):

```sql
-- ch03:知识库
CREATE TABLE IF NOT EXISTS knowledge_chunks (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '知识块主键,同时是 Milvus 主键',
  doc_id        VARCHAR(255)    NOT NULL                COMMENT '来源文档/批次标识',
  category      VARCHAR(255)    NOT NULL DEFAULT ''     COMMENT '分类(章节上级路径/挖矿)',
  questions     TEXT            NOT NULL                COMMENT '问法(真实问法或章节标题)',
  answer        TEXT            NOT NULL                COMMENT '答案正文',
  text          TEXT            NOT NULL                COMMENT '拼装文本,向量化输入',
  section_path  VARCHAR(512)    NOT NULL DEFAULT ''     COMMENT '章节路径(只存不进向量)',
  content_type  VARCHAR(32)     NOT NULL DEFAULT '政策' COMMENT '内容类型元数据',
  is_key_clause TINYINT(1)      NOT NULL DEFAULT 0      COMMENT '是否关键条款元数据',
  prev_id       BIGINT UNSIGNED NULL                    COMMENT '前块指针',
  next_id       BIGINT UNSIGNED NULL                    COMMENT '后块指针',
  content_hash  CHAR(64)        NOT NULL                COMMENT '归一化内容哈希(幂等/去重)',
  vector_id     VARCHAR(64)     NULL                    COMMENT 'Milvus 主键回填',
  status        ENUM('pending','embedded') NOT NULL DEFAULT 'pending',
  created_at    DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_content_hash (content_hash),
  KEY idx_status (status),
  KEY idx_doc_id (doc_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识块(原文权威源)';

CREATE TABLE IF NOT EXISTS knowledge_staging (
  id                 BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_id    BIGINT UNSIGNED NOT NULL          COMMENT '来源会话',
  source_message_ids VARCHAR(255)    NOT NULL DEFAULT '' COMMENT '来源消息 id 列表',
  questions          TEXT            NOT NULL          COMMENT '抽出的问法(JSON 数组)',
  answer             TEXT            NOT NULL,
  category           VARCHAR(255)    NOT NULL DEFAULT '',
  dedupe_hash        CHAR(64)        NOT NULL,
  status             ENUM('staged','promoted','dropped') NOT NULL DEFAULT 'staged',
  created_at         DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at         DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_dedupe_hash (dedupe_hash),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='挖矿暂存';
```

`app/db/models.py` 追加:

```python
class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(255), nullable=False, server_default="")
    questions: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    section_path: Mapped[str] = mapped_column(String(512), nullable=False, server_default="")
    content_type: Mapped[str] = mapped_column(String(32), nullable=False, server_default="政策")
    is_key_clause: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="0")
    prev_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    next_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("pending", "embedded", name="chunk_status"), nullable=False, server_default="pending"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class KnowledgeStaging(Base):
    __tablename__ = "knowledge_staging"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    source_message_ids: Mapped[str] = mapped_column(String(255), nullable=False, server_default="")
    questions: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(255), nullable=False, server_default="")
    dedupe_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(
        Enum("staged", "promoted", "dropped", name="staging_status"), nullable=False, server_default="staged"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
```

(`models.py` 顶部 import 需补 `Boolean`。)

- [ ] **Step 4: 应用到开发库与测试库**

```bash
./.venv/Scripts/python.exe -m scripts.init_db
```
Expected: 打印 6 张表(4 旧 + 2 新)。测试库的表由 `tests/db_utils.apply_schema` 自动建(它读同一份 `sql/schema.sql`)。

- [ ] **Step 5: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_knowledge_models.py tests/test_db_models.py -q`
Expected: PASS

- [ ] **Step 6: Commit + dev-notes**

```bash
git add sql/schema.sql app/db/models.py tests/test_knowledge_models.py
git commit -m "feat(rag): knowledge_chunks / knowledge_staging DDL and models"
```

---

### Task 3: chunker(结构感知切分)

**Files:**
- Create: `app/rag/__init__.py`
- Create: `app/rag/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Produces:
  - `@dataclass Chunk { doc_id: str; category: str; questions: str; answer: str; text: str; section_path: str; content_type: str; is_key_clause: bool; order_index: int }`
  - `build_text(category: str, questions: str, answer: str) -> str`(拼装:`category\nquestions\nanswer`,空段跳过)
  - `chunk_markdown(doc_id: str, markdown: str, *, content_type: str = "政策", max_chars: int = 800, overlap: int = 120) -> list[Chunk]`

**规则**(spec §5):按 Markdown 标题层级切;超长递归切;相邻块重叠 `overlap` 字符且**回退到最近句号**;表格按行切且**每块复制表头**;政策类 `questions=章节标题`、`category=上级标题路径`。

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_chunker.py -q`
Expected: FAIL(`ModuleNotFoundError: app.rag`)

- [ ] **Step 3: 实现**

```python
# app/rag/__init__.py
"""RAG:切分 / 向量化 / 向量库 / 检索。"""
```

```python
# app/rag/chunker.py
"""结构感知切分:Markdown 标题层级 → 表格按行(复制表头) → 超长递归 → 重叠裁到句号。"""
import re
from dataclasses import dataclass

_SENT_END = "。!?！？;；"
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class Chunk:
    doc_id: str
    category: str
    questions: str
    answer: str
    text: str
    section_path: str
    content_type: str
    is_key_clause: bool
    order_index: int


def build_text(category: str, questions: str, answer: str) -> str:
    return "\n".join(p for p in (category, questions, answer) if p)


def _is_table_sep(line: str) -> bool:
    return bool(re.fullmatch(r"\s*\|[\s:\-|]+\|\s*", line))


def _split_sections(markdown: str) -> list[tuple[str, str, str]]:
    """→ [(上级路径, 章节标题, 正文)];正文不含标题行。"""
    sections: list[tuple[str, str, str]] = []
    stack: list[str] = []
    cur_title, cur_lines = "", []
    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            if cur_title or cur_lines:
                sections.append((stack[-2] if len(stack) >= 2 else "", cur_title, "\n".join(cur_lines).strip()))
            level, title = len(m.group(1)), m.group(2).strip()
            stack = stack[: level - 1]
            stack.append(title)
            cur_title, cur_lines = title, []
        else:
            cur_lines.append(line)
    if cur_title or cur_lines:
        sections.append((stack[-2] if len(stack) >= 2 else "", cur_title, "\n".join(cur_lines).strip()))
    return sections


def _split_table(body: str) -> list[str]:
    """表格:表头(含分隔行)复制到每个按行切出的块。"""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    header, rows = [], []
    for ln in lines:
        if _TABLE_ROW.match(ln):
            if len(header) < 2:
                header.append(ln)
            elif _is_table_sep(ln):
                header.append(ln)
            else:
                rows.append(ln)
        else:
            rows.append(ln)
    if not rows:
        return [body]
    prefix = "\n".join(header[:2])
    out = []
    for r in rows:
        out.append(f"{prefix}\n{r}" if prefix else r)
    return out


def _trim_to_sentence(text: str, overlap: int) -> str:
    """取末尾 overlap 字符做重叠,并回退到最近句号,避免半截话。"""
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    idx = max((tail.rfind(ch) for ch in _SENT_END), default=-1)
    if idx == -1:
        return tail
    return tail[idx + 1 :]


def _recursive_split(body: str, max_chars: int, overlap: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    # 1) 优先在表格行边界切;2) 其次句子边界;3) 兜底按长度
    pieces: list[str] = []
    if any(_TABLE_ROW.match(ln) for ln in body.splitlines()):
        return _split_table(body) if max(len(x) for x in _split_table(body)) <= max_chars else _force_split(body, max_chars, overlap)
    buf = ""
    for sent in re.split(r"(?<=[。!?！？;；])", body):
        if not sent:
            continue
        if len(buf) + len(sent) > max_chars and buf:
            pieces.append(buf)
            buf = _trim_to_sentence(buf, overlap) + sent
        else:
            buf += sent
    if buf:
        pieces.append(buf)
    out: list[str] = []
    for p in pieces:
        out.extend(_force_split(p, max_chars, overlap) if len(p) > max_chars * 1.5 else [p])
    return out


def _force_split(text: str, max_chars: int, overlap: int) -> list[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + max_chars])
        i += max(1, max_chars - overlap)
    return out


def chunk_markdown(
    doc_id: str,
    markdown: str,
    *,
    content_type: str = "政策",
    max_chars: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for parent, title, body in _split_sections(markdown):
        if not body.strip():
            continue
        section_path = f"{parent} > {title}" if parent else title
        category = parent or title
        questions = title
        for piece in _recursive_split(body.strip(), max_chars, overlap):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    category=category,
                    questions=questions,
                    answer=piece,
                    text=build_text(category, questions, piece),
                    section_path=section_path,
                    content_type=content_type,
                    is_key_clause=("关键" in piece or "必须" in piece),
                    order_index=idx,
                )
            )
            idx += 1
    # 前后块指针
    return chunks
```

> 说明:`prev_id/next_id` 是**数据库主键**,插入后才存在 → 由 Task 6 的 `link_neighbors()` 回填(本模块只产出 `order_index`)。

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_chunker.py -q`
Expected: PASS(若表格用例因切分粒度不满足,调 `max_chars` 或断言口径,但**不得放宽"每块带表头"这一条**)

- [ ] **Step 5: Commit + dev-notes**

```bash
git add app/rag/__init__.py app/rag/chunker.py tests/test_chunker.py
git commit -m "feat(rag): structure-aware markdown chunker"
```

---

### Task 4: embedder(BGE-M3 封装 + 测试替身)

**Files:**
- Create: `app/rag/embedder.py`
- Test: `tests/test_embedder.py`

**Interfaces:**
- Produces:
  - `class Embedder(Protocol)`:`encode(self, texts: list[str]) -> list[list[float]]`
  - `class BgeM3Embedder(model_name: str, device: str = "cpu", batch_size: int = 12, max_length: int = 8192)`:懒加载权重;`encode()` 返回 1024 维 float 列表
  - `class FakeEmbedder(dim: int = 1024)`:确定性哈希伪向量(测试用,不下载模型)
  - `get_embedder(settings) -> Embedder`(按 `EMBED_DEVICE` 决定 `use_fp16`,**CPU 时 `use_fp16=False`**)
  - `EMBED_DIM = 1024`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_embedder.py
from app.rag.embedder import EMBED_DIM, FakeEmbedder, get_embedder


def test_fake_embedder_shape_and_determinism():
    e = FakeEmbedder()
    v1 = e.encode(["邮费是多少"])[0]
    v2 = e.encode(["邮费是多少"])[0]
    assert len(v1) == EMBED_DIM and v1 == v2
    assert e.encode(["邮费是多少"])[0] != e.encode(["运费怎么算"])[0]


def test_fake_embedder_batch():
    e = FakeEmbedder()
    out = e.encode(["a", "b", "c"])
    assert len(out) == 3 and all(len(v) == EMBED_DIM for v in out)


def test_get_embedder_returns_real_impl_without_loading(monkeypatch):
    """get_embedder 只构造对象,不在构造期下载/加载模型。"""
    from app.config import get_settings

    e = get_embedder(get_settings())
    assert e.__class__.__name__ == "BgeM3Embedder"
    assert getattr(e, "_model", None) is None  # 懒加载
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_embedder.py -q`
Expected: FAIL(`ModuleNotFoundError: app.rag.embedder`)

- [ ] **Step 3: 实现**

```python
# app/rag/embedder.py
"""BGE-M3 向量化封装。真实模型懒加载;测试用 FakeEmbedder 免得下载权重。"""
import hashlib
from typing import Protocol

EMBED_DIM = 1024


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """确定性伪向量:同一文本恒得同一向量,不同文本不同(测试替身)。"""

    def __init__(self, dim: int = EMBED_DIM):
        self._dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [((h[i % len(h)] / 255.0) - 0.5) for i in range(self._dim)]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out


class BgeM3Embedder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 12, max_length: int = 8192):
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None  # 懒加载

    def _load(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            # CPU 上不能用 fp16
            self._model = BGEM3FlagModel(self._model_name, use_fp16=(self._device != "cpu"))
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        dense = model.encode(
            texts, batch_size=self._batch_size, max_length=self._max_length, return_dense=True
        )["dense_vecs"]
        return [list(map(float, v)) for v in dense]


def get_embedder(settings) -> Embedder:
    return BgeM3Embedder(
        model_name=settings.embed_model, device=settings.embed_device, batch_size=settings.embed_batch_size
    )
```

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_embedder.py -q`
Expected: PASS

- [ ] **Step 5: 真实模型冒烟(一次性,会下载权重 ~2.3GB)**

```bash
./.venv/Scripts/python.exe -c "
from app.rag.embedder import BgeM3Embedder
e = BgeM3Embedder('BAAI/bge-m3')
v = e.encode(['邮费是多少'])[0]
print('dim =', len(v))
"
```
Expected: `dim = 1024`。若下载慢,可先 `set HF_ENDPOINT=https://hf-mirror.com` 再跑。

- [ ] **Step 6: Commit + dev-notes**

```bash
git add app/rag/embedder.py tests/test_embedder.py
git commit -m "feat(rag): BGE-M3 embedder with lazy load and fake double"
```

---

### Task 5: store(Milvus 封装)

**Files:**
- Create: `app/rag/store.py`
- Test: `tests/test_vector_store.py`

**Interfaces:**
- Produces: `class VectorStore`:
  - `__init__(uri: str, token: str, collection: str, dim: int = 1024)`
  - `ensure_collection(self) -> None`(不存在则 `create_collection(collection_name, dimension=dim, metric_type="COSINE")`)
  - `upsert(self, ids: list[int], vectors: list[list[float]]) -> int`
  - `search(self, vector: list[float], top_k: int) -> list[tuple[int, float]]`(按分数降序,返回 `(id, score)`)
  - `delete(self, ids: list[int]) -> int`
  - `count(self) -> int`
  - `drop(self) -> None`(测试清理用)

- [ ] **Step 1: 写失败测试**(跑真 Milvus,使用独立集合)

```python
# tests/test_vector_store.py
import pytest

from app.config import get_settings
from app.rag.store import VectorStore

pytestmark = pytest.mark.anyio


@pytest.fixture
def store():
    s = get_settings()
    vs = VectorStore(uri=s.milvus_uri, token=s.milvus_token, collection=s.milvus_test_collection)
    vs.drop()
    vs.ensure_collection()
    yield vs
    vs.drop()


def _vec(seed: float, dim: int = 1024):
    return [seed] + [0.0] * (dim - 1)


def test_ensure_is_idempotent_and_count_starts_zero(store):
    store.ensure_collection()
    assert store.count() == 0


def test_upsert_then_search_returns_nearest(store):
    store.upsert([1, 2, 3], [_vec(1.0), _vec(0.5), _vec(-1.0)])
    hits = store.search(_vec(1.0), top_k=2)
    assert hits[0][0] == 1
    assert hits[0][1] > hits[1][1]


def test_upsert_is_idempotent_by_primary_key(store):
    store.upsert([7], [_vec(1.0)])
    store.upsert([7], [_vec(1.0)])  # 同主键再写
    assert store.count() == 1


def test_delete_removes_rows(store):
    store.upsert([1, 2], [_vec(1.0), _vec(0.5)])
    assert store.delete([1]) == 1
    assert store.count() == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_vector_store.py -q`
Expected: FAIL(`ModuleNotFoundError: app.rag.store`)。若 Milvus 未起先 `bash scripts/milvus.sh up`。

- [ ] **Step 3: 实现**

```python
# app/rag/store.py
"""Milvus 封装:集合只存 id + vector(1024, COSINE)。"""
import logging

from pymilvus import MilvusClient

logger = logging.getLogger("mewhelp.rag.store")


class VectorStore:
    def __init__(self, uri: str, token: str = "", collection: str = "knowledge", dim: int = 1024):
        self._client = MilvusClient(uri=uri, token=token) if token else MilvusClient(uri=uri)
        self._collection = collection
        self._dim = dim

    def ensure_collection(self) -> None:
        if not self._client.has_collection(self._collection):
            self._client.create_collection(
                collection_name=self._collection, dimension=self._dim, metric_type="COSINE"
            )

    def upsert(self, ids: list[int], vectors: list[list[float]]) -> int:
        if not ids:
            return 0
        rows = [{"id": int(i), "vector": v} for i, v in zip(ids, vectors)]
        res = self._client.upsert(collection_name=self._collection, data=rows)
        return int(res.get("upsert_count", 0))

    def search(self, vector: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        res = self._client.search(
            collection_name=self._collection, data=[vector], limit=top_k, output_fields=["id"]
        )
        hits = res[0] if res else []
        out = [(int(h["id"]), float(h["distance"])) for h in hits]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def delete(self, ids: list[int]) -> int:
        if not ids:
            return 0
        res = self._client.delete(collection_name=self._collection, ids=[int(i) for i in ids])
        return int(res.get("delete_count", 0))

    def count(self) -> int:
        stats = self._client.get_collection_stats(self._collection)
        return int(stats.get("row_count", 0))

    def drop(self) -> None:
        if self._client.has_collection(self._collection):
            self._client.drop_collection(self._collection)
```

> 若 `get_collection_stats` 在当前 pymilvus 版本签名不同 → 先查 Context7 再改,不要猜。

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_vector_store.py -q`
Expected: PASS(4 passed)

- [ ] **Step 5: Commit + dev-notes**

```bash
git add app/rag/store.py tests/test_vector_store.py
git commit -m "feat(rag): Milvus vector store (id + 1024-d COSINE)"
```

---

### Task 6: repository_knowledge(状态机 + 断点续跑 + 挖矿暂存)

**Files:**
- Create: `app/db/repository_knowledge.py`
- Test: `tests/test_repository_knowledge.py`

**Interfaces:**
- Produces(全部 `async`,首参 `AsyncSession`):
  - `content_hash(text: str) -> str`(归一化:去空白+lower+sha256)
  - `insert_chunks(session, chunks: list[dict]) -> list[int]`(逐条按 `content_hash` 幂等;返回新插入的 id 列表;`status='pending'`)
  - `link_neighbors(session, ids: list[int]) -> None`(按 id 顺序回填 `prev_id/next_id`)
  - `list_pending(session, limit: int | None = None) -> list[dict]`(条件:`status='pending' OR vector_id IS NULL`,按 id)
  - `mark_embedded(session, ids: list[int]) -> None`(回填 `vector_id=str(id)` 且 `status='embedded'`)
  - `fetch_by_ids(session, ids: list[int]) -> list[dict]`(返回顺序与入参一致;供 retriever)
  - `staging_insert(session, items: list[dict]) -> int`(按 `dedupe_hash` 幂等,状态 `staged`)
  - `staging_list(session, limit: int | None = None) -> list[dict]`(`status='staged'`)
  - `staging_mark(session, ids: list[int], status: str) -> None`(`promoted` / `dropped`)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_repository_knowledge.py
import pytest

from app.db import repository_knowledge as rk

pytestmark = pytest.mark.anyio


def _chunk(**kw):
    base = dict(
        doc_id="policy.md", category="售后政策", questions="退货政策",
        answer="7 天无理由退货。", text="售后政策\n退货政策\n7 天无理由退货。",
        section_path="售后政策 > 退货政策", content_type="政策",
        is_key_clause=False, order_index=0,
    )
    base.update(kw)
    base["content_hash"] = rk.content_hash(base["text"])
    return base


async def test_insert_chunks_is_idempotent_by_hash(db_session):
    ids1 = await rk.insert_chunks(db_session, [_chunk()])
    await db_session.commit()
    ids2 = await rk.insert_chunks(db_session, [_chunk()])  # 同内容再插
    await db_session.commit()
    assert ids2 == [] and len(ids1) == 1
    pending = await rk.list_pending(db_session)
    assert [c["id"] for c in pending] == ids1


async def test_mark_embedded_removes_from_pending(db_session):
    ids = await rk.insert_chunks(db_session, [_chunk()])
    await db_session.commit()
    await rk.mark_embedded(db_session, ids)
    await db_session.commit()
    assert await rk.list_pending(db_session) == []
    rows = await rk.fetch_by_ids(db_session, ids)
    assert rows[0]["vector_id"] == str(ids[0]) and rows[0]["status"] == "embedded"


async def test_link_neighbors_sets_pointers(db_session):
    ids = await rk.insert_chunks(
        db_session, [_chunk(text="A", content_hash=""), _chunk(text="B", content_hash="")]
    )
    await db_session.commit()
    await rk.link_neighbors(db_session, ids)
    await db_session.commit()
    rows = await rk.fetch_by_ids(db_session, ids)
    assert rows[0]["next_id"] == ids[1] and rows[1]["prev_id"] == ids[0]


async def test_fetch_by_ids_preserves_input_order(db_session):
    ids = await rk.insert_chunks(
        db_session, [_chunk(text="A"), _chunk(text="B"), _chunk(text="C")]
    )
    await db_session.commit()
    rows = await rk.fetch_by_ids(db_session, list(reversed(ids)))
    assert [r["id"] for r in rows] == list(reversed(ids))


async def test_staging_insert_dedupes_and_marks(db_session):
    items = [dict(conversation_id=1, source_message_ids="1,2", questions='["邮费怎么算"]',
                  answer="按地区收取", category="运费", dedupe_hash=rk.content_hash("邮费怎么算|按地区收取"))]
    assert await rk.staging_insert(db_session, items) == 1
    await db_session.commit()
    assert await rk.staging_insert(db_session, items) == 0  # 幂等
    staged = await rk.staging_list(db_session)
    assert len(staged) == 1
    await rk.staging_mark(db_session, [staged[0]["id"]], "promoted")
    await db_session.commit()
    assert await rk.staging_list(db_session) == []
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_repository_knowledge.py -q`
Expected: FAIL(`ModuleNotFoundError: app.db.repository_knowledge`)

- [ ] **Step 3: 实现**

```python
# app/db/repository_knowledge.py
"""knowledge_chunks / knowledge_staging 的读写与状态机(唯一写这两张 SQL 的地方)。"""
import hashlib
import re

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import KnowledgeChunk, KnowledgeStaging


def content_hash(text: str) -> str:
    norm = re.sub(r"\s+", "", (text or "")).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


async def insert_chunks(session: AsyncSession, chunks: list[dict]) -> list[int]:
    """按 content_hash 幂等插入;返回本次真正新插入的 id。"""
    new_ids: list[int] = []
    for c in chunks:
        h = c.get("content_hash") or content_hash(c["text"])
        exists = (
            await session.execute(select(KnowledgeChunk.id).where(KnowledgeChunk.content_hash == h))
        ).scalar_one_or_none()
        if exists:
            continue
        row = KnowledgeChunk(
            doc_id=c["doc_id"], category=c.get("category", ""), questions=c["questions"],
            answer=c["answer"], text=c["text"], section_path=c.get("section_path", ""),
            content_type=c.get("content_type", "政策"), is_key_clause=bool(c.get("is_key_clause", False)),
            content_hash=h, status="pending",
        )
        session.add(row)
        await session.flush()
        new_ids.append(row.id)
    return new_ids


async def link_neighbors(session: AsyncSession, ids: list[int]) -> None:
    for prev_id, cur_id, next_id in zip([None, *ids[:-1]], ids, [*ids[1:], None]):
        row = await session.get(KnowledgeChunk, cur_id)
        if row is not None:
            row.prev_id, row.next_id = prev_id, next_id
    await session.flush()


async def list_pending(session: AsyncSession, limit: int | None = None) -> list[dict]:
    stmt = (
        select(KnowledgeChunk)
        .where(or_(KnowledgeChunk.status == "pending", KnowledgeChunk.vector_id.is_(None)))
        .order_by(KnowledgeChunk.id)
    )
    if limit:
        stmt = stmt.limit(limit)
    rows = (await session.execute(stmt)).scalars().all()
    return [_to_dict(r) for r in rows]


async def mark_embedded(session: AsyncSession, ids: list[int]) -> None:
    for i in ids:
        row = await session.get(KnowledgeChunk, i)
        if row is not None:
            row.vector_id = str(i)
            row.status = "embedded"
    await session.flush()


async def fetch_by_ids(session: AsyncSession, ids: list[int]) -> list[dict]:
    if not ids:
        return []
    rows = (await session.execute(select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids)))).scalars().all()
    by_id = {r.id: _to_dict(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


async def staging_insert(session: AsyncSession, items: list[dict]) -> int:
    n = 0
    for it in items:
        h = it.get("dedupe_hash") or content_hash(f"{it['questions']}|{it['answer']}")
        exists = (
            await session.execute(select(KnowledgeStaging.id).where(KnowledgeStaging.dedupe_hash == h))
        ).scalar_one_or_none()
        if exists:
            continue
        session.add(
            KnowledgeStaging(
                conversation_id=it["conversation_id"], source_message_ids=it.get("source_message_ids", ""),
                questions=it["questions"], answer=it["answer"], category=it.get("category", ""),
                dedupe_hash=h, status="staged",
            )
        )
        n += 1
    await session.flush()
    return n


async def staging_list(session: AsyncSession, limit: int | None = None) -> list[dict]:
    stmt = select(KnowledgeStaging).where(KnowledgeStaging.status == "staged").order_by(KnowledgeStaging.id)
    if limit:
        stmt = stmt.limit(limit)
    rows = (await session.execute(stmt)).scalars().all()
    return [
        dict(id=r.id, conversation_id=r.conversation_id, source_message_ids=r.source_message_ids,
             questions=r.questions, answer=r.answer, category=r.category, dedupe_hash=r.dedupe_hash)
        for r in rows
    ]


async def staging_mark(session: AsyncSession, ids: list[int], status: str) -> None:
    for i in ids:
        row = await session.get(KnowledgeStaging, i)
        if row is not None:
            row.status = status
    await session.flush()


def _to_dict(r: KnowledgeChunk) -> dict:
    return dict(
        id=r.id, doc_id=r.doc_id, category=r.category, questions=r.questions, answer=r.answer,
        text=r.text, section_path=r.section_path, content_type=r.content_type,
        is_key_clause=bool(r.is_key_clause), prev_id=r.prev_id, next_id=r.next_id,
        content_hash=r.content_hash, vector_id=r.vector_id, status=r.status,
    )
```

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_repository_knowledge.py -q`
Expected: PASS

- [ ] **Step 5: Commit + dev-notes**

```bash
git add app/db/repository_knowledge.py tests/test_repository_knowledge.py
git commit -m "feat(rag): knowledge repository with idempotent status machine"
```

---

### Task 7: 建库脚本 build_knowledge(切分 → 双写 → 断点续跑)

**Files:**
- Create: `scripts/build_knowledge.py`
- Test: `tests/test_build_knowledge.py`

**Interfaces:**
- Consumes: `chunk_markdown`(Task 3)、`Embedder`(Task 4)、`VectorStore`(Task 5)、`repository_knowledge`(Task 6)。
- Produces:
  - `async def ingest_markdown(session_factory, embedder, store, doc_id: str, markdown: str, *, content_type="政策", max_chars=800, overlap=120, skip_embed: bool = False) -> dict`(返回 `{"inserted": n, "embedded": n}`)
  - `async def embed_pending(session_factory, embedder, store, *, limit: int | None = None, batch: int = 32) -> dict`(返回 `{"embedded": n}`)
  - `async def main(argv)` CLI:`--dir knowledge_docs`(默认)、`--doc <path>`、`--limit N`、`--content-type`、`--skip-embed`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_build_knowledge.py
import pytest

from app.config import get_settings
from app.db import repository_knowledge as rk
from app.rag.embedder import FakeEmbedder
from app.rag.store import VectorStore
from scripts.build_knowledge import embed_pending, ingest_markdown

pytestmark = pytest.mark.anyio

DOC = "# 售后政策\n\n## 运费说明\n\n下单运费按地区收取。退货寄回的运费由责任方承担。\n"


@pytest.fixture
def store():
    s = get_settings()
    vs = VectorStore(uri=s.milvus_uri, token=s.milvus_token, collection=s.milvus_test_collection)
    vs.drop()
    vs.ensure_collection()
    yield vs
    vs.drop()


async def test_ingest_writes_mysql_then_embeds(session_factory, db_session, store):
    stats = await ingest_markdown(session_factory, FakeEmbedder(), store, "policy.md", DOC)
    assert stats["inserted"] >= 1 and stats["embedded"] == stats["inserted"]
    assert await rk.list_pending(db_session) == []
    assert store.count() == stats["inserted"]


async def test_ingest_is_idempotent_on_rerun(session_factory, db_session, store):
    first = await ingest_markdown(session_factory, FakeEmbedder(), store, "policy.md", DOC)
    second = await ingest_markdown(session_factory, FakeEmbedder(), store, "policy.md", DOC)
    assert second["inserted"] == 0
    assert store.count() == first["inserted"]


async def test_interrupted_run_resumes_only_pending(session_factory, db_session, store):
    """验收②:模拟中断 —— 只写 MySQL 不向量化,再跑 embed_pending 补齐。"""
    await ingest_markdown(session_factory, FakeEmbedder(), store, "policy.md", DOC, skip_embed=True)
    pending = await rk.list_pending(db_session)
    assert pending, "应有未向量化的块"
    stats = await embed_pending(session_factory, FakeEmbedder(), store)
    assert stats["embedded"] == len(pending)
    assert await rk.list_pending(db_session) == []
    assert store.count() == len(pending)
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_build_knowledge.py -q`
Expected: FAIL(`ModuleNotFoundError: scripts.build_knowledge` 或 `ingest_markdown` 缺 `skip_embed` 参数)

- [ ] **Step 3: 实现**

```python
# scripts/build_knowledge.py
"""离线建库:Markdown → 结构切分 → MySQL(pending)→ 向量化 → Milvus → 回填。可重跑。

用法:
  python -m scripts.build_knowledge                 # 处理 knowledge_docs/ 全部
  python -m scripts.build_knowledge --doc knowledge_docs/退货政策.md
  python -m scripts.build_knowledge --limit 50 --skip-embed
"""
import argparse
import asyncio
import sys
from pathlib import Path

from app.config import get_settings
from app.db.base import get_sessionmaker
from app.db import repository_knowledge as rk
from app.rag.chunker import chunk_markdown
from app.rag.embedder import get_embedder
from app.rag.store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "knowledge_docs"


async def ingest_markdown(
    session_factory, embedder, store, doc_id: str, markdown: str,
    *, content_type: str = "政策", max_chars: int = 800, overlap: int = 120, skip_embed: bool = False,
) -> dict:
    chunks = chunk_markdown(doc_id, markdown, content_type=content_type, max_chars=max_chars, overlap=overlap)
    payload = [
        dict(doc_id=c.doc_id, category=c.category, questions=c.questions, answer=c.answer, text=c.text,
             section_path=c.section_path, content_type=c.content_type, is_key_clause=c.is_key_clause)
        for c in chunks
    ]
    async with session_factory() as session:
        new_ids = await rk.insert_chunks(session, payload)
        await rk.link_neighbors(session, new_ids)
        await session.commit()
    embedded = 0
    if not skip_embed:
        res = await embed_pending(session_factory, embedder, store)
        embedded = res["embedded"]
    return {"inserted": len(new_ids), "embedded": embedded}


async def embed_pending(session_factory, embedder, store, *, limit: int | None = None, batch: int = 32) -> dict:
    store.ensure_collection()
    embedded = 0
    while True:
        async with session_factory() as session:
            rows = await rk.list_pending(session, limit=limit)
        if not rows:
            break
        take = rows[: (limit or len(rows))]
        for i in range(0, len(take), batch):
            group = take[i : i + batch]
            vectors = embedder.encode([r["text"] for r in group])
            ids = [r["id"] for r in group]
            store.upsert(ids, vectors)
            async with session_factory() as session:
                await rk.mark_embedded(session, ids)
                await session.commit()
            embedded += len(ids)
        if limit:
            break
    return {"embedded": embedded}


async def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    p = argparse.ArgumentParser(description="离线建库:文档 → 切分 → 双写")
    p.add_argument("--doc", help="单个文件路径;缺省处理 --dir 下全部 .md")
    p.add_argument("--dir", default=str(DEFAULT_DIR))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--content-type", default="政策")
    p.add_argument("--skip-embed", action="store_true", help="只写 MySQL 不向量化(便于演练断点续跑)")
    args = p.parse_args(argv)

    sf = get_sessionmaker()
    store = VectorStore(uri=s.milvus_uri, token=s.milvus_token, collection=s.knowledge_collection)
    embedder = get_embedder(s)

    docs = [Path(args.doc)] if args.doc else sorted(Path(args.dir).glob("*.md"))
    if not docs:
        print(f"没有找到文档:{args.dir}", file=sys.stderr)
        return 2
    total = {"inserted": 0, "embedded": 0}
    for path in docs:
        stats = await ingest_markdown(
            sf, embedder, store, doc_id=path.name, markdown=path.read_text(encoding="utf-8"),
            content_type=args.content_type, max_chars=s.chunk_max_chars, overlap=s.chunk_overlap,
            skip_embed=args.skip_embed,
        )
        total["inserted"] += stats["inserted"]
        total["embedded"] += stats["embedded"]
        print(f"[{path.name}] 新增 {stats['inserted']} 块,已向量化 {stats['embedded']}")
    if args.skip_embed:
        print("(已跳过向量化;跑 embed_pending 或重跑本脚本会补齐)")
    print(f"合计:新增 {total['inserted']} 块,向量化 {total['embedded']} 块")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_build_knowledge.py -q`
Expected: PASS(3 passed;验收②的等价单测在此)

- [ ] **Step 5: Commit + dev-notes**

```bash
git add scripts/build_knowledge.py tests/test_build_knowledge.py
git commit -m "feat(rag): build_knowledge pipeline with resumable embedding"
```

---

### Task 8: 挖知识 job mine_knowledge(抽取 → 暂存 → 去重 → 入库)

**Files:**
- Create: `app/rag/miner.py`
- Create: `scripts/mine_knowledge.py`
- Test: `tests/test_miner.py`

**Interfaces:**
- Produces(`app/rag/miner.py`):
  - `class QAPair(BaseModel) { questions: list[str]; answer: str; category: str = "" }`
  - `class QABatch(BaseModel) { pairs: list[QAPair] }`
  - `async def extract_qa(llm, conversation_text: str) -> list[QAPair]`(用 `llm.with_structured_output(QABatch)`)
  - `def dedupe_pairs(pairs: list[QAPair], *, existing_vectors: list[list[float]] | None = None, embedder=None, sim_threshold: float = 0.95) -> list[QAPair]`(哈希精确 + 向量近重)
  - `async def mine(session_factory, llm, embedder, *, limit=20, lookback_days=30, sim_threshold=0.95) -> dict`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_miner.py
import pytest

from app.rag.embedder import FakeEmbedder
from app.rag.miner import QAPair, dedupe_pairs

pytestmark = pytest.mark.anyio


def test_dedupe_removes_exact_duplicates():
    pairs = [
        QAPair(questions=["邮费怎么算", "运费多少"], answer="按地区收取", category="运费"),
        QAPair(questions=["邮费怎么算", "运费多少"], answer="按地区收取", category="运费"),
    ]
    assert len(dedupe_pairs(pairs, embedder=FakeEmbedder(), existing_vectors=[])) == 1


def test_dedupe_removes_near_duplicates_by_vector():
    class SameEmbedder(FakeEmbedder):
        """所有文本都给同一向量,模拟"换说法但语义相同"。"""

        def encode(self, texts):
            return [[1.0] + [0.0] * 1023 for _ in texts]

    pairs = [
        QAPair(questions=["邮费怎么算"], answer="按地区收取", category="运费"),
        QAPair(questions=["运费如何计算"], answer="按地区收取运费", category="运费"),
    ]
    out = dedupe_pairs(pairs, embedder=SameEmbedder(), existing_vectors=[], sim_threshold=0.95)
    assert len(out) == 1  # 余弦≈1 → 判为近重


def test_dedupe_keeps_different_pairs():
    pairs = [
        QAPair(questions=["邮费怎么算"], answer="按地区收取", category="运费"),
        QAPair(questions=["怎么开发票"], answer="下单时勾选", category="发票"),
    ]
    assert len(dedupe_pairs(pairs, embedder=FakeEmbedder(), existing_vectors=[])) == 2
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_miner.py -q`
Expected: FAIL(`ModuleNotFoundError: app.rag.miner`)

- [ ] **Step 3: 实现**

```python
# app/rag/miner.py
"""从历史客服对话挖问答对:LLM 抽取 → 暂存 → 哈希+向量近重去重 → 入库(pending)。"""
import json
import logging

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db import repository_knowledge as rk
from app.db.models import Conversation, Message

logger = logging.getLogger("mewhelp.rag.miner")


class QAPair(BaseModel):
    questions: list[str] = Field(description="用户可能的问法,1-3 条,取自对话原话或同义改写")
    answer: str = Field(description="客服给出的结论性回答")
    category: str = Field(default="", description="主题分类,如 运费/发票/退换货")


class QABatch(BaseModel):
    pairs: list[QAPair] = Field(default_factory=list)


PROMPT = """你是知识库编辑。下面是一段客服与用户的对话,请抽取其中「用户问题 → 客服结论」的知识点。
要求:只抽有明确结论的;questions 用用户原话或同义改写(1-3 条);没有知识点就返回空列表。"""


def _pair_key(p: QAPair) -> str:
    return rk.content_hash(json.dumps(sorted(p.questions), ensure_ascii=False) + "|" + p.answer)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(x * x for x in b) ** 0.5 or 1.0
    return dot / (na * nb)


def dedupe_pairs(pairs, *, embedder=None, existing_vectors=None, sim_threshold: float = 0.95):
    """先按内容哈希精确去重,再用向量余弦做近重复过滤。"""
    seen_hash: set[str] = set()
    kept: list[QAPair] = []
    kept_vecs: list[list[float]] = list(existing_vectors or [])
    for p in pairs:
        h = _pair_key(p)
        if h in seen_hash:
            continue
        seen_hash.add(h)
        if embedder is not None:
            vec = embedder.encode([p.questions[0] + "|" + p.answer])[0]
            if any(_cosine(vec, other) >= sim_threshold for other in kept_vecs):
                continue
            kept_vecs.append(vec)
        kept.append(p)
    return kept


async def extract_qa(llm, conversation_text: str) -> list[QAPair]:
    structured = llm.with_structured_output(QABatch)
    out = await structured.ainvoke([{"role": "system", "content": PROMPT},
                                    {"role": "user", "content": conversation_text}])
    return list(getattr(out, "pairs", []) or [])


async def _load_conversations(session, *, limit: int, lookback_days: int) -> list[tuple[int, list[dict], list[int]]]:
    from datetime import datetime, timedelta

    since = datetime.now() - timedelta(days=lookback_days)
    rows = (
        await session.execute(
            select(Conversation).where(Conversation.created_at >= since).order_by(Conversation.id.desc()).limit(limit)
        )
    ).scalars().all()
    out = []
    for conv in rows:
        msgs = (
            await session.execute(select(Message).where(Message.conversation_id == conv.id).order_by(Message.id))
        ).scalars().all()
        if not msgs:
            continue
        out.append((conv.id, [{"role": m.role, "content": m.content or ""} for m in msgs], [m.id for m in msgs]))
    return out


async def mine(session_factory, llm, embedder, *, limit: int = 20, lookback_days: int = 30, sim_threshold: float = 0.95) -> dict:
    async with session_factory() as session:
        conversations = await _load_conversations(session, limit=limit, lookback_days=lookback_days)
    staged = 0
    for conv_id, messages, msg_ids in conversations:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages if m["content"])
        if not text.strip():
            continue
        pairs = await extract_qa(llm, text)
        pairs = dedupe_pairs(pairs, embedder=embedder, sim_threshold=sim_threshold)
        if not pairs:
            continue
        items = [
            dict(conversation_id=conv_id, source_message_ids=",".join(map(str, msg_ids)),
                 questions=json.dumps(p.questions, ensure_ascii=False), answer=p.answer,
                 category=p.category, dedupe_hash=_pair_key(p))
            for p in pairs
        ]
        async with session_factory() as session:
            staged += await rk.staging_insert(session, items)
            await session.commit()
    return {"conversations": len(conversations), "staged": staged}


async def promote_staged(session_factory, *, limit: int | None = None) -> dict:
    """把暂存区里存活项提升进 knowledge_chunks(pending),并标记 promoted。"""
    async with session_factory() as session:
        rows = await rk.staging_list(session, limit=limit)
        promoted = 0
        for r in rows:
            questions = json.loads(r["questions"]) if r["questions"].startswith("[") else [r["questions"]]
            chunk = dict(
                doc_id=f"mine:{r['conversation_id']}", category=r["category"], questions="、".join(questions),
                answer=r["answer"], text=rk_build_text(r["category"], "、".join(questions), r["answer"]),
                section_path="", content_type="挖矿QA", is_key_clause=False,
            )
            ids = await rk.insert_chunks(session, [chunk])
            await rk.staging_mark(session, [r["id"]], "promoted")
            promoted += 1 if ids else 0
        await session.commit()
    return {"promoted": promoted}


def rk_build_text(category: str, questions: str, answer: str) -> str:
    from app.rag.chunker import build_text

    return build_text(category, questions, answer)
```

```python
# scripts/mine_knowledge.py
"""挖知识 job:历史对话 → LLM 抽 QA → 暂存 → 去重 → 入库。可 cron / 任务计划调用。

用法:
  python -m scripts.mine_knowledge                 # 默认近 30 天、20 个会话
  python -m scripts.mine_knowledge --limit 50 --lookback 60
  python -m scripts.mine_knowledge --no-promote    # 只挖到暂存区,便于人工审核
"""
import argparse
import asyncio
import sys

from app.config import get_settings
from app.db.base import get_sessionmaker
from app.llm import get_chat_model
from app.rag.embedder import get_embedder
from app.rag.miner import mine, promote_staged


async def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    p = argparse.ArgumentParser(description="从历史对话挖知识")
    p.add_argument("--limit", type=int, default=s.mine_batch_size)
    p.add_argument("--lookback", type=int, default=s.mine_lookback_days)
    p.add_argument("--no-promote", action="store_true")
    args = p.parse_args(argv)

    sf = get_sessionmaker()
    stats = await mine(
        sf, get_chat_model(), get_embedder(s),
        limit=args.limit, lookback_days=args.lookback, sim_threshold=s.dedupe_sim_threshold,
    )
    print(f"扫描会话 {stats['conversations']} 个,新入暂存 {stats['staged']} 条")
    if not args.no_promote:
        pr = await promote_staged(sf)
        print(f"提升入库 {pr['promoted']} 条(状态 pending,待向量化)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_miner.py -q`
Expected: PASS

- [ ] **Step 5: Commit + dev-notes**

```bash
git add app/rag/miner.py scripts/mine_knowledge.py tests/test_miner.py
git commit -m "feat(rag): conversation mining job with hash+vector dedupe"
```

---

### Task 9: retriever + 替换 query_faq 内部实现

**Files:**
- Create: `app/rag/retriever.py`
- Modify: `app/tools/kb.py`
- Modify: `app/routers/chat.py`(装配处注入 retriever)
- Test: `tests/test_retriever.py`

**Interfaces:**
- Produces: `class KnowledgeRetriever(embedder, store, session_factory, top_k, score_threshold)`,`async def search(self, query: str) -> str`(命中返回组织好的文本;**无命中返回含「未找到」的提示**)。
- `make_kb_tools(session_factory, retriever=None)`:`retriever` 为 None 时回退到旧的 SQL LIKE 行为(ch02 兼容,便于过渡测试)。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_retriever.py
import pytest

from app.db import repository_knowledge as rk
from app.rag.embedder import FakeEmbedder
from app.rag.retriever import KnowledgeRetriever

pytestmark = pytest.mark.anyio


class StubStore:
    def __init__(self, hits):
        self._hits = hits
        self.upserted = None

    def ensure_collection(self): ...
    def upsert(self, ids, vectors): self.upserted = (ids, vectors)
    def search(self, vector, top_k=5): return self._hits[:top_k]
    def count(self): return len(self._hits)
    def delete(self, ids): return 0
    def drop(self): ...


async def _seed(db_session):
    ids = await rk.insert_chunks(
        db_session,
        [dict(doc_id="policy.md", category="售后政策", questions="运费说明",
              answer="退货寄回的运费由责任方承担。", text="售后政策\n运费说明\n退货寄回的运费由责任方承担。",
              section_path="售后政策 > 运费说明", content_type="政策", is_key_clause=False)],
    )
    await db_session.commit()
    return ids


async def test_search_returns_answer_text_on_hit(session_factory, db_session):
    ids = await _seed(db_session)
    r = KnowledgeRetriever(FakeEmbedder(), StubStore([(ids[0], 0.88)]), session_factory, top_k=5, score_threshold=0.5)
    out = await r.search("邮费是多少")
    assert "责任方承担" in out and "未找到" not in out


async def test_search_returns_not_found_below_threshold(session_factory, db_session):
    ids = await _seed(db_session)
    r = KnowledgeRetriever(FakeEmbedder(), StubStore([(ids[0], 0.10)]), session_factory, top_k=5, score_threshold=0.5)
    out = await r.search("邮费是多少")
    assert "未找到" in out


async def test_search_returns_not_found_on_empty_store(session_factory):
    r = KnowledgeRetriever(FakeEmbedder(), StubStore([]), session_factory, top_k=5, score_threshold=0.5)
    assert "未找到" in await r.search("任何问题")


async def test_query_faq_uses_retriever_and_keeps_contract(session_factory, db_session):
    ids = await _seed(db_session)
    from app.tools.kb import make_kb_tools

    r = KnowledgeRetriever(FakeEmbedder(), StubStore([(ids[0], 0.9)]), session_factory, top_k=5, score_threshold=0.5)
    [query_faq] = make_kb_tools(session_factory, retriever=r)
    out = await query_faq.ainvoke({"keyword": "邮费是多少"})
    assert isinstance(out, str) and "责任方承担" in out
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_retriever.py -q`
Expected: FAIL(`ModuleNotFoundError: app.rag.retriever`)

- [ ] **Step 3: 实现**

```python
# app/rag/retriever.py
"""在线检索:query → 向量 → Milvus Top-K → MySQL 取答案 → 拼文本。"""
import logging

logger = logging.getLogger("mewhelp.rag.retriever")


class KnowledgeRetriever:
    def __init__(self, embedder, store, session_factory, top_k: int = 5, score_threshold: float = 0.5):
        self._embedder = embedder
        self._store = store
        self._session_factory = session_factory
        self._top_k = top_k
        self._threshold = score_threshold

    async def search(self, query: str) -> str:
        from app.db import repository_knowledge as rk

        vector = self._embedder.encode([query])[0]
        hits = self._store.search(vector, top_k=self._top_k)
        kept = [(i, s) for i, s in hits if s >= self._threshold]
        if not kept:
            return f"FAQ 未找到与「{query}」相关的条目(向量检索无命中)"
        async with self._session_factory() as session:
            rows = await rk.fetch_by_ids(session, [i for i, _ in kept])
        score_by_id = {i: s for i, s in kept}
        parts = []
        for r in rows:
            head = f"问:{r['questions']}" if r["questions"] else ""
            cat = f"[{r['category']}] " if r["category"] else ""
            parts.append(f"{cat}{head}\n答:{r['answer']}\n(相似度 {score_by_id.get(r['id'], 0):.3f})")
        return "\n".join(parts)
```

`app/tools/kb.py` 改为支持注入 retriever(保留旧实现作为回退):

```python
def make_kb_tools(session_factory, retriever=None) -> list[BaseTool]:
    @tool
    async def query_faq(keyword: str) -> str:
        """查询店铺知识库(退货政策、运费、发票、保修等规则类问题)。用户问政策/规则/怎么退/运费/能不能开票时使用。"""
        if retriever is not None:
            return await retriever.search(keyword)
        async with session_factory() as session:  # 回退:ch02 的 SQL LIKE 实现
            hits = await repo.search_faq(session, keyword)
        if not hits:
            return f"FAQ 未找到与「{keyword}」相关条目(关键词检索无命中)"
        return "\n".join(f"问:{h['question']}\n答:{h['answer']}" for h in hits)

    return [query_faq]
```

`app/routers/chat.py` 的 `build_registry` 注入 retriever:

```python
def build_registry(session_factory, conversation_id: int | None) -> ToolRegistry:
    s = get_settings()
    retriever = KnowledgeRetriever(
        embedder=get_embedder(s),
        store=VectorStore(uri=s.milvus_uri, token=s.milvus_token, collection=s.knowledge_collection),
        session_factory=session_factory,
        top_k=s.rag_top_k,
        score_threshold=s.rag_score_threshold,
    )
    tools = [query_order, query_product, query_logistics, *make_kb_tools(session_factory, retriever=retriever)]
    if conversation_id is not None:
        tools += make_ops_tools(session_factory, conversation_id=conversation_id)
    return ToolRegistry(tools, non_retryable={"create_ticket"})
```

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_retriever.py tests/test_tools_kb_ops.py tests/test_chat_route_tools.py -q`
Expected: PASS(旧 kb/路由测试不回归)

- [ ] **Step 5: Commit + dev-notes**

```bash
git add app/rag/retriever.py app/tools/kb.py app/routers/chat.py tests/test_retriever.py
git commit -m "feat(rag): vector retriever wired into query_faq (contract unchanged)"
```

---

### Task 10: 样例知识文档 + 检索标注集 eval

**Files:**
- Create: `knowledge_docs/退货政策.md`、`knowledge_docs/运费说明.md`、`knowledge_docs/商品FAQ.md`、`knowledge_docs/售后手册.md`
- Create: `eval_data/retrieval_samples.json`
- Create: `scripts/eval_retrieval.py`

**Interfaces:**
- Produces: `python -m scripts.eval_retrieval` → 逐条「问句 → 命中块 → 是否命中期望」+ 命中率(需 Milvus + 真 BGE-M3 + 已建库)。

- [ ] **Step 1: 写样例文档**(关键:运费说明必须能回答「邮费」)

`knowledge_docs/运费说明.md` 至少包含:

```markdown
# 运费说明

## 下单运费

下单运费按收货地区收取:江浙沪 8 元,其他地区 12 元,满 99 元包邮。

## 退换货运费

退货寄回的运费由责任方承担:质量问题由商家承担,非质量问题由买家承担。
```

另三份文档分别覆盖 退货政策 / 商品 FAQ(含大表格) / 售后手册(长章节,用于验证递归切分与重叠)。

- [ ] **Step 2: 写标注集**

```json
[
  {"question": "邮费是多少", "expect_doc": "运费说明.md", "expect_keywords": ["运费", "地区"]},
  {"question": "退货运费谁出", "expect_doc": "运费说明.md", "expect_keywords": ["责任方"]},
  {"question": "怎么退货", "expect_doc": "退货政策.md", "expect_keywords": ["7 天"]},
  {"question": "能开票吗", "expect_doc": "商品FAQ.md", "expect_keywords": ["发票"]}
]
```

- [ ] **Step 3: 写评测脚本**

```python
# scripts/eval_retrieval.py
"""向量检索质量评测。用法:python -m scripts.eval_retrieval"""
import asyncio
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.db.base import get_sessionmaker
from app.db import repository_knowledge as rk
from app.rag.embedder import get_embedder
from app.rag.retriever import KnowledgeRetriever
from app.rag.store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "eval_data" / "retrieval_samples.json"


async def main() -> int:
    s = get_settings()
    sf = get_sessionmaker()
    store = VectorStore(uri=s.milvus_uri, token=s.milvus_token, collection=s.knowledge_collection)
    retriever = KnowledgeRetriever(get_embedder(s), store, sf, top_k=s.rag_top_k, score_threshold=s.rag_score_threshold)
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    hits = 0
    for i, sm in enumerate(samples, 1):
        out = await retriever.search(sm["question"])
        kw_ok = all(k in out for k in sm["expect_keywords"])
        doc_ok = sm["expect_doc"].replace(".md", "") in out
        ok = kw_ok and doc_ok
        hits += ok
        print(f"[{'PASS' if ok else 'FAIL'}] #{i} {sm['question']} -> {out[:80].replace(chr(10),' ')}")
    rate = hits / len(samples)
    print(f"\n检索命中率: {hits}/{len(samples)} = {rate:.0%}")
    return 0 if rate >= 0.75 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

> 说明:`expect_doc` 的判定依赖文本里能看出处;若 `retriever` 输出不含 doc 名,则本脚本的 `doc_ok` 退化为"关键词级"判定 —— 执行时若发现判定不成立,优先**在 retriever 文案里带上来源 doc_id**(同时更利于人工核对),而不是放宽标注意义。

- [ ] **Step 4: 建库 + 跑评测(真模型)**

```bash
bash scripts/milvus.sh up
./.venv/Scripts/python.exe -m scripts.build_knowledge
./.venv/Scripts/python.exe -m scripts.eval_retrieval
```
Expected: 命中率 ≥ 75%(基线记入 dev-notes);「邮费是多少」应 PASS。

- [ ] **Step 5: Commit + dev-notes**

```bash
git add knowledge_docs eval_data/retrieval_samples.json scripts/eval_retrieval.py
git commit -m "feat(rag): sample docs and retrieval eval set"
```

---

### Task 11: 端到端验收(聊天页 → query_faq → 向量召回)

**Files:**
- Create: `scripts/demo_rag.sh`, `scripts/fixtures/rag_miss.json`、`rag_hit.json`
- Modify: `README.md`

- [ ] **Step 1: 起服务并跑验收①**

```bash
./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000 &
bash scripts/demo_rag.sh http://127.0.0.1:8000
```
`demo_rag.sh` 复用 ch02 的 UTF-8 fixture 模式:
- `rag_hit.json` = `{"user_id":"demo","message":"邮费是多少"}`
- `rag_miss.json` = `{"user_id":"demo","message":"你们店几点开门"}`(知识库外的问题)
验收①通过标准:SSE 出现 `event: tool`(`query_faq`,ok),最终答复**包含运费说明内容**(如「按地区收取」「责任方承担」),且不再出现"查不到"。

- [ ] **Step 2: 验收②(真实中断)**

```bash
# 1) 只写 MySQL 不向量化(等价中断态)
./.venv/Scripts/python.exe -m scripts.build_knowledge --skip-embed
# 2) 查看未完成块数
./.venv/Scripts/python.exe -c "
import asyncio
from app.db.base import get_sessionmaker
from app.db import repository_knowledge as rk
async def main():
    sf = get_sessionmaker()
    async with sf() as s:
        print('pending:', len(await rk.list_pending(s)))
asyncio.run(main())"
# 3) 重跑补齐
./.venv/Scripts/python.exe -m scripts.build_knowledge
# 4) 再查应为 0
```
Expected: 步骤2 打印 `pending: N`(N>0),步骤4 打印 `pending: 0`。

- [ ] **Step 3: README 增补 ch03 段**

写明:Milvus 起停(`bash scripts/milvus.sh up|down|status`)、建库、挖矿 job、评测、验收命令、以及"只跑 dense 单路"的边界说明。

- [ ] **Step 4: Commit + dev-notes**(含两条验收实测输出)

```bash
git add scripts README.md
git commit -m "docs(rag): demo scripts, acceptance and README for ch03"
```

---

## Self-Review 记录

- **Spec 覆盖**:§1 目标/验收(→Task 7/10/11);§2 技术栈(→Task 1/4/5);§3 环境(→已完成并留痕);§4 分层骨架(→Task 3-9);§5 切分策略(→Task 3);§6 落库结构(→Task 2/6);§7 双写幂等与断点续跑(→Task 6/7 + 验收②);§8 挖矿 job(→Task 8);§9 在线检索与「未找到」(→Task 9);§10 配置(→Task 1);§11 测试/标注集/eval(→各 Task TDD + Task 10 + Task 11);§12 边界(→Global Constraints);§13 过程约束(→Global Constraints)。
- **占位扫描**:无 TBD/TODO;每个代码步给出可运行实现。Task 10 Step 3 有一处"执行时若判定不成立…"的处理指引,属明确的两选一处置,非占位。
- **类型/签名一致性**:`Chunk`(Task 3)→ `insert_chunks` 入参 dict(Task 6,字段与 `Chunk` 对齐)→ `ingest_markdown(..., skip_embed)`(Task 7,测试与实现一致);`Embedder.encode`(Task 4)被 Task 6/7/8/9 一致消费;`VectorStore.upsert/search/count/drop`(Task 5)被 Task 7/9/10 一致消费;`KnowledgeRetriever.search`(Task 9)被 kb 工具与 eval 脚本一致消费;`rk.content_hash` 在 Task 6 定义、Task 8 复用。
- **已知执行期需复核点**:`MilvusClient.get_collection_stats` 返回键名(Task 5,签名不符先查 Context7);`BGEM3FlagModel` CPU 下 `use_fp16=False`(Task 4 已写死);`with_structured_output` 承载 list 字段(Task 8,沿用 ch02 已验路径);FlagEmbedding 首次运行会下权重(Task 4 Step 5 冒烟)。
