# ch02 Function Calling 工具链 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给现有电商客服聊天接上五个业务工具的 Function Calling 能力——模型单轮自主选工具、执行(超时/重试)、结果回灌后收敛,最终答复仍 SSE 逐 token;聊天与工具轨迹落 MySQL;聊天页显示工具徽章。

**Architecture:** 在 ch01 的 FastAPI 服务上新增 `app/db`(SQLAlchemy 2.0 async 模型与 repository,唯一写 SQL 处)、`app/tools`(五个 `@tool` + 注册表/派发/执行外壳)、`app/agent/tool_runner.py`(单轮工具往返)。`ChatService` 改两段式:①绑定工具流式跑第一段(顺带 live 吐文本),累加 chunk 得 `tool_calls`;②若有则全执行并推 `event: tool` 状态帧、落库 assistant(tool_calls)+tool 结果;③用**不绑工具**的模型回灌后流式收敛。历史以 `messages` 表为准。

**Tech Stack:** FastAPI、SQLAlchemy 2.0 async + aiomysql(MySQL 5.7 本地)、LangChain 1.x(`langchain.tools.tool` / `bind_tools` / `AIMessageChunk.tool_call_chunks` / `ToolMessage`)、pydantic-settings、pytest。

**Spec:** `docs/superpowers/specs/2026-09-10-function-calling-tools-design.md`

## Global Constraints

- **单轮语义(硬约束)**:全程只有**一次工具往返**;收敛阶段**不绑定工具**,物理上不可能二次调用。一次返回多个 tool_calls **全执行**,但仍属同一轮。
- **历史以数据库为准**:`conversations`/`messages` 是唯一真相;ch01 的内存 `SessionStore` 退役(连同 `app/sessions.py` 与其测试)。
- **DB 连接**:本机 MySQL `127.0.0.1:3306`,库 `mewhelp`(用户已建表,见 `sql/schema.sql`),**不改 DDL、不引 Alembic**。测试用独立库 `mewhelp_test`(由测试 fixture 用 SQLAlchemy metadata 建表)。
- **mock 确定性**:`random.Random(sha256(输入 + MOCK_SEED_SALT))`,同一输入恒返回同一数据。
- **不做**:Agent 多轮循环、RAG/向量检索、工具接真实电商/物流接口。
- **每 Task 结束**向 `dev-notes/ch02.md` 追加一段(四样:用户关键原话 / 我的关键产出 / 用户拒绝或纠偏 / 翻车与返工)。禁止收尾一次性补记。
- **文档先行**:凡 FastAPI / SQLAlchemy / LangChain / aiomysql 具体 API,动手前先用 Context7 查官方当前文档核对,禁止凭记忆。已核:`langchain.tools.tool`、`bind_tools`、`AIMessageChunk.tool_call_chunks`、`ToolMessage(content, tool_call_id=)`、`create_async_engine`/`async_sessionmaker`/`DeclarativeBase`。
- **执行模式**:Subagent-Driven(每 Task 独立子代理 + 任务间 review)。
- **聊天页改造是 Vibe 例外**(Task 12),不套 TDD/评审。
- 安装驱动:`pip install "sqlalchemy>=2.0" aiomysql pymysql`(已装:sqlalchemy 2.0.52 / aiomysql 0.3.2;`pymysql` 供建表脚本)。

---

### Task 1: 配置与 DB 基建(engine / sessionmaker)

**Files:**
- Modify: `app/config.py`
- Modify: `.env.example`
- Create: `app/db/__init__.py`
- Create: `app/db/base.py`
- Test: `tests/test_db_base.py`

**Interfaces:**
- Produces(Settings 新增字段): `mysql_host: str="127.0.0.1"`, `mysql_port: int=3306`, `mysql_user: str="root"`, `mysql_password: str=""`, `mysql_database: str="mewhelp"`, `mysql_test_database: str="mewhelp_test"`, `tool_timeout_seconds: float=8.0`, `tool_max_retries: int=2`, `mock_seed_salt: str="mewhelp"`。
- Produces(`app/db/base.py`): `class Base(DeclarativeBase)`;`build_database_url(settings, database: str | None = None) -> str`;`get_engine() -> AsyncEngine`;`get_sessionmaker() -> async_sessionmaker[AsyncSession]`;`async def dispose_engine() -> None`;`async def create_all(engine) -> None`。

- [ ] **Step 1: 文档核对**

Context7 `/websites/sqlalchemy_en_20` 核对 `create_async_engine` / `async_sessionmaker` / `DeclarativeBase`;确认 MySQL 异步 URL 形如 `mysql+aiomysql://user:pass@host:port/db?charset=utf8mb4`。记录到 dev-notes。

- [ ] **Step 2: 写失败测试**

```python
# tests/test_db_base.py
from urllib.parse import quote_plus

from app.config import get_settings
from app.db.base import Base, build_database_url


def test_build_database_url_uses_aiomysql_and_charset():
    s = get_settings()
    url = build_database_url(s)
    assert url.startswith("mysql+aiomysql://")
    assert "charset=utf8mb4" in url
    assert url.endswith("/" + s.mysql_database)


def test_build_database_url_can_target_other_database():
    s = get_settings()
    url = build_database_url(s, database="mewhelp_test")
    assert url.endswith("/mewhelp_test")


def test_build_database_url_escapes_password():
    from app.config import Settings

    s = Settings(mysql_user="u", mysql_password="p@ss:word/1", mysql_host="h", mysql_port=3307, mysql_database="d")
    url = build_database_url(s)
    assert quote_plus("p@ss:word/1") in url and "h:3307" in url


def test_base_metadata_null():
    assert Base.metadata is not None
```

- [ ] **Step 3: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_db_base.py -q`
Expected: FAIL(`ModuleNotFoundError: app.db` / Settings 无字段)

- [ ] **Step 4: 实现**

`app/config.py` 追加字段(在 `cs_staff_name` 之后):

```python
    # MySQL
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "mewhelp"
    mysql_test_database: str = "mewhelp_test"
    # 工具执行
    tool_timeout_seconds: float = 8.0
    tool_max_retries: int = 2
    mock_seed_salt: str = "mewhelp"
```

`.env.example` 追加:

```
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=change-me
MYSQL_DATABASE=mewhelp
MYSQL_TEST_DATABASE=mewhelp_test

TOOL_TIMEOUT_SECONDS=8
TOOL_MAX_RETRIES=2
MOCK_SEED_SALT=mewhelp
```

```python
# app/db/__init__.py
"""数据访问层(db)。"""
```

```python
# app/db/base.py
from urllib.parse import quote_plus

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


def build_database_url(settings: Settings, database: str | None = None) -> str:
    db = database or settings.mysql_database
    pwd = quote_plus(settings.mysql_password)
    return (
        f"mysql+aiomysql://{settings.mysql_user}:{pwd}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{db}?charset=utf8mb4"
    )


_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            build_database_url(get_settings()),
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def dispose_engine() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


async def create_all(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

- [ ] **Step 5: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_db_base.py -q`
Expected: PASS

- [ ] **Step 6: Commit + dev-notes**

```bash
git add app/config.py .env.example app/db tests/test_db_base.py
git commit -m "feat(db): async engine/sessionmaker and MySQL settings"
```
并向 `dev-notes/ch02.md` 追加 Task 1 段。

---

### Task 2: SQLAlchemy 模型(镜像 DDL)

**Files:**
- Create: `app/db/models.py`
- Test: `tests/test_db_models.py`

**Interfaces:**
- Produces: `Conversation`、`Message`、`Faq`、`Ticket` 四个 ORM 类(`Base` 子类),表名与列名严格对齐 `sql/schema.sql`;`Message.role` 取 `user/assistant/tool`;`Conversation.status` 取 `进行中/已转人工/已结束`;`Ticket.ticket_type` 取 `售后/投诉/咨询`、`Ticket.status` 取 `待处理/已处理`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_db_models.py
from sqlalchemy import inspect

from app.db.models import Conversation, Faq, Message, Ticket


def test_table_names_match_ddl():
    assert Conversation.__tablename__ == "conversations"
    assert Message.__tablename__ == "messages"
    assert Faq.__tablename__ == "faq"
    assert Ticket.__tablename__ == "tickets"


def test_message_columns():
    cols = {c.name for c in inspect(Message).columns}
    assert {"conversation_id", "role", "content", "tool_calls", "tool_call_id", "created_at"} <= cols


def test_role_enum_values():
    role = Message.__table__.c.role.type
    assert set(role.enums) == {"user", "assistant", "tool"}


def test_ticket_primary_key_and_enums():
    assert [c.name for c in Ticket.__table__.primary_key.columns] == ["ticket_no"]
    assert set(Ticket.__table__.c.ticket_type.type.enums) == {"售后", "投诉", "咨询"}
    assert set(Ticket.__table__.c.status.type.enums) == {"待处理", "已处理"}
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_db_models.py -q`
Expected: FAIL(`ModuleNotFoundError: app.db.models`)

- [ ] **Step 3: 实现**

```python
# app/db/models.py
import datetime

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        Enum("进行中", "已转人工", "已结束", name="conversation_status"),
        nullable=False,
        server_default="进行中",
    )
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool", name="message_role"), nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())


class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id"), nullable=False, index=True
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(Enum("售后", "投诉", "咨询", name="ticket_type"), nullable=False)
    status: Mapped[str] = mapped_column(
        Enum("待处理", "已处理", name="ticket_status"), nullable=False, server_default="待处理"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
```

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_db_models.py -q`
Expected: PASS

- [ ] **Step 5: Commit + dev-notes**

```bash
git add app/db/models.py tests/test_db_models.py
git commit -m "feat(db): SQLAlchemy models mirroring schema.sql"
```
并向 `dev-notes/ch02.md` 追加 Task 2 段。

---

### Task 3: 测试用 DB fixture(mewhelp_test)

**Files:**
- Create: `tests/db_utils.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_db_fixture.py`

**Interfaces:**
- Produces: `tests/db_utils.py` 提供 `async def create_database_if_missing(name: str) -> None`(用 pymysql 连 server 建库)、`async_session_factory` fixture 用的 `db_session` fixture。
- Produces(conftest fixtures): `db_session`(AsyncSession,指向 `mewhelp_test`,每个测试前建表+清空)、`session_factory`(`async_sessionmaker`)。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_db_fixture.py
import pytest
from sqlalchemy import text

pytestmark = pytest.mark.anyio


async def test_db_session_points_to_test_database(db_session):
    r = await db_session.execute(text("SELECT DATABASE()"))
    assert r.scalar() == "mewhelp_test"


async def test_tables_exist_in_test_db(db_session):
    r = await db_session.execute(text("SHOW TABLES"))
    tables = {row[0] for row in r.fetchall()}
    assert {"conversations", "messages", "faq", "tickets"} <= tables
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_db_fixture.py -q`
Expected: FAIL(`fixture 'db_session' not found`)

- [ ] **Step 3: 实现**

```python
# tests/db_utils.py
"""测试库工具:建库、按 sql/schema.sql 建表、清空表。全部走 pymysql(同步,供 sync/async fixture 共用)。"""
from pathlib import Path

import pymysql

from app.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "sql" / "schema.sql"
_TABLES = ("messages", "tickets", "conversations", "faq")  # 清空顺序:先子表(有外键)


def _connect(database: str | None = None):
    s = get_settings()
    return pymysql.connect(
        host=s.mysql_host, port=s.mysql_port, user=s.mysql_user, password=s.mysql_password,
        database=database, charset="utf8mb4", autocommit=True,
    )


def create_database_if_missing(name: str) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{name}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        conn.close()


def apply_schema(name: str) -> None:
    text = SCHEMA.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("--")]
    stmts = [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
    conn = _connect(name)
    try:
        with conn.cursor() as cur:
            for stmt in stmts:
                cur.execute(stmt)
    finally:
        conn.close()


def truncate_all(name: str) -> None:
    conn = _connect(name)
    try:
        with conn.cursor() as cur:
            cur.execute("SET FOREIGN_KEY_CHECKS=0")
            for t in _TABLES:
                cur.execute(f"TRUNCATE TABLE `{t}`")
            cur.execute("SET FOREIGN_KEY_CHECKS=1")
    finally:
        conn.close()
```

`tests/conftest.py` **整体替换**为下面内容(注意:环境变量必须在 import app.config 之前设置,否则会连到开发库):

```python
import os

# 必须最先执行:.env 里 MYSQL_DATABASE=mewhelp,这里用环境变量覆盖到测试库
# (pydantic-settings 中环境变量优先级高于 .env 文件)
os.environ.setdefault("MYSQL_DATABASE", "mewhelp_test")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.base import build_database_url
from app.main import app
from tests.db_utils import apply_schema, create_database_if_missing, truncate_all


@pytest.fixture(scope="session")
def _test_db() -> str:
    s = get_settings()
    create_database_if_missing(s.mysql_test_database)
    apply_schema(s.mysql_test_database)
    return s.mysql_test_database


@pytest.fixture
def client(_test_db):
    """同步 TestClient。依赖 _test_db 保证测试库有表,并在每个用例前清空。"""
    truncate_all(_test_db)
    app.state.chat_model = None
    app.state.extract_model = None
    with TestClient(app) as c:
        yield c
    truncate_all(_test_db)


@pytest.fixture
async def session_factory(_test_db):
    s = get_settings()
    engine = create_async_engine(build_database_url(s, database=s.mysql_test_database))
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


@pytest.fixture
async def db_session(session_factory):
    truncate_all(get_settings().mysql_test_database)
    async with session_factory() as session:
        yield session
```

> 注意:`sql/schema.sql` 的表用 `IF NOT EXISTS`,故 `apply_schema` 可重复执行;每个用例前用 `truncate_all` 清数据。**路由测试(client,fake model)也会连测试库**,不再污染 `mewhelp`。

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_db_fixture.py -q`
Expected: PASS(需本机 MySQL 可连;连不上则报错并停下排查)

- [ ] **Step 5: Commit + dev-notes**

```bash
git add tests/db_utils.py tests/conftest.py tests/test_db_fixture.py
git commit -m "test(db): mewhelp_test fixtures with engine-per-test"
```
并向 `dev-notes/ch02.md` 追加 Task 3 段。

---

### Task 4: repository(唯一写 SQL 处)

**Files:**
- Create: `app/db/repository.py`
- Test: `tests/test_repository.py`

**Interfaces:**
- Produces(全部 `async`,首参 `session: AsyncSession`):
  - `get_or_create_conversation(session, *, conversation_id: int | None, user_id: str) -> Conversation`
  - `append_message(session, *, conversation_id: int, role: str, content: str | None = None, tool_calls: list | None = None, tool_call_id: str | None = None) -> None`
  - `load_history(session, conversation_id: int) -> list[dict]`(每项 `{"role","content","tool_calls","tool_call_id"}`)
  - `search_faq(session, keyword: str, limit: int = 3) -> list[dict]`(每项 `{"question","answer","category"}`)
  - `create_ticket(session, *, conversation_id: int, description: str, ticket_type: str) -> str`(返回 ticket_no,冲突重试 5 次)
  - `set_conversation_status(session, conversation_id: int, status: str) -> None`
  - `next_ticket_no(session) -> str`(形如 `T20260910001`)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_repository.py
import datetime

import pytest

from app.db import repository as repo

pytestmark = pytest.mark.anyio


async def test_get_or_create_conversation_creates_and_reuses(db_session):
    c = await repo.get_or_create_conversation(db_session, conversation_id=None, user_id="u1")
    await db_session.commit()
    assert c.id and c.user_id == "u1" and c.status == "进行中"
    same = await repo.get_or_create_conversation(db_session, conversation_id=c.id, user_id="u1")
    assert same.id == c.id


async def test_append_and_load_history_roundtrip(db_session):
    c = await repo.get_or_create_conversation(db_session, conversation_id=None, user_id="u")
    await repo.append_message(db_session, conversation_id=c.id, role="user", content="你好")
    await repo.append_message(
        db_session, conversation_id=c.id, role="assistant", content=None,
        tool_calls=[{"id": "call_1", "name": "query_order", "args": {"order_id": "1001"}}],
    )
    await repo.append_message(
        db_session, conversation_id=c.id, role="tool", content="订单已发货", tool_call_id="call_1"
    )
    await db_session.commit()
    hist = await repo.load_history(db_session, c.id)
    assert [m["role"] for m in hist] == ["user", "assistant", "tool"]
    assert hist[1]["tool_calls"][0]["name"] == "query_order"
    assert hist[2]["tool_call_id"] == "call_1"


async def test_search_faq_matches_keyword_else_empty(db_session):
    from app.db.models import Faq

    db_session.add_all([
        Faq(question="退货政策是什么", answer="7 天无理由退货", category="售后"),
        Faq(question="发票怎么开", answer="下单时勾选", category="发票"),
    ])
    await db_session.commit()
    hits = await repo.search_faq(db_session, "退货")
    assert hits and "无理由" in hits[0]["answer"]
    assert await repo.search_faq(db_session, "邮费") == []


async def test_create_ticket_returns_no_and_sets_status(db_session):
    c = await repo.get_or_create_conversation(db_session, conversation_id=None, user_id="u")
    await db_session.commit()
    no = await repo.create_ticket(db_session, conversation_id=c.id, description="要人工", ticket_type="售后")
    await db_session.commit()
    assert no.startswith("T" + datetime.date.today().strftime("%Y%m%d"))
    await repo.set_conversation_status(db_session, c.id, "已转人工")
    await db_session.commit()
    fresh = await repo.get_or_create_conversation(db_session, conversation_id=c.id, user_id="u")
    assert fresh.status == "已转人工"
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_repository.py -q`
Expected: FAIL(`ModuleNotFoundError: app.db.repository`)

- [ ] **Step 3: 实现**

```python
# app/db/repository.py
import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Faq, Message, Ticket


async def get_or_create_conversation(
    session: AsyncSession, *, conversation_id: int | None, user_id: str
) -> Conversation:
    if conversation_id is not None:
        existing = await session.get(Conversation, conversation_id)
        if existing is not None:
            return existing
    conv = Conversation(user_id=user_id)
    session.add(conv)
    await session.flush()
    return conv


async def append_message(
    session: AsyncSession,
    *,
    conversation_id: int,
    role: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
) -> None:
    session.add(
        Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
    )
    await session.flush()


async def load_history(session: AsyncSession, conversation_id: int) -> list[dict]:
    rows = (
        await session.execute(
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
        )
    ).scalars().all()
    return [
        {"role": r.role, "content": r.content, "tool_calls": r.tool_calls, "tool_call_id": r.tool_call_id}
        for r in rows
    ]


async def search_faq(session: AsyncSession, keyword: str, limit: int = 3) -> list[dict]:
    kw = (keyword or "").strip()
    if not kw:
        return []
    rows = (
        await session.execute(
            select(Faq)
            .where(Faq.question.like(f"%{kw}%") | Faq.answer.like(f"%{kw}%"))
            .limit(limit)
        )
    ).scalars().all()
    return [{"question": r.question, "answer": r.answer, "category": r.category} for r in rows]


async def next_ticket_no(session: AsyncSession) -> str:
    day = datetime.date.today().strftime("%Y%m%d")
    prefix = f"T{day}"
    rows = (
        await session.execute(select(Ticket.ticket_no).where(Ticket.ticket_no.like(f"{prefix}%")))
    ).scalars().all()
    seq = max([int(x[len(prefix):] or 0) for x in rows], default=0) + 1
    return f"{prefix}{seq:03d}"


async def create_ticket(
    session: AsyncSession, *, conversation_id: int, description: str, ticket_type: str
) -> str:
    last_err: Exception | None = None
    for _ in range(5):
        no = await next_ticket_no(session)
        session.add(
            Ticket(
                ticket_no=no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
            )
        )
        try:
            await session.flush()
            return no
        except IntegrityError as exc:  # 并发撞号,重试
            last_err = exc
            await session.rollback()
    raise RuntimeError(f"生成工单号失败: {last_err}")


async def set_conversation_status(session: AsyncSession, conversation_id: int, status: str) -> None:
    conv = await session.get(Conversation, conversation_id)
    if conv is not None:
        conv.status = status
        await session.flush()
```

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_repository.py -q`
Expected: PASS

- [ ] **Step 5: Commit + dev-notes**

```bash
git add app/db/repository.py tests/test_repository.py
git commit -m "feat(db): conversation/message/faq/ticket repository"
```
并向 `dev-notes/ch02.md` 追加 Task 4 段。

---

### Task 5: mock 数据 + 三个业务工具(business.py)

**Files:**
- Create: `app/tools/__init__.py`
- Create: `app/tools/mockdata.py`
- Create: `app/tools/business.py`
- Test: `tests/test_tools_business.py`

**Interfaces:**
- Produces(`app/tools/mockdata.py`): `mock_order(order_id: str) -> dict`、`mock_product(product_id: str) -> dict`、`mock_logistics(order_id: str) -> dict`(确定性)。
- Produces(`app/tools/business.py`): 模块级 `@tool` 常量 `query_order`、`query_product`、`query_logistics`,各自返回 JSON 字符串。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tools_business.py
import json

from app.tools.business import query_logistics, query_order, query_product
from app.tools.mockdata import mock_logistics, mock_order


def test_mock_is_deterministic_per_input():
    assert mock_order("1001") == mock_order("1001")
    assert mock_order("1001") != mock_order("1002")
    assert mock_logistics("1001") == mock_logistics("1001")


def test_order_shape():
    o = mock_order("1001")
    assert o["order_id"] == "1001" and o["status"] and o["amount"] > 0 and o["product_id"]


def test_tools_expose_name_and_schema():
    assert query_order.name == "query_order"
    assert query_product.name == "query_product"
    assert query_logistics.name == "query_logistics"
    assert "order_id" in query_order.args_schema.model_fields
    assert query_order.description  # 供模型选型的说明不能为空


def test_tool_invoke_returns_json_string():
    out = json.loads(query_order.invoke({"order_id": "1001"}))
    assert out["order_id"] == "1001"
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tools_business.py -q`
Expected: FAIL(`ModuleNotFoundError: app.tools`)

- [ ] **Step 3: 实现**

```python
# app/tools/__init__.py
"""业务工具层。"""
```

```python
# app/tools/mockdata.py
"""确定性 mock 数据:同一输入恒返回同一结果,便于演示与评测断言。"""
import hashlib
import random

from app.config import get_settings

_PRODUCTS = [
    ("猫粮 5kg", "宠物主粮", 129.0),
    ("自动喂食器", "宠物电器", 299.0),
    ("猫抓板", "宠物玩具", 49.0),
    ("逗猫棒", "宠物玩具", 19.0),
    ("跑步机", "宠物器材", 1899.0),
]
_CARRIERS = ["顺丰速运", "中通快递", "圆通速递", "京东物流"]
_NODES = ["已揽收", "运输中", "到达分拨中心", "派送中", "已签收"]


def _rng(*parts: object) -> random.Random:
    raw = "|".join(str(p) for p in parts) + "|" + get_settings().mock_seed_salt
    seed = int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12], 16)
    return random.Random(seed)


def mock_order(order_id: str) -> dict:
    r = _rng("order", order_id)
    name, cat, price = r.choice(_PRODUCTS)
    return {
        "order_id": str(order_id),
        "status": r.choice(["已发货", "待发货", "已完成", "已取消"]),
        "amount": round(price * r.randint(1, 3), 2),
        "product_id": f"P{r.randint(100, 999)}",
        "product_name": name,
        "category": cat,
        "created_at": f"2026-{r.randint(1, 9):02d}-{r.randint(1, 28):02d}",
    }


def mock_product(product_id: str) -> dict:
    r = _rng("product", product_id)
    name, cat, price = r.choice(_PRODUCTS)
    return {
        "product_id": str(product_id),
        "name": name,
        "category": cat,
        "price": price,
        "stock": r.randint(0, 200),
        "warranty": r.choice(["7 天无理由", "15 天包换", "一年质保"]),
    }


def mock_logistics(order_id: str) -> dict:
    r = _rng("logistics", order_id)
    steps = _NODES[: r.randint(2, len(_NODES))]
    return {
        "order_id": str(order_id),
        "carrier": r.choice(_CARRIERS),
        "tracking_no": "SF" + "".join(str(r.randint(0, 9)) for _ in range(12)),
        "current_node": steps[-1],
        "traces": [
            {"node": n, "time": f"2026-09-{r.randint(1, 9):02d} {r.randint(8, 20)}:00"} for n in steps
        ],
        "eta_days": r.randint(1, 4),
    }
```

```python
# app/tools/business.py
"""订单 / 商品 / 物流三个查询工具(演示用 mock 数据,不接真实接口)。"""
import json

from langchain_core.tools import tool

from app.tools.mockdata import mock_logistics, mock_order, mock_product


@tool
def query_order(order_id: str) -> str:
    """按订单号查询订单信息(状态、金额、购买的商品)。当用户提到具体订单、订单号、买的东西时使用。"""
    return json.dumps(mock_order(order_id), ensure_ascii=False)


@tool
def query_product(product_id: str) -> str:
    """按商品编号查询商品信息(名称、价格、库存、保修)。当用户问商品本身的价格/库存/保修时使用。"""
    return json.dumps(mock_product(product_id), ensure_ascii=False)


@tool
def query_logistics(order_id: str) -> str:
    """按订单号查询物流进度(承运商、运单号、当前节点、预计到达)。当用户问发货没、到哪了、什么时候到、物流进度时使用。"""
    return json.dumps(mock_logistics(order_id), ensure_ascii=False)
```

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tools_business.py -q`
Expected: PASS

- [ ] **Step 5: Commit + dev-notes**

```bash
git add app/tools tests/test_tools_business.py
git commit -m "feat(tools): deterministic mock order/product/logistics tools"
```
并向 `dev-notes/ch02.md` 追加 Task 5 段。

---

### Task 6: FAQ 与工单工具(需要 DB 会话,按请求构建)

**Files:**
- Create: `app/tools/kb.py`
- Create: `app/tools/ops.py`
- Test: `tests/test_tools_kb_ops.py`

**Interfaces:**
- Consumes: `app.db.repository`(Task 4)。
- Produces: `make_kb_tools(session_factory) -> list[BaseTool]`(内含 `query_faq(keyword: str) -> str`);`make_ops_tools(session_factory, conversation_id: int) -> list[BaseTool]`(内含 `create_ticket(description: str, ticket_type: str) -> str`)。工具内部各自开 `async with session_factory() as session` 并 commit。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tools_kb_ops.py
import pytest

from app.tools.kb import make_kb_tools
from app.tools.ops import make_ops_tools

pytestmark = pytest.mark.anyio


async def test_query_faq_finds_seeded_row(session_factory, db_session):
    from app.db.models import Faq

    db_session.add(Faq(question="退货政策是什么", answer="7 天无理由退货", category="售后"))
    await db_session.commit()

    [query_faq] = make_kb_tools(session_factory)
    out = await query_faq.ainvoke({"keyword": "退货"})
    assert "7 天无理由退货" in out


async def test_query_faq_returns_explicit_miss(session_factory):
    [query_faq] = make_kb_tools(session_factory)
    out = await query_faq.ainvoke({"keyword": "邮费"})
    assert "未找到" in out  # 漏召回时给模型一个明确信号,而非空字符串


async def test_create_ticket_writes_row_and_returns_no(session_factory, db_session):
    from app.db import repository as repo
    from app.db.models import Ticket

    conv = await repo.get_or_create_conversation(db_session, conversation_id=None, user_id="u")
    await db_session.commit()

    [create_ticket] = make_ops_tools(session_factory, conversation_id=conv.id)
    out = await create_ticket.ainvoke({"description": "需要人工处理", "ticket_type": "售后"})
    assert "T" in out and "已创建" in out

    from sqlalchemy import select

    rows = (await db_session.execute(select(Ticket).where(Ticket.conversation_id == conv.id))).scalars().all()
    assert len(rows) == 1 and rows[0].description == "需要人工处理"
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tools_kb_ops.py -q`
Expected: FAIL(`ModuleNotFoundError: app.tools.kb`)

- [ ] **Step 3: 实现**

```python
# app/tools/kb.py
"""FAQ 查询工具:关键词 LIKE 查 faq 表。"""
from langchain_core.tools import BaseTool, tool
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db import repository as repo


def make_kb_tools(session_factory: async_sessionmaker) -> list[BaseTool]:
    @tool
    async def query_faq(keyword: str) -> str:
        """查询店铺常见问题库(退货政策、发票、运费、保修等规则类问题)。用户问政策/规则/怎么退/能不能开票时使用。"""
        async with session_factory() as session:
            hits = await repo.search_faq(session, keyword)
        if not hits:
            return f"FAQ 未找到与「{keyword}」相关条目(关键词检索无命中)"
        return "\n".join(f"问:{h['question']}\n答:{h['answer']}" for h in hits)

    return [query_faq]
```

```python
# app/tools/ops.py
"""人工工单工具:写 tickets 表。"""
from langchain_core.tools import BaseTool, tool
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db import repository as repo


def make_ops_tools(session_factory: async_sessionmaker, conversation_id: int) -> list[BaseTool]:
    @tool
    async def create_ticket(description: str, ticket_type: str) -> str:
        """创建人工工单转人工处理。ticket_type 只能取「售后」「投诉」「咨询」。用户要求人工/投诉/机器人解决不了时使用。"""
        if ticket_type not in ("售后", "投诉", "咨询"):
            return f"工单类型非法:{ticket_type},只能是 售后/投诉/咨询"
        async with session_factory() as session:
            no = await repo.create_ticket(
                session, conversation_id=conversation_id, description=description, ticket_type=ticket_type
            )
            await session.commit()
        return f"人工工单已创建,工单号 {no},我们会尽快处理"

    return [create_ticket]
```

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tools_kb_ops.py -q`
Expected: PASS

- [ ] **Step 5: Commit + dev-notes**

```bash
git add app/tools/kb.py app/tools/ops.py tests/test_tools_kb_ops.py
git commit -m "feat(tools): faq search and create_ticket tools"
```
并向 `dev-notes/ch02.md` 追加 Task 6 段。

---

### Task 7: 工具注册表与派发

**Files:**
- Create: `app/tools/registry.py`
- Test: `tests/test_tools_registry.py`

**Interfaces:**
- Produces: `ToolRegistry(tools: list[BaseTool])`,方法 `names() -> list[str]`、`by_name(name: str) -> BaseTool | None`、`bindable() -> list[BaseTool]`、`async execute(name: str, args: dict, *, timeout: float, retries: int) -> ToolExecutionResult`。
- Produces: `@dataclass ToolExecutionResult { call_id: str; name: str; args: dict; ok: bool; content: str; error: str | None; attempts: int }`。
- 派发规则:未知工具 → `ok=False, error="未知工具"`,不重试;pydantic 参数校验失败 → `ok=False`,不重试;异常/超时 → 重试 `retries` 次后 `ok=False`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tools_registry.py
import asyncio

import pytest
from langchain_core.tools import tool

from app.tools.registry import ToolExecutionResult, ToolRegistry

pytestmark = pytest.mark.anyio


@tool
def echo(text: str) -> str:
    """回显。"""
    return "echo:" + text


@tool
def boom() -> str:
    """总是失败。"""
    raise RuntimeError("kaboom")


async def test_execute_ok_and_schema_validation_error():
    reg = ToolRegistry([echo])
    r = await reg.execute("echo", {"text": "hi"}, timeout=2, retries=0)
    assert r.ok and r.content == "echo:hi" and r.attempts == 1
    bad = await reg.execute("echo", {"wrong": 1}, timeout=2, retries=0)
    assert not bad.ok and bad.attempts == 1  # 参数校验失败不重试


async def test_unknown_tool_is_error_without_retry():
    reg = ToolRegistry([echo])
    r = await reg.execute("nope", {}, timeout=2, retries=2)
    assert not r.ok and "未知工具" in (r.error or "") and r.attempts == 1


async def test_retry_then_fail_captures_error():
    reg = ToolRegistry([boom])
    r = await reg.execute("boom", {}, timeout=2, retries=2)
    assert not r.ok and r.attempts == 3 and "kaboom" in (r.error or "")


async def test_timeout_is_enforced_and_retried(monkeypatch):
    calls = {"n": 0}

    async def slow(*_a, **_k):
        calls["n"] += 1
        await asyncio.sleep(0.5)
        return "late"

    reg = ToolRegistry([echo])
    reg.by_name("echo").ainvoke = slow  # type: ignore[assignment]
    r = await reg.execute("echo", {"text": "x"}, timeout=0.05, retries=1)
    assert not r.ok and calls["n"] == 2 and "超时" in (r.error or "")


def test_bindable_and_names():
    reg = ToolRegistry([echo, boom])
    assert set(reg.names()) == {"echo", "boom"}
    assert reg.bindable() == [echo, boom]
    assert isinstance(ToolExecutionResult("c", "echo", {}, True, "x", None, 1), ToolExecutionResult)
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tools_registry.py -q`
Expected: FAIL(`ModuleNotFoundError: app.tools.registry`)

- [ ] **Step 3: 实现**

```python
# app/tools/registry.py
"""工具注册表:登记、按名派发、参数校验、超时与重试。"""
import asyncio
import logging
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool
from pydantic import ValidationError

logger = logging.getLogger("mewhelp.tools")


@dataclass
class ToolExecutionResult:
    call_id: str
    name: str
    args: dict
    ok: bool
    content: str
    error: str | None = None
    attempts: int = 1


class ToolRegistry:
    def __init__(self, tools: list[BaseTool]):
        self._tools = {t.name: t for t in tools}
        self._order = list(tools)

    def names(self) -> list[str]:
        return [t.name for t in self._order]

    def bindable(self) -> list[BaseTool]:
        return list(self._order)

    def by_name(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    async def execute(
        self, name: str, args: dict, *, timeout: float, retries: int, call_id: str = ""
    ) -> ToolExecutionResult:
        tool = self.by_name(name)
        if tool is None:
            return ToolExecutionResult(call_id, name, args, False, "", f"未知工具:{name}", 1)

        attempt = 0
        last_err = ""
        while attempt <= retries:
            attempt += 1
            try:
                out = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
                text = out if isinstance(out, str) else str(out)
                return ToolExecutionResult(call_id, name, args, True, text, None, attempt)
            except ValidationError as exc:
                # 参数不合法,重试无意义
                return ToolExecutionResult(call_id, name, args, False, "", f"参数校验失败:{exc}", attempt)
            except asyncio.TimeoutError:
                last_err = f"工具执行超时(>{timeout}s)"
                logger.warning("tool %s timeout attempt=%s", name, attempt)
            except Exception as exc:  # noqa: BLE001 —— 工具失败不应打断整轮
                last_err = f"{type(exc).__name__}: {exc}"
                logger.warning("tool %s failed attempt=%s: %s", name, attempt, last_err)
            if attempt <= retries:
                await asyncio.sleep(0.2 * attempt)
        return ToolExecutionResult(call_id, name, args, False, "", last_err, attempt)
```

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tools_registry.py -q`
Expected: PASS

- [ ] **Step 5: Commit + dev-notes**

```bash
git add app/tools/registry.py tests/test_tools_registry.py
git commit -m "feat(tools): registry with dispatch, validation, timeout retry"
```
并向 `dev-notes/ch02.md` 追加 Task 7 段。

---

### Task 8: 单轮工具往返(tool_runner)

**Files:**
- Create: `app/agent/__init__.py`
- Create: `app/agent/tool_runner.py`
- Test: `tests/test_tool_runner.py`

**Interfaces:**
- Consumes: `ToolRegistry.execute`(Task 7)。
- Produces:
  - `async def stream_first_round(model_with_tools, messages, on_delta) -> tuple[list[dict], str]`:对**已 bind_tools 的模型** `astream`,逐 chunk 调 `on_delta(text)`(可能有文本),最终返回 `(tool_calls, text)`;`tool_calls` 为 `[{"id","name","args"}]`。
  - `async def execute_tool_calls(registry, tool_calls, on_status, *, timeout, retries) -> list[ToolExecutionResult]`:`asyncio.gather` 并行执行全部调用;每个调用 `on_status(phase)` 两次(`running` / `ok|error`)。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tool_runner.py
import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import tool

from app.agent.tool_runner import execute_tool_calls, stream_first_round
from app.tools.registry import ToolRegistry

pytestmark = pytest.mark.anyio


class FakeBoundModel:
    """模拟已 bind_tools 的模型:按脚本吐出 chunk。"""

    def __init__(self, chunks):
        self._chunks = chunks

    async def astream(self, _messages, **_kw):
        for c in self._chunks:
            yield c


def _text_chunk(text):
    return AIMessageChunk(content=text)


def _tool_chunk(name, args_json, call_id):
    return AIMessageChunk(content="", tool_call_chunks=[
        {"name": name, "args": args_json, "id": call_id, "index": 0, "type": "tool_call_chunk"}
    ])


async def test_stream_first_round_collects_text_and_no_tools():
    model = FakeBoundModel([_text_chunk("你好"), _text_chunk("呀")])
    seen = []
    calls, text = await stream_first_round(model, [], lambda t: seen.append(t))
    assert text == "你好呀" and seen == ["你好", "呀"] and calls == []


async def test_stream_first_round_collects_tool_calls():
    model = FakeBoundModel([_tool_chunk("query_order", '{"order_id": "1001"}', "call_a")])
    calls, text = await stream_first_round(model, [], lambda t: None)
    assert text == "" and len(calls) == 1
    assert calls[0]["name"] == "query_order" and calls[0]["args"] == {"order_id": "1001"}


@tool
def ping(x: str) -> str:
    """ping。"""
    return "pong:" + x


async def test_execute_tool_calls_reports_status_for_each():
    reg = ToolRegistry([ping])
    events = []

    def on_status(phase, call_id, name, args, summary):
        events.append((phase, name, summary))

    results = await execute_tool_calls(
        reg, [{"id": "c1", "name": "ping", "args": {"x": "a"}}],
        on_status, timeout=2, retries=0,
    )
    assert results[0].ok and results[0].content == "pong:a"
    assert events[0][0] == "running" and events[-1][0] == "ok"
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tool_runner.py -q`
Expected: FAIL(`ModuleNotFoundError: app.agent.tool_runner`)

- [ ] **Step 3: 实现**

```python
# app/agent/__init__.py
"""单轮工具调用编排层。"""
```

```python
# app/agent/tool_runner.py
"""单轮工具往返:第一阶段流式收集 tool_calls,第二阶段执行并回灌。"""
from typing import Callable

from app.tools.registry import ToolExecutionResult, ToolRegistry


def _text_of(chunk) -> str:
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


async def stream_first_round(model_with_tools, messages, on_delta: Callable[[str], None]) -> tuple[list[dict], str]:
    """跑绑定工具的第一段:逐 chunk 回调文本,累加得到 tool_calls(合并自 tool_call_chunks)。"""
    acc = None
    parts: list[str] = []
    async for chunk in model_with_tools.astream(messages):
        acc = chunk if acc is None else acc + chunk
        text = _text_of(chunk)
        if text:
            parts.append(text)
            on_delta(text)
    calls = getattr(acc, "tool_calls", None) or []
    normalized = [{"id": c.get("id") or "", "name": c.get("name") or "", "args": c.get("args") or {}} for c in calls]
    return normalized, "".join(parts)


async def execute_tool_calls(
    registry: ToolRegistry,
    tool_calls: list[dict],
    on_status: Callable[[str, str, str, dict, str], None],
    *,
    timeout: float,
    retries: int,
) -> list[ToolExecutionResult]:
    """并行执行该轮全部工具调用;每次调用前后各回调一次状态。"""
    import asyncio

    async def run_one(call: dict) -> ToolExecutionResult:
        call_id, name, args = call["id"], call["name"], call["args"]
        on_status("running", call_id, name, args, "")
        result = await registry.execute(name, args, timeout=timeout, retries=retries, call_id=call_id)
        summary = result.content if result.ok else (result.error or "")
        on_status("ok" if result.ok else "error", call_id, name, args, summary)
        return result

    if not tool_calls:
        return []
    return list(await asyncio.gather(*(run_one(c) for c in tool_calls)))
```

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tool_runner.py -q`
Expected: PASS

- [ ] **Step 5: Commit + dev-notes**

```bash
git add app/agent tests/test_tool_runner.py
git commit -m "feat(agent): single-round tool runner (collect + execute)"
```
并向 `dev-notes/ch02.md` 追加 Task 8 段。

---

### Task 9: ChatService 两段式改造(工具往返 + 收敛流式 + 落库)

**Files:**
- Modify: `app/chat.py`
- Modify: `app/context.py`(仅新增消息↔dict 的转换辅助,不动裁剪逻辑)
- Test: `tests/test_chat_tools.py`

**Interfaces:**
- Consumes: `repository`(Task 4)、`ToolRegistry`(Task 7)、`tool_runner`(Task 8)。
- Produces: `ChatService(model, registry_factory, session_factory, system_prompt, settings)`,其中 `registry_factory: Callable[[int], ToolRegistry]` 按 conversation_id 造该轮工具集(首轮也能拿到 `create_ticket`);`async stream_turn(user_id: str, conversation_id: int | None, user_text: str) -> AsyncIterator[ServerSentEvent]`;
  帧序:新建会话先 `session`(data 含 `conversation_id`);有工具时 `tool`(running/ok|error);文本 `delta`;结尾 `done`(data 含 `conversation_id`);异常 `error`。
- Produces(`app/context.py`): `to_langchain_messages(messages: list[dict]) -> list[BaseMessage]`(支持 `tool` 行 → `ToolMessage(content, tool_call_id=)`;`assistant` 行带 `tool_calls` → `AIMessage(content="", tool_calls=[...])`)。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_chat_tools.py
import pytest
from langchain_core.messages import AIMessageChunk

from app.chat import ChatService
from app.config import Settings
from app.tools.registry import ToolRegistry

pytestmark = pytest.mark.anyio

SETTINGS = Settings(history_budget_tokens=2048, tool_timeout_seconds=2, tool_max_retries=0)


class ScriptedModel:
    """第一次 astream 出工具调用; 第二段(不带工具)出文本。"""

    def __init__(self, first_chunks, final_text):
        self._first, self._final = first_chunks, final_text
        self.bound = False

    def bind_tools(self, tools, **_kw):
        self.bound = True
        return self

    async def astream(self, _messages, **_kw):
        if self.bound:
            for c in self._first:
                yield c
        else:
            for ch in self._final:
                yield AIMessageChunk(content=ch)


def _toolcall(name, args_json, cid):
    return AIMessageChunk(content="", tool_call_chunks=[
        {"name": name, "args": args_json, "id": cid, "index": 0, "type": "tool_call_chunk"}
    ])


async def _collect(gen):
    return [e async for e in gen]


async def test_tool_round_emits_frames_and_persists(session_factory, db_session):
    from app.tools.business import query_order

    model = ScriptedModel([_toolcall("query_order", '{"order_id": "1001"}', "c1")], ["订单", "已发货"])
    svc = ChatService(
        model=model,
        registry_factory=lambda cid: ToolRegistry([query_order]),
        session_factory=session_factory,
        system_prompt="你是客服",
        settings=SETTINGS,
    )
    events = await _collect(svc.stream_turn("u1", None, "订单1001到哪了"))
    names = [e.event for e in events]
    assert names[0] == "session" and names[-1] == "done"
    assert "tool" in names and "delta" in names
    tool_frames = [e.data for e in events if e.event == "tool"]
    assert tool_frames[0]["status"] == "running" and tool_frames[-1]["status"] == "ok"
    final_text = "".join(e.data["content"] for e in events if e.event == "delta")
    assert final_text == "订单已发货"

    from app.db import repository as repo

    conv_id = events[0].data["conversation_id"]
    hist = await repo.load_history(db_session, conv_id)
    assert [m["role"] for m in hist][:3] == ["user", "assistant", "tool"]
    assert hist[-1]["role"] == "assistant" and hist[-1]["content"] == "订单已发货"


async def test_pure_chat_path_still_streams_without_tools(session_factory):
    model = ScriptedModel([], ["你好", "呀"])
    svc = ChatService(
        model=model, registry_factory=lambda cid: ToolRegistry([]), session_factory=session_factory,
        system_prompt="你是客服", settings=SETTINGS,
    )
    events = await _collect(svc.stream_turn("u1", None, "你好"))
    assert "tool" not in [e.event for e in events]
    assert "".join(e.data["content"] for e in events if e.event == "delta") == "你好呀"
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_chat_tools.py -q`
Expected: FAIL(ChatService 签名不含 registry/session_factory)

- [ ] **Step 3: 实现**

`app/context.py` 追加:

```python
def to_langchain_messages(messages: list[dict]) -> list:
    """dict 行 → LangChain 消息。支持 assistant 带 tool_calls 与 role=tool 的 ToolMessage。"""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    out = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            out.append(SystemMessage(m.get("content") or ""))
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            out.append(AIMessage(content=m.get("content") or "", tool_calls=list(tcs)))
        elif role == "tool":
            out.append(ToolMessage(m.get("content") or "", tool_call_id=m.get("tool_call_id") or ""))
        else:
            out.append(HumanMessage(m.get("content") or ""))
    return out
```
(同时把 `app/chat.py` 里旧的 `to_langchain_messages` 删除,统一用 `context` 的这份;`text_of_chunk` 保留在 chat.py。)

`app/chat.py` 顶部 import 需新增:`from app.agent.tool_runner import execute_tool_calls, stream_first_round`、`from app.context import to_langchain_messages, trim_to_budget`、`from app.db import repository as repo`。

`app/chat.py` 重写 `ChatService`(保留 `text_of_chunk`):

```python
class ChatService:
    def __init__(self, model, registry_factory, session_factory, system_prompt: str, settings):
        self._model = model
        self._registry_factory = registry_factory
        self._session_factory = session_factory
        self._system_prompt = system_prompt
        self._settings = settings

    def _build_lc_messages(self, history: list[dict]) -> list:
        """messages 表历史 → 裁到 token 预算内 → LangChain 消息(system 置顶;当前 user 已在 history 尾部)。"""
        rows = [
            {
                "role": h["role"],
                "content": h["content"],
                **({"tool_calls": h["tool_calls"]} if h.get("tool_calls") else {}),
                **({"tool_call_id": h["tool_call_id"]} if h.get("tool_call_id") else {}),
            }
            for h in history
        ]
        body = trim_to_budget(rows, self._settings.history_budget_tokens)
        return to_langchain_messages([{"role": "system", "content": self._system_prompt}, *body])

    async def stream_turn(self, user_id: str, conversation_id: int | None, user_text: str):
        async with self._session_factory() as session:
            conv = await repo.get_or_create_conversation(
                session, conversation_id=conversation_id, user_id=user_id
            )
            created = conversation_id is None or conv.id != conversation_id
            await repo.append_message(session, conversation_id=conv.id, role="user", content=user_text)
            await session.commit()
            cid = conv.id

        registry = self._registry_factory(cid)  # 该轮工具集(create_ticket 需要 cid,首轮也可用)

        if created:
            yield ServerSentEvent(event="session", data={"conversation_id": cid})

        try:
            async with self._session_factory() as session:
                lc_messages = self._build_lc_messages(await repo.load_history(session, cid))

            # ── 第一段:绑定工具,唯一一次工具往返 ──
            first_parts: list[str] = []
            tool_calls, first_text = await stream_first_round(
                self._model.bind_tools(registry.bindable()), lc_messages, first_parts.append
            )
            if first_text:
                yield ServerSentEvent(event="delta", data={"content": first_text})
                await asyncio.sleep(0)

            if tool_calls:
                status_frames: list[ServerSentEvent] = []

                def on_status(phase, call_id, name, args, summary):
                    data = {"call_id": call_id, "name": name, "status": phase}
                    if phase == "running":
                        data["args"] = args
                    else:
                        data["summary"] = summary[:200]
                    status_frames.append(ServerSentEvent(event="tool", data=data))

                results = await execute_tool_calls(
                    registry, tool_calls, on_status,
                    timeout=self._settings.tool_timeout_seconds,
                    retries=self._settings.tool_max_retries,
                )
                for frame in status_frames:
                    yield frame

                async with self._session_factory() as session:
                    await repo.append_message(
                        session, conversation_id=cid, role="assistant", content=None,
                        tool_calls=[{"id": c["id"], "name": c["name"], "args": c["args"]} for c in tool_calls],
                    )
                    for r in results:
                        await repo.append_message(
                            session, conversation_id=cid, role="tool",
                            content=r.content if r.ok else f"[工具失败] {r.error}",
                            tool_call_id=r.call_id,
                        )
                    if any(r.name == "create_ticket" and r.ok for r in results):
                        await repo.set_conversation_status(session, cid, "已转人工")
                    await session.commit()
                    converge_messages = self._build_lc_messages(await repo.load_history(session, cid))

                # ── 第二段:不绑工具 → 物理上不可能再调工具,必然收敛 ──
                final_parts: list[str] = []
                async for chunk in self._model.astream(converge_messages):
                    text = text_of_chunk(chunk)
                    if not text:
                        continue
                    final_parts.append(text)
                    yield ServerSentEvent(event="delta", data={"content": text})
                    await asyncio.sleep(0)
                final_text = "".join(final_parts)
            else:
                final_text = first_text

            if final_text:
                async with self._session_factory() as session:
                    await repo.append_message(session, conversation_id=cid, role="assistant", content=final_text)
                    await session.commit()
        except Exception:
            logger.exception("chat stream error conversation=%s", cid)
            yield ServerSentEvent(event="error", data={"message": "抱歉,服务暂时不可用,请稍后重试"})
            return

        yield ServerSentEvent(event="done", data={"conversation_id": cid, "finish": True})
```

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_chat_tools.py -q`
Expected: PASS

- [ ] **Step 5: Commit + dev-notes**

```bash
git add app/chat.py app/context.py tests/test_chat_tools.py
git commit -m "feat(chat): two-stage tool round with streaming convergence"
```
并向 `dev-notes/ch02.md` 追加 Task 9 段。

---

### Task 10: 路由与装配(conversation_id 契约、退役 sessions.py)

**Files:**
- Modify: `app/schemas.py`(ChatRequest)
- Modify: `app/routers/chat.py`
- Modify: `app/main.py`
- Delete: `app/sessions.py`、`tests/test_sessions.py`(ch01 内存会话退役)
- Modify: `tests/test_chat_route.py`、`tests/test_chat_service.py`(改到新契约)
- Test: `tests/test_chat_route_tools.py`

**Interfaces:**
- Produces: `ChatRequest(session_id: str | None = None, conversation_id: int | None = None, user_id: str = "web-anonymous", message: str)`。
- Produces: `resolve_chat_service(request) -> ChatService`(用 `request.app.state.chat_model`、`get_sessionmaker()`、`registry_factory=lambda cid: build_registry(get_sessionmaker(), cid)`)。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_chat_route_tools.py
from langchain_core.messages import AIMessageChunk

from app.main import app


class ScriptedModel:
    def __init__(self, first, final):
        self._first, self._final, self.bound = first, final, False

    def bind_tools(self, tools, **_kw):
        self.bound = True
        return self

    async def astream(self, _m, **_kw):
        for c in (self._first if self.bound else self._final):
            yield c


def _toolcall(name, args_json, cid):
    return AIMessageChunk(content="", tool_call_chunks=[
        {"name": name, "args": args_json, "id": cid, "index": 0, "type": "tool_call_chunk"}
    ])


def _parse(body):
    import json
    import re

    return [(m[0], json.loads(m[1])) for m in re.findall(r"event: (\w+)\ndata: (.+)", body)]


def test_route_returns_conversation_id_and_tool_frames(client):
    app.state.chat_model = ScriptedModel([_toolcall("query_order", '{"order_id":"1001"}', "c1")], [AIMessageChunk(content="已发货")])
    r1 = client.post("/api/chat", json={"user_id": "u1", "message": "订单1001到哪了"})
    assert r1.status_code == 200
    frames = _parse(r1.text)
    names = [n for n, _ in frames]
    assert names[0] == "session" and names[-1] == "done" and "tool" in names
    cid = dict(frames)["session"]["conversation_id"]
    assert cid > 0

    # 第二轮:带 conversation_id,不应再发 session 帧
    app.state.chat_model = ScriptedModel([], [AIMessageChunk(content="嗯嗯")])
    r2 = client.post("/api/chat", json={"user_id": "u1", "conversation_id": cid, "message": "谢谢"})
    frames2 = _parse(r2.text)
    assert "session" not in [n for n, _ in frames2]
    assert dict(frames2)["done"]["conversation_id"] == cid
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_chat_route_tools.py -q`
Expected: FAIL(路由仍用旧契约)

- [ ] **Step 3: 实现**

`app/schemas.py` 的 `ChatRequest` 改为:

```python
class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=128, description="(ch01 遗留,忽略)")
    conversation_id: int | None = Field(default=None, description="会话 id;缺省则新建并回传")
    user_id: str = Field(default="web-anonymous", max_length=64)
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def _message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message 不能为空白")
        return v
```

`app/routers/chat.py`:

```python
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.chat import ChatService
from app.config import get_settings
from app.db.base import get_sessionmaker
from app.llm import get_chat_model
from app.prompts import format_chat_system_prompt
from app.schemas import ChatRequest
from app.tools.business import query_logistics, query_order, query_product
from app.tools.kb import make_kb_tools
from app.tools.ops import make_ops_tools
from app.tools.registry import ToolRegistry

router = APIRouter()


def build_registry(session_factory, conversation_id: int | None) -> ToolRegistry:
    tools = [query_order, query_product, query_logistics, *make_kb_tools(session_factory)]
    if conversation_id is not None:
        tools += make_ops_tools(session_factory, conversation_id=conversation_id)
    return ToolRegistry(tools)


def resolve_chat_service(request: Request) -> ChatService:
    model = getattr(request.app.state, "chat_model", None) or get_chat_model()
    s = get_settings()
    return ChatService(
        model=model,
        registry_factory=lambda cid: build_registry(get_sessionmaker(), cid),
        session_factory=get_sessionmaker(),
        system_prompt=format_chat_system_prompt(s),
        settings=s,
    )


@router.post("/api/chat", response_class=EventSourceResponse)
async def chat(body: ChatRequest, service: ChatService = Depends(resolve_chat_service)) -> AsyncIterator[ServerSentEvent]:
    async for event in service.stream_turn(body.user_id, body.conversation_id, body.message):
        yield event
```

> 说明:`create_ticket` 需要 `conversation_id`,而首轮 conversation 在 `stream_turn` 内部才创建 → 由 `registry_factory(cid)` 在拿到 cid 后按轮造工具集解决(Task 9 已在 `stream_turn` 里调用),因此**首轮也能调用 `create_ticket`**。测试用 `ScriptedModel` 不关心工具集合。

`app/main.py`:去掉 `state.chat_model` 之外的无关项无需改动;`app/sessions.py` 与 `tests/test_sessions.py` 删除;`tests/test_chat_service.py`、`tests/test_chat_route.py` 按新契约改写(用 `session_factory` fixture + `ScriptedModel`),或并入 `test_chat_tools.py` / `test_chat_route_tools.py` 后删除旧文件。

- [ ] **Step 4: 运行确认通过**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: PASS(全套;失败项按新契约修正)

- [ ] **Step 5: Commit + dev-notes**

```bash
git add -A
git commit -m "feat(chat): conversation_id contract, retire in-memory sessions"
```
并向 `dev-notes/ch02.md` 追加 Task 10 段。

---

### Task 11: 工具选型标注集 + eval(真模型)

**Files:**
- Create: `eval_data/tool_selection_samples.json`
- Create: `scripts/eval_tool_selection.py`

**Interfaces:**
- Consumes: `ToolRegistry`(Task 7)、`langchain` `bind_tools`。
- Produces: `python -m scripts.eval_tool_selection` 打印逐样例「问句 → 期望工具 → 实际选中」与准确率;**「邮费是多少」记录为 known_gap**(选对 `query_faq` 但内容查不到)。

- [ ] **Step 1: 写标注样例集**

```json
[
  {"question": "订单 1001 的物流到哪了", "expected_tool": "query_logistics"},
  {"question": "帮我查下订单1002发货了没", "expected_tool": "query_logistics"},
  {"question": "订单 1001 里买的是什么", "expected_tool": "query_order"},
  {"question": "商品 P123 多少钱", "expected_tool": "query_product"},
  {"question": "退货政策是什么", "expected_tool": "query_faq"},
  {"question": "保修多久", "expected_tool": "query_faq"},
  {"question": "我要投诉,给我转人工", "expected_tool": "create_ticket"},
  {"question": "今天天气怎么样", "expected_tool": "none"},
  {"question": "你好呀", "expected_tool": "none"},
  {"question": "邮费是多少", "expected_tool": "query_faq", "known_gap": "关键词查表查不出来,预期漏召回"}
]
```

- [ ] **Step 2: 写 eval 脚本**

```python
# scripts/eval_tool_selection.py
"""跑真模型验证工具选型。用法:python -m scripts.eval_tool_selection"""
import json
import sys
from pathlib import Path

from app.llm import get_chat_model
from app.tools.business import query_logistics, query_order, query_product
from app.tools.kb import make_kb_tools
from app.tools.ops import make_ops_tools
from app.tools.registry import ToolRegistry
from app.db.base import get_sessionmaker

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "eval_data" / "tool_selection_samples.json"


async def main() -> int:
    sf = get_sessionmaker()
    registry = ToolRegistry([
        query_order, query_product, query_logistics,
        *make_kb_tools(sf), *make_ops_tools(sf, conversation_id=0),
    ])
    model = get_chat_model().bind_tools(registry.bindable())
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    hits = 0
    for i, s in enumerate(samples, 1):
        msg = await model.ainvoke([{"role": "user", "content": s["question"]}])
        calls = getattr(msg, "tool_calls", None) or []
        picked = calls[0]["name"] if calls else "none"
        ok = picked == s["expected_tool"]
        hits += ok
        tag = "PASS" if ok else ("GAP" if s.get("known_gap") else "FAIL")
        print(f"[{tag}] #{i} {s['question']} -> 期望 {s['expected_tool']}, 实际 {picked}")
        if s.get("known_gap"):
            print(f"      已知缺口: {s['known_gap']}")
    rate = hits / len(samples)
    print(f"\n选型准确率: {hits}/{len(samples)} = {rate:.0%}")
    return 0 if hits == len(samples) - sum(1 for s in samples if s.get("known_gap")) else 1


if __name__ == "__main__":
    import anyio

    sys.exit(anyio.run(main))
```

- [ ] **Step 3: 首次运行,记录基线**

Run: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m scripts.eval_tool_selection`
Expected: 出准确率(baseline 记入 dev-notes),`邮费是多少` 能为 GAP 更好、被选成 query_faq 即算符合预期。

- [ ] **Step 4: 迭代到预期**

若把「你好呀」误判成工具、或显然该调却不调:优先修**工具 docstring**(描述使用场景),而不是改标注集。所有非 known_gap 样例通过即停。

- [ ] **Step 5: Commit + dev-notes**

```bash
git add eval_data/tool_selection_samples.json scripts/eval_tool_selection.py
git commit -m "feat(eval): tool selection samples and runner"
```
并向 `dev-notes/ch02.md` 追加 Task 11 段(含准确率与已知缺口)。

---

### Task 12: 聊天页工具徽章(Vibe Coding 例外)+ 演示与三连验收

**Files:**
- Modify: `web/index.html`
- Modify: `README.md`
- Create: `scripts/demo_tools.sh`

**说明**:本 Task 的**前端部分按用户约定走 Vibe Coding**(不套 TDD/评审),由用户描述效果、我直接改;下面只给后端可见的接口事实与验收步骤。

**Interfaces:**
- SSE 新增帧(前端消费):`event: tool  data:{"call_id","name","args","status":"running"}` / `data:{"call_id","name","status":"ok|error","summary"}`。

- [ ] **Step 1: 聊天页工具徽章(Vibe,迭代)**

页面在助手气泡上方/内部展示本轮工具轨迹小徽章:running 时 🐾 `name` 加载态,ok 时 ✅ `name`,error 时 ⚠ `name`;多工具按顺序并列。实现走 `event: tool` 帧;`done` 后徽章定稿。

- [ ] **Step 2: 演示脚本**

```bash
# scripts/demo_tools.sh
#!/usr/bin/env bash
# 验收:工具调用(物流 / FAQ / 漏召回)
set -uo pipefail
BASE="${1:-http://127.0.0.1:8000}"
FIX="$(cd "$(dirname "$0")" && pwd)/fixtures"
for f in tool_logistics tool_faq tool_miss; do
  echo "== $f =="
  curl -sN -X POST "$BASE/api/chat" -H 'Content-Type: application/json' --data-binary "@$FIX/$f.json"
  echo
done
```
配 `scripts/fixtures/tool_logistics.json`(`{"user_id":"demo","message":"订单 1001 的物流到哪了"}`)、`tool_faq.json`(`退货政策是什么`)、`tool_miss.json`(`邮费是多少`)——中文一律走 UTF-8 fixture 文件(Win 控制台编码坑)。

- [ ] **Step 3: 三连验收(真模型 + MySQL)**

```bash
./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000 &
bash scripts/demo_tools.sh http://127.0.0.1:8000
```
- ①`订单 1001 的物流到哪了` → 见 `event: tool`(query_logistics)+ 据物流结果作答
- ②`退货政策是什么` → `query_faq` 命中并作答(需 faq 表有数据,见下)
- ③`邮费是多少` → 选 query_faq 但 `FAQ 未找到…`(漏召回,**预期**)

> 验收②需先灌 FAQ 种子数据。追加 `scripts/seed_faq.py`:写 5~8 条常见问答(退货政策/发票/运费/保修/发货时效),`python -m scripts.seed_faq` 幂等插入。README 增加该步骤。

- [ ] **Step 4: Commit + dev-notes**

```bash
git add web README.md scripts
git commit -m "docs(web): tool badges, faq seed, demo scripts"
```
并向 `dev-notes/ch02.md` 追加 Task 12 段(含三条验收实测输出)。

---

## Self-Review 记录

- **Spec 覆盖**:目标/验收(→Task 9/11/12);分层骨架 `db/tools/agent`(→Task 1-8);四表 DDL 与模型(→Task 2,DDL 已建);五工具(→Task 5/6);注册/校验/超时重试(→Task 7);结果回灌(→Task 9);单轮一次往返 + 收敛不绑工具(→Task 8/9);SSE 扩展含 tool 帧(→Task 9/10/12);落库与状态(→Task 4/9);conversation_id 契约与 user_id(→Task 10);DB 为准、退役内存会话(→Task 10);配置与 .env(→Task 1);测试分工 TDD/eval/Vibe(→Task 1-11 + Task 12);已知缺口「邮费」(→Task 11/12);章节边界无 Agent 循环/RAG/Alembic(→Global Constraints)。
- **占位扫描**:无 TBD/TODO;每个代码步给出可运行实现。Task 9 含一处"实现提示"文字(裁剪组装抽内部函数),属重构指引而非占位,代码主体完整。
- **类型/签名一致性**:`ToolExecutionResult(call_id,name,args,ok,content,error,attempts)` 在 Task 7 定义、Task 8/9 消费;`ToolRegistry.execute(name,args,*,timeout,retries,call_id)` 一致;`stream_first_round(model,messages,on_delta)->(tool_calls,text)`、`execute_tool_calls(registry,tool_calls,on_status,*,timeout,retries)` 在 Task 8 定义、Task 9 消费;`to_langchain_messages(messages:list[dict])` 与 `trim_to_budget` 在 Task 9 由 `context.py` 提供、`ChatService._build_lc_messages` 使用;`build_registry(session_factory, conversation_id)` 在 Task 10 定义,经 `registry_factory` 注入;`ChatService(model, registry_factory, session_factory, system_prompt, settings)` / `stream_turn(user_id, conversation_id, user_text)` 前后一致。
- **已知执行期需复核点**:`AIMessageChunk.tool_call_chunks` 累加成 `tool_calls`(Task 8 测试即验证);`ToolMessage(content, tool_call_id=)`;aiomysql+MySQL5.7 的 ENUM/JSON 兼容(已实测连接与建表);`pytest.mark.anyio` 已可用。
