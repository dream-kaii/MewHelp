# ch01 电商智能客服 · 纯对话 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 跑通电商售后客服后端:多轮对话 + SSE 流式、PromptTemplate 系统提示、售后诉求 `with_structured_output` 结构化抽取,模型走统一 OpenAI 协议可换 GPT/Claude(网关)/DeepSeek/Ollama。

**Architecture:** FastAPI 提供两个端点:`POST /api/chat`(EventSourceResponse SSE 逐 token)与 `POST /api/extract`(结构化 JSON)。多轮历史由服务端按 `session_id` 进程内存持有;`app/context.py` 纯函数做历史裁剪 + token 预算;`app/llm.py` 是唯一建模型处(OpenAI 协议 ChatOpenAI),测试用 `app.state` 注入 fake model;真实结构化抽取用标注集 eval 验证(不用单测)。

**Tech Stack:** Python 3.12, FastAPI(含 `fastapi.sse`), uvicorn, langchain-openai + langchain-core, pydantic-settings, pytest/httpx(TestClient)。

**Spec:** `docs/superpowers/specs/2026-09-08-ecommerce-cs-ch01-design.md`(本计划论证自该 spec,执行者两篇同读)

## Global Constraints

- **模型接入定死**:应用侧只统一说 OpenAI 协议(`langchain_openai.ChatOpenAI`),`base_url/api_key/model` 全部来自 `.env`。GPT/DeepSeek/Ollama 原生直连;Claude 经 OpenAI 兼容网关(用户已确认,禁止为 Claude 加 Anthropic 专用分支)。
- **本章边界(不做)**:工具调用 / Agent 循环 / LangGraph / 历史落库持久化 / 前端页面。发现与定死选型矛盾 → 停下问用户,不自行换方案。
- **先查文档再写 API**:凡涉及 FastAPI / LangChain / langchain-openai / pydantic-settings 具体用法,动手前先用 Context7 查官方当前文档核对签名,禁止凭记忆。每次改动开始前,先把实际安装版本记进 dev-notes(`pip show`),API 对不上立即停下来查,不要硬写。
- **每任务结束追加留痕**:每个 Task 验收通过后,向 `dev-notes/ch01.md` 追加一段(记四样:用户关键原话 / 我的关键产出 / 用户拒绝或纠偏 / 翻车返工)。禁止收尾一次性补记。
- **TDD 替代规则**:产出是「prompt + 模型输出」(System Prompt 内容、抽取 schema、with_structured_output)的任务,不以单测断言质量,改用 `eval_data/after_sales_samples.json` 标注集 + 真实模型跑通过率验证(Task 9)。纯逻辑与可注入代码仍走 TDD。
- **回复语言**:客服与抽取均面向中文用户;模型对话 System Prompt 全中文。
- **提交**:每个 step 结束按给定命令 commit,message 以 `feat:`/`test:`/`chore:` 开头。
- **代码盘上只信这两件事的现状**:fastapi.sse、GenericFakeChatModel 的 import 路径、`model.astream` 的 chunk.content 形态、`with_structured_output` 的 method 取值 —— 任一在你安装的版本里对不上,在对应 Task 的 doc-check step 就地调整并把差异记进 dev-notes,而不是悄悄改设计。

---

### Task 1: 项目脚手架 + 配置(Settings)+ /healthz

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `app/main.py`
- Create: `tests/conftest.py`
- Create: `tests/test_healthz.py`

**Interfaces:**
- Produces: `get_settings() -> Settings`;字段: `llm_base_url: str = "http://localhost:11434/v1"`, `llm_api_key: str = ""`, `llm_model: str = "qwen2.5:7b"`, `llm_provider: str = "ollama"`, `temperature: float = 0.7`, `max_tokens: int = 1024`, `history_budget_tokens: int = 2048`, `session_max_turns: int = 30`, `session_max_count: int = 200`, `extract_method: str = "function_calling"`, `cs_shop_name: str = "MewHelp"`, `cs_staff_name: str = "小喵"`。
- Produces: `app = create_app()` 于 `app.main`;FastAPI 实例 `.state.chat_model = None`、`.state.extract_model = None`;`GET /healthz` 返回 `{"status":"ok","provider":...,"model":...}`。

- [ ] **Step 1: 文档核对(先查后写)**

Context7 `/websites/fastapi_tiangolo` 与 `/pydantic/pydantic-settings`,核对:当前 FastAPI 包结构是否存在 `fastapi.sse.EventSourceResponse`;pydantic-settings `SettingsConfigDict(env_file=".env")` 用法。记录你实际安装到的版本号到 dev-notes。

- [ ] **Step 2: 写失败测试**

```python
# tests/test_healthz.py
from fastapi.testclient import TestClient
from app.main import app

def test_healthz_ok():
    with TestClient(app) as client:
        r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "provider" in body and "model" in body
```

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    app.state.chat_model = None
    app.state.extract_model = None
    with TestClient(app) as c:
        yield c
```

- [ ] **Step 3: 运行确认失败**

Run: `python -m pytest tests/test_healthz.py -v`
Expected: FAIL(`ModuleNotFoundError: app`)

- [ ] **Step 4: 建包并实现**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "mewhelp-cs"
version = "0.1.0"
description = "ch01: e-commerce after-sales customer service (pure chat)"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "langchain-openai>=0.3",
    "langchain-core>=0.3",
    "pydantic-settings>=2.4",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27"]

[tool.setuptools.packages.find]
include = ["app*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```toml
# .env.example
# 统一 OpenAI 协议。换上游只改这三行(GPT/DeepSeek/Ollama 直连;Claude 填 OpenAI 兼容网关)
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-your-key-here
LLM_MODEL=deepseek-chat
LLM_PROVIDER=deepseek        # 仅日志/标注用途

TEMPERATURE=0.7
MAX_TOKENS=1024
HISTORY_BUDGET_TOKENS=2048
SESSION_MAX_TURNS=30
SESSION_MAX_COUNT=200
# with_structured_output 的 method:function_calling | json_mode | json_schema
EXTRACT_METHOD=function_calling

CS_SHOP_NAME=MewHelp
CS_STAFF_NAME=小喵
```

```python
# app/__init__.py
"""MewHelp 电商售后客服系统(ch01 纯对话)。"""
```

```python
# app/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全部来自 .env / 环境变量;缺省值面向本地 Ollama 与离线单测。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = ""
    llm_model: str = "qwen2.5:7b"
    llm_provider: str = "ollama"
    temperature: float = 0.7
    max_tokens: int = 1024
    history_budget_tokens: int = 2048
    session_max_turns: int = 30
    session_max_count: int = 200
    extract_method: str = "function_calling"
    cs_shop_name: str = "MewHelp"
    cs_staff_name: str = "小喵"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

```python
# app/main.py
from fastapi import FastAPI
from app.config import get_settings


def create_app() -> FastAPI:
    app = FastAPI(title="MewHelp CS ch01")
    app.state.chat_model = None
    app.state.extract_model = None

    @app.get("/healthz")
    async def healthz():
        s = get_settings()
        return {"status": "ok", "provider": s.llm_provider, "model": s.llm_model}

    return app


app = create_app()
```

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest tests/test_healthz.py -v`
Expected: PASS

- [ ] **Step 6: 安装 + 版本留痕**

```bash
python -m pip install -e ".[dev]"
python -c "import fastapi,langchain_openai,langchain_core,pydantic_settings;print('fastapi',fastapi.__version__);print('langchain_openai',langchain_openai.__version__);print('langchain_core',langchain_core.__version__)"
python -c "from fastapi.sse import EventSourceResponse; print('fastapi.sse OK')"
```

若 `fastapi.sse` 导入失败 → 升级 fastapi 到含 SSE 的版本后重跑,并把差异记进 dev-notes。

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .env.example app tests && git commit -m "feat: scaffold FastAPI app, settings, healthz"
```

- [ ] **Step 8: dev-notes 追加 Task 1 段落**

在 `dev-notes/ch01.md` 追加「Task 1 完成」段(记四样,含实际安装到的各库版本、doc-check 有无发现版本差异)。

---

### Task 2: schemas(请求模型 + 售后抽取模型 + 枚举)

**Files:**
- Create: `app/schemas.py`
- Create: `tests/test_schemas.py`

**Interfaces:**
- Produces: `ChatRequest(session_id: str | None = None, message: str)`;`ExtractRequest(text: str)`;`RequestType(str, Enum)` = {退货退款/仅退款/换货/维修};`AfterSalesExtract(order_no: str|None, request_type: RequestType|None, desired_solution: str|None)`,三字段默认 `None` 并带中文 `Field(description=...)`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_schemas.py
import pytest
from pydantic import ValidationError
from app.schemas import AfterSalesExtract, ChatRequest, RequestType


def test_chat_request_defaults_session_none():
    r = ChatRequest(message="你好")
    assert r.session_id is None
    assert r.message == "你好"


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_request_type_has_exactly_four_members():
    assert {e.value for e in RequestType} == {"退货退款", "仅退款", "换货", "维修"}


def test_extract_allows_null_fields():
    e = AfterSalesExtract()
    assert e.order_no is None and e.request_type is None and e.desired_solution is None


def test_extract_accepts_valid_request_type():
    e = AfterSalesExtract(order_no="20260908001", request_type=RequestType.RETURN_REFUND)
    assert e.request_type == RequestType.RETURN_REFUND
    assert e.order_no == "20260908001"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: FAIL(`ModuleNotFoundError: app.schemas`)

- [ ] **Step 3: 实现**

```python
# app/schemas.py
from enum import Enum

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, description="会话 id;缺省由服务端生成并回传")
    message: str = Field(min_length=1, max_length=4000)


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class RequestType(str, Enum):
    RETURN_REFUND = "退货退款"
    REFUND_ONLY = "仅退款"
    EXCHANGE = "换货"
    REPAIR = "维修"


class AfterSalesExtract(BaseModel):
    """售后诉求抽取结果。只从原文抽取,禁止编造;无法判断一律 null。"""

    order_no: str | None = Field(default=None, description="原文出现的订单号;未提及为 null")
    request_type: RequestType | None = Field(
        default=None,
        description="诉求类型,仅可取 退货退款/仅退款/换货/维修;无法归类为 null",
    )
    desired_solution: str | None = Field(default=None, description="用户期望的处理方案(自由文本);未明说为 null")
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py tests/test_schemas.py && git commit -m "feat: request/extract pydantic schemas and RequestType enum"
```

- [ ] **Step 6: dev-notes 追加 Task 2 段落**

---

### Task 3: context(历史裁剪 + token 预算,纯函数)

**Files:**
- Create: `app/context.py`
- Create: `tests/test_context.py`

**Interfaces:**
- Consumes: 无(纯模块)。
- Produces: `estimate_tokens(text: str, chars_per_token: float = 2.0) -> int`;`total_tokens(messages: list[dict]) -> int`;`trim_to_budget(messages: list[dict], budget: int) -> list[dict]`;`build_messages(system_prompt: str | None, history: list[dict], current_user_msg: str, budget: int) -> list[dict]`。
  - 消息 dict 形如 `{"role": "system"|"user"|"assistant", "content": str}`。
  - `trim_to_budget` 不变量:**首条 system(若有)与最后一条消息永不删除**;超预算时从最旧整对(`[user, assistant]`)删起;只剩 `system + 当前 user` 仍超预算时原样返回(单条过大不硬截)。
  - `estimate_tokens`:中文按约 2 字符/token 的启发式,`len(text)/chars_per_token` 向上取整,空串返回 0。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_context.py
from app.context import build_messages, estimate_tokens, trim_to_budget

SYS = {"role": "system", "content": "你是客服"}
U = lambda i: {"role": "user", "content": f"用户问题{i}，内容补长一些"}
A = lambda i: {"role": "assistant", "content": f"客服回复{i}，同样补长一些"}


def test_estimate_tokens_empty_is_zero():
    assert estimate_tokens("") == 0


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("你好") >= 1
    assert estimate_tokens("你好" * 100) > estimate_tokens("你好")


def test_trim_keeps_system_and_current_message():
    history = [U(0), A(0), U(1), A(1), U(2), A(2)]
    out = trim_to_budget([SYS] + history + [U(3)], budget=50)
    assert out[0]["role"] == "system"
    assert out[-1] == U(3)


def test_trim_drops_oldest_pair_first():
    history = [U(0), A(0), U(1), A(1)]
    out = trim_to_budget([SYS] + history + [U(2)], budget=80)
    roles = [m["role"] for m in out]
    # 系统 + 新近两轮(第1、2轮)+ 当前第3轮;最旧 user0/assistant0 被删
    assert roles[0] == "system"
    assert U(1) in out and A(1) in out and U(2) in out
    assert U(0) not in out and A(0) not in out


def test_trim_returns_everything_when_within_budget():
    msgs = [SYS, U(0), A(0), U(1)]
    assert trim_to_budget(msgs, budget=10_000) == msgs


def test_build_messages_order_system_history_current():
    history = [U(0), A(0)]
    out = build_messages("你是客服", history, "当前问题", budget=10_000)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]
    assert out[-1]["content"] == "当前问题"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_context.py -v`
Expected: FAIL(`ModuleNotFoundError: app.context`)

- [ ] **Step 3: 实现**

```python
# app/context.py
"""历史裁剪 + token 预算。纯函数,不依赖 langchain / 网络,历史一律为 list[dict]。"""

_MESSAGE_OVERHEAD = 4  # 每条消息的固定估算开销(角色/格式)


def estimate_tokens(text: str, chars_per_token: float = 2.0) -> int:
    """启发式估算 token 数。默认约 2 个字符 = 1 token(中文为主的场景)。"""
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def total_tokens(messages: list[dict]) -> int:
    return sum(_tokens_of_message(m) for m in messages)


def _tokens_of_message(msg: dict) -> int:
    return estimate_tokens(str(msg.get("content") or "")) + _MESSAGE_OVERHEAD


def trim_to_budget(messages: list[dict], budget: int) -> list[dict]:
    """超预算时从最旧整对(user,assistant)删起。

    不变量:首条 system(若有)与最后一条消息永不删除;仅剩 system+当前消息仍超预算则原样返回。
    """
    if not messages:
        return messages

    head = [messages[0]] if messages[0].get("role") == "system" else []
    body = messages[len(head):]

    while total_tokens(head + body) > budget and len(body) >= 2:
        body = body[2:]  # 删最旧一整轮
    return head + body


def build_messages(system_prompt: str | None, history: list[dict], current_user_msg: str, budget: int) -> list[dict]:
    full: list[dict] = []
    if system_prompt:
        full.append({"role": "system", "content": system_prompt})
    full.extend(history)
    full.append({"role": "user", "content": current_user_msg})
    return trim_to_budget(full, budget)
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_context.py -v`
Expected: PASS(若 `test_trim_drops_oldest_pair_first` 因预算/字数取值不当失败,调整测试文案长度使第 0 轮超出预算而第 1、2 轮合计在预算内——断言的是行为,不是具体常数)

- [ ] **Step 5: Commit**

```bash
git add app/context.py tests/test_context.py && git commit -m "feat: history trimming and token-budget pure functions"
```

- [ ] **Step 6: dev-notes 追加 Task 3 段落**

---

### Task 4: sessions(进程内存会话存储)

**Files:**
- Create: `app/sessions.py`
- Create: `tests/test_sessions.py`

**Interfaces:**
- Consumes: `app.config.get_settings()`(取 `session_max_turns` / `session_max_count`)。
- Produces: `class Session(session_id, created_at, turns: list[dict])`;`class SessionStore(max_sessions=200, max_turns=30)` 方法 `new_session() -> Session`、`get(session_id) -> Session | None`、`get_or_create(session_id: str | None) -> tuple[Session, bool]`(bool=是否新建)、`append_turn(session_id, user_msg: str, assistant_msg: str) -> None`、`clear() -> None`;模块级 `_store = SessionStore()` 与 `get_store() -> SessionStore`。
- `turns` 只存 `user/assistant` 交替消息 dict;超 `max_turns` 轮时从最旧整对删除(保持配对)。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_sessions.py
from app.sessions import SessionStore


def test_get_or_create_new_without_id():
    s, created = SessionStore().get_or_create(None)
    assert s.session_id and created


def test_get_or_create_returns_existing():
    store = SessionStore()
    sid = "abc"
    s1, created = store.get_or_create(sid)
    assert created is True
    s2, created2 = store.get_or_create(sid)
    assert created2 is False and s2.session_id == sid


def test_append_turn_orders_pairs():
    store = SessionStore()
    sid, _ = store.get_or_create("s1")
    store.append_turn(sid, "问1", "答1")
    store.append_turn(sid, "问2", "答2")
    assert [m["role"] for m in store.get(sid).turns] == ["user", "assistant", "user", "assistant"]
    assert store.get(sid).turns[0]["content"] == "问1"


def test_append_turn_caps_to_max_turns_keeping_pairs():
    store = SessionStore(max_turns=2)
    sid, _ = store.get_or_create("cap")
    for i in range(5):
        store.append_turn(sid, f"问{i}", f"答{i}")
    assert len(store.get(sid).turns) == 4  # 仅保留最近 2 轮(user+assistant)
    assert store.get(sid).turns[0] == {"role": "user", "content": "问3"}


def test_store_evicts_oldest_when_over_max():
    store = SessionStore(max_sessions=2)
    a, _ = store.get_or_create("a")
    store.get_or_create("b")
    store.get_or_create("c")  # 触发淘汰
    assert store.get("a") is None
    assert store.get("b") is not None and store.get("c") is not None
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_sessions.py -v`
Expected: FAIL(`ModuleNotFoundError: app.sessions`)

- [ ] **Step 3: 实现**

```python
# app/sessions.py
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from app.config import get_settings


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    turns: list[dict] = field(default_factory=list)  # [{"role":"user"|"assistant","content":str}, ...]


class SessionStore:
    """进程内存会话存储。LRU 淘汰最旧会话;单会话超轮数时删最旧整对。"""

    def __init__(self, max_sessions: int | None = None, max_turns: int | None = None):
        s = get_settings()
        self._max_sessions = max_sessions or s.session_max_count
        self._max_turns = max_turns or s.session_max_turns
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()

    def new_session(self) -> Session:
        return self._put(Session(session_id=uuid.uuid4().hex))

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None) -> tuple[Session, bool]:
        if session_id:
            existing = self._sessions.get(session_id)
            if existing is not None:
                self._sessions.move_to_end(session_id)
                return existing, False
            return self._put(Session(session_id=session_id)), True
        return self.new_session(), True

    def append_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        s = self._sessions[session_id]
        s.turns.append({"role": "user", "content": user_msg})
        s.turns.append({"role": "assistant", "content": assistant_msg})
        max_msgs = self._max_turns * 2
        while len(s.turns) > max_msgs:
            del s.turns[:2]  # 删最旧一整对,保持 user/assistant 配对

    def clear(self) -> None:
        self._sessions.clear()

    def _put(self, session: Session) -> Session:
        self._sessions[session.session_id] = session
        while len(self._sessions) > self._max_sessions:
            _, oldest = self._sessions.popitem(last=False)
        return session


_store = SessionStore()


def get_store() -> SessionStore:
    return _store
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_sessions.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/sessions.py tests/test_sessions.py && git commit -m "feat: in-memory session store with LRU and turn cap"
```

- [ ] **Step 6: dev-notes 追加 Task 4 段落**

---

### Task 5: prompts(PromptTemplate 化系统提示 + 抽取提示)

**Files:**
- Create: `app/prompts.py`
- Create: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `app.config.Settings`。
- Produces: `format_chat_system_prompt(settings: Settings) -> str`(全中文客服角色 + 行为约束,用 `PromptTemplate.from_template(...).format(...)`,含 `{shop_name}`/`{staff_name}` 插值);`EXTRACT_SYSTEM: str`(抽取规则常量);`format_extract_user(text: str) -> str`(把售后原文填进 PromptTemplate)。
- **质量不以单测断言**(属 prompt 数据类),本任务只验证「模板能渲染、渲染结果含关键约束词」;抽取效果是否达标由 Task 9 标注集判定。

- [ ] **Step 1: 文档核对(先查后写)**

Context7 `/langchain-ai/docs` 核对 `langchain_core.prompts.PromptTemplate` 当前 `from_template(...)` / `.format(...)` 用法与 import 路径;记版本差异到 dev-notes。

- [ ] **Step 2: 写失败测试**

```python
# tests/test_prompts.py
from app.config import Settings
from app.prompts import EXTRACT_SYSTEM, format_chat_system_prompt, format_extract_user


def test_chat_system_prompt_renders_and_constrains():
    text = format_chat_system_prompt(Settings(cs_shop_name="测试店", cs_staff_name="阿喵"))
    assert "测试店" in text and "阿喵" in text
    for key in ("客服", "不编造", "订单号"):
        assert key in text


def test_extract_user_prompt_embeds_text():
    out = format_extract_user("订单20260908001漏气想退")
    assert "20260908001" in out


def test_extract_system_lists_enum_and_null_rule():
    assert "退货退款" in EXTRACT_SYSTEM and "仅退款" in EXTRACT_SYSTEM
    assert "换货" in EXTRACT_SYSTEM and "维修" in EXTRACT_SYSTEM
    assert "null" in EXTRACT_SYSTEM
```

- [ ] **Step 3: 运行确认失败**

Run: `python -m pytest tests/test_prompts.py -v`
Expected: FAIL(`ModuleNotFoundError: app.prompts`)

- [ ] **Step 4: 实现**

```python
# app/prompts.py
from langchain_core.prompts import PromptTemplate

from app.config import Settings

CHAT_SYSTEM_TEMPLATE = PromptTemplate.from_template(
    """你是{shop_name}的售后客服,代号{staff_name}。
角色:亲切、简洁,只就用户描述的问题作答;不编造订单、物流或赔付信息,不确定就明说并请用户补充。
行为约束:
1. 用户问题涉及具体订单时,先礼貌索取订单号再往下处理。
2. 需要时,引导用户明确诉求类型(退货退款 / 仅退款 / 换货 / 维修)与期望方案。
3. 不承诺超出你能力范围的补偿或时限。
4. 全程用中文回答。"""
)


def format_chat_system_prompt(settings: Settings) -> str:
    return CHAT_SYSTEM_TEMPLATE.format(shop_name=settings.cs_shop_name, staff_name=settings.cs_staff_name)


EXTRACT_SYSTEM = (
    "你是一个售后诉求信息抽取器。只从用户原文中抽取字段,禁止推断或编造。\n"
    "规则:\n"
    "- request_type 只能取以下四类之一:退货退款 / 仅退款 / 换货 / 维修;无法归类时输出 null。\n"
    "- order_no 必须是原文中明确出现过的订单号;未出现则输出 null。\n"
    "- desired_solution 是用户期望的处理方式(自由文本,如“上门取件退货”“补偿优惠券”);原文未明说则输出 null。\n"
    "严格输出 JSON,不要输出任何解释。"
)

EXTRACT_USER_TEMPLATE = PromptTemplate.from_template("请抽取以下售后描述:\n{text}")


def format_extract_user(text: str) -> str:
    return EXTRACT_USER_TEMPLATE.format(text=text)
```

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest tests/test_prompts.py -v`
Expected: PASS(若 `Settings(cs_shop_name=...)` 直接实例化触发其它必填缺失报错,改用 `get_settings().model_copy(update={...})`,并同步修正测试)

- [ ] **Step 6: Commit**

```bash
git add app/prompts.py tests/test_prompts.py && git commit -m "feat: PromptTemplate-based CS system prompt and extract prompts"
```

- [ ] **Step 7: dev-notes 追加 Task 5 段落**

---

### Task 6: llm 工厂 + chat 服务(SSE 生成器)

**Files:**
- Create: `app/llm.py`
- Create: `app/chat.py`
- Create: `tests/test_chat_service.py`

**Interfaces:**
- Consumes: `app.config.get_settings()`、`app.context.build_messages`、`app.sessions.SessionStore`。
- Produces:
  - `get_chat_model() -> ChatOpenAI`(读 settings 构造,`streaming=True`,超时 60s;`api_key` 为空时给占位 `"not-needed"`)。
  - `to_langchain_messages(messages: list[dict]) -> list[BaseMessage]`(system→SystemMessage,user→HumanMessage,assistant→AIMessage,未知 role 兜底 HumanMessage)。
  - `text_of_chunk(chunk) -> str`(兼容 `chunk.content` 为 str 或 OpenAI content block list)。
  - `class ChatService(model, store, system_prompt, settings)`,方法 `async stream_turn(session_id: str | None, user_text: str) -> AsyncIterator[fastapi.sse.ServerSentEvent]`:
    - 新建会话先发 `ServerSentEvent(event="session", data={"session_id": ...})`;否则不发。
    - `build_messages(system_prompt, session.turns, user_text, settings.history_budget_tokens)` → `to_langchain_messages` → `model.astream(...)`。
    - 每有非空文本发 `ServerSentEvent(event="delta", data={"content": text})`,`await asyncio.sleep(0)` 让出事件循环。
    - 模型异常:记日志,发 `ServerSentEvent(event="error", data={"message": "抱歉,服务暂时不可用,请稍后重试"})` 并 return,**不追加历史**。
    - 正常结束后 `store.append_turn(...)`(助手全文拼接),再发 `ServerSentEvent(event="done", data={"finish": True})`。

- [ ] **Step 1: 文档核对(先查后写)**

Context7 `/langchain-ai/docs`,核对当前版本下:① `langchain_openai.ChatOpenAI` 构造参数名(`base_url/api_key/model/streaming/timeout/max_tokens/temperature`);② `model.astream(messages)` 仍可用及其产出 `chunk.content` 的形态(str 或 list);③ `GenericFakeChatModel` import 路径 `langchain_core.language_models.fake_chat_models`。任一与下方案例不一致,以官方为准就地修正并记入 dev-notes。

- [ ] **Step 2: 写失败测试**

```python
# tests/test_chat_service.py
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from app.chat import ChatService, text_of_chunk
from app.config import Settings
from app.sessions import SessionStore

SETTINGS = Settings(history_budget_tokens=2048, session_max_turns=30, session_max_count=50)


class RecordingChatModel(GenericFakeChatModel):
    """在 astream 前记录收到的消息列表,用于断言上下文是否传入模型。"""

    def __init__(self, responses):
        super().__init__(messages=iter(responses))
        self.seen_inputs: list[list] = []

    async def astream(self, messages, **kwargs):
        self.seen_inputs.append(list(messages))
        async for chunk in super().astream(messages, **kwargs):
            yield chunk


class FailingChatModel(GenericFakeChatModel):
    def __init__(self):
        super().__init__(messages=iter(["x"]))

    async def astream(self, messages, **kwargs):
        raise RuntimeError("upstream down")
        yield  # pragma: no cover


async def _collect(agen):
    return [e async for e in agen]


async def test_text_of_chunk_handles_str_and_blocks():
    class Chunk:
        def __init__(self, content):
            self.content = content

    assert text_of_chunk(Chunk("你好")) == "你好"
    assert text_of_chunk(Chunk([{"type": "text", "text": "你"}, {"type": "text", "text": "好"}])) == "你好"
    assert text_of_chunk(Chunk(None)) == ""


async def test_new_session_emits_session_delta_done_and_appends():
    model = RecordingChatModel(["我是客服,已收到。"])
    store = SessionStore(max_turns=10, max_sessions=10)
    svc = ChatService(model=model, store=store, system_prompt="你是客服", settings=SETTINGS)
    events = await _collect(svc.stream_turn(None, "你好"))
    assert [e.event for e in events] == ["session", "delta", "done"]
    sid = events[0].data["session_id"]
    assert "已收到" in "".join(e.data.get("content", "") for e in events if e.event == "delta")
    turns = store.get(sid).turns
    assert turns == [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "我是客服,已收到。"}]


async def test_second_turn_receives_history_in_messages():
    model = RecordingChatModel(["第一轮回复", "第二轮回复"])
    store = SessionStore(max_turns=10, max_sessions=10)
    svc = ChatService(model=model, store=store, system_prompt="你是客服", settings=SETTINGS)
    sid, _ = store.get_or_create("sess-1")
    await _collect(svc.stream_turn("sess-1", "第一问"))
    await _collect(svc.stream_turn("sess-1", "第二问"))
    inputs = model.seen_inputs
    assert len(inputs) == 2
    roles = [type(m).__name__ for m in inputs[1]]
    assert roles == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]
    contents = [getattr(m, "content", "") for m in inputs[1]]
    assert "第一问" in contents and "第一轮回复" in contents and "第二问" in contents


async def test_model_error_emits_error_and_no_append():
    model = FailingChatModel()
    store = SessionStore(max_turns=10, max_sessions=10)
    svc = ChatService(model=model, store=store, system_prompt="你是客服", settings=SETTINGS)
    events = await _collect(svc.stream_turn("bad", "hi"))
    assert [e.event for e in events] == ["error"]
    assert store.get("bad").turns == []
```

对 `async` 测试加 `pytestmark = pytest.mark.anyio`(需在文件顶部 `import pytest` 并安装 anyio——已随 starlette 依赖带入;若缺,`pip install anyio`)。`GenericFakeChatModel` 的 `.event`/`.data` 属性见 Step 4 说明。

- [ ] **Step 3: 运行确认失败**

Run: `python -m pytest tests/test_chat_service.py -v`
Expected: FAIL(`ModuleNotFoundError: app.chat` 或 `app.llm`;随实现缺失逐步转绿)

- [ ] **Step 4: 实现**

```python
# app/llm.py
from langchain_openai import ChatOpenAI

from app.config import get_settings


def get_chat_model() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key or "not-needed",  # 本地 Ollama 等不需要真实 key 的上游
        model=s.llm_model,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        streaming=True,
        timeout=60,
    )
```

```python
# app/chat.py
import asyncio
import logging

from fastapi.sse import ServerSentEvent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import Settings
from app.context import build_messages
from app.sessions import SessionStore

logger = logging.getLogger("mewhelp.chat")

_ROLE_TO_CLS = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


def to_langchain_messages(messages: list[dict]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in messages:
        cls = _ROLE_TO_CLS.get(m.get("role"), HumanMessage)
        out.append(cls(m.get("content") or ""))
    return out


def text_of_chunk(chunk) -> str:
    """从 AIMessageChunk 提取文本增量,兼容 content 为 str 或 OpenAI content-block list。"""
    content = getattr(chunk, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "".join(parts)
    return str(content)


class ChatService:
    def __init__(self, model, store: SessionStore, system_prompt: str, settings: Settings):
        self._model = model
        self._store = store
        self._system_prompt = system_prompt
        self._settings = settings

    async def stream_turn(self, session_id: str | None, user_text: str):
        session, created = self._store.get_or_create(session_id)
        if created:
            yield ServerSentEvent(event="session", data={"session_id": session.session_id})

        messages = build_messages(
            self._system_prompt,
            session.turns,
            user_text,
            self._settings.history_budget_tokens,
        )
        collected: list[str] = []
        try:
            async for chunk in self._model.astream(to_langchain_messages(messages)):
                text = text_of_chunk(chunk)
                if not text:
                    continue
                collected.append(text)
                yield ServerSentEvent(event="delta", data={"content": text})
                await asyncio.sleep(0)  # 让出事件循环,便于取消
        except Exception:
            logger.exception("chat stream error session=%s", session.session_id)
            yield ServerSentEvent(event="error", data={"message": "抱歉,服务暂时不可用,请稍后重试"})
            return

        if collected:
            self._store.append_turn(session.session_id, user_text, "".join(collected))
        yield ServerSentEvent(event="done", data={"finish": True})
```

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest tests/test_chat_service.py -v`
Expected: PASS(若 `ServerSentEvent` 的属性名不是 `.event/.data` 而是 `.event_type` 之类,以安装版本的 API 为准,统一在测试与实现里修正并记入 dev-notes)

- [ ] **Step 6: Commit**

```bash
git add app/llm.py app/chat.py tests/test_chat_service.py && git commit -m "feat: chat service streaming SSE generator with fake-model tests"
```

- [ ] **Step 7: dev-notes 追加 Task 6 段落**

---

### Task 7: /api/chat SSE 路由 + 装配 main

**Files:**
- Create: `app/routers/__init__.py`
- Create: `app/routers/chat.py`
- Modify: `app/main.py`
- Create: `tests/test_chat_route.py`

**Interfaces:**
- Consumes: Task 6 全部;`app.prompts.format_chat_system_prompt`;`app.sessions.get_store`。
- Produces: `router = APIRouter()`;`resolve_chat_service(request: Request) -> ChatService`(优先 `request.app.state.chat_model`,否则 `get_chat_model()`);`POST /api/chat` body `ChatRequest` → `EventSourceResponse(service.stream_turn(...), headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})`。`app.main` include `chat.router`。

- [ ] **Step 1: 文档核对(先查后写)**

Context7 `/websites/fastapi_tiangolo` 核对 `fastapi.sse.EventSourceResponse` 用法与路由中如何返回(见 spec 第 5 节);确认 headers 传法。记版本到 dev-notes。

- [ ] **Step 2: 写失败测试**

```python
# tests/test_chat_route.py
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from app.main import app
from app.sessions import get_store


def _post(client, payload):
    return client.post("/api/chat", json=payload)


def test_chat_route_streams_sse_and_returns_session(client):
    app.state.chat_model = GenericFakeChatModel(messages=iter(["喵,我在。"]))
    r = _post(client, {"message": "在吗"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "event: session" in r.text
    assert "event: delta" in r.text and "喵,我在。" in r.text
    assert "event: done" in r.text


def test_chat_route_two_turns_share_session_context(client):
    sid_box = {}

    app.state.chat_model = GenericFakeChatModel(messages=iter(["第一轮:收到订单查询。"]))
    r1 = _post(client, {"session_id": "demo-1", "message": "我的订单怎么了"})
    assert r1.status_code == 200 and "第一轮" in r1.text

    # 第二轮换一个 fake 输出,验证会话仍挂同一 session_id 且历史已累积
    app.state.chat_model = GenericFakeChatModel(messages=iter(["第二轮:已结合上文回复你。"]))
    r2 = _post(client, {"session_id": "demo-1", "message": "那能退货吗"})
    assert r2.status_code == 200 and "第二轮" in r2.text

    turns = get_store().get("demo-1").turns
    assert [m["role"] for m in turns] == ["user", "assistant", "user", "assistant"]
    assert turns[0]["content"] == "我的订单怎么了"
    assert turns[2]["content"] == "那能退货吗"


def test_chat_route_model_error_streams_error_event(client):
    class Boom(GenericFakeChatModel):
        def __init__(self):
            super().__init__(messages=iter(["x"]))

        async def astream(self, messages, **kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    app.state.chat_model = Boom()
    r = _post(client, {"message": "hi"})
    assert r.status_code == 200
    assert "event: error" in r.text
    assert "event: done" not in r.text
```

（用 `client` fixture 隔离;测试文件顶部 `from app.sessions import get_store`,并在两个用例结束后视需要清理 `get_store()` 中手动创建的 session,避免互相污染:每个用例用独立 session_id 即可。）

- [ ] **Step 3: 运行确认失败**

Run: `python -m pytest tests/test_chat_route.py -v`
Expected: FAIL(`404` 或 import 错误)

- [ ] **Step 4: 实现**

```python
# app/routers/__init__.py
"""API 路由包。"""
```

```python
# app/routers/chat.py
from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse

from app.chat import ChatService
from app.config import get_settings
from app.llm import get_chat_model
from app.prompts import format_chat_system_prompt
from app.schemas import ChatRequest
from app.sessions import get_store

router = APIRouter()


def resolve_chat_service(request: Request) -> ChatService:
    model = getattr(request.app.state, "chat_model", None)
    if model is None:
        model = get_chat_model()
    s = get_settings()
    return ChatService(
        model=model,
        store=get_store(),
        system_prompt=format_chat_system_prompt(s),
        settings=s,
    )


@router.post("/api/chat")
async def chat(body: ChatRequest, service: ChatService = Depends(resolve_chat_service)):
    return EventSourceResponse(
        service.stream_turn(body.session_id, body.message),
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

`app/main.py` 顶部改为:

```python
from fastapi import FastAPI
from app.config import get_settings
from app.routers import chat as chat_router
from app.routers import extract as extract_router  # Task 8 启用;先注释避免 import 失败

app = FastAPI(title="MewHelp CS ch01")
app.state.chat_model = None
app.state.extract_model = None


@app.get("/healthz")
async def healthz():
    s = get_settings()
    return {"status": "ok", "provider": s.llm_provider, "model": s.llm_model}


app.include_router(chat_router.router)
# app.include_router(extract_router.router)  # Task 8 取消注释
```

(将原 `create_app()` 结构改为模块级 `app` 直出更利于 TestClient 与 state 注入;若你偏好保留工厂,测试改用 `create_app()` 实例注入亦可,但全仓库统一一种。)

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest tests/test_healthz.py tests/test_chat_service.py tests/test_chat_route.py -v`
Expected: PASS(若 SSE 事件行在 httpx 聚合 body 里含空行分隔导致 `"event: delta" in r.text` 断言失败,改用 `assert "event: delta" in r.text.replace("\n\n", "\n")` 并对两个 event 断言都做同样规整,记入 dev-notes)

- [ ] **Step 6: Commit**

```bash
git add app/routers app/main.py tests/test_chat_route.py && git commit -m "feat: SSE /api/chat route wired into app"
```

- [ ] **Step 7: dev-notes 追加 Task 7 段落**

---

### Task 8: 售后结构化抽取服务 + /api/extract 路由

**Files:**
- Create: `app/extract.py`
- Create: `app/routers/extract.py`
- Modify: `app/main.py`(取消注释 include extract)
- Create: `tests/test_extract_route.py`

**Interfaces:**
- Consumes: `app.schemas.AfterSalesExtract`、`app.prompts.EXTRACT_SYSTEM/format_extract_user`、`get_chat_model()`、`get_settings()`。
- Produces: `class ExtractService(model, method: str | None = None)` 方法 `async extract(text: str) -> AfterSalesExtract`:内部 `self._structured = model.with_structured_output(AfterSalesExtract, method=method or settings.extract_method)`,再 `ainvoke([SystemMessage(EXTRACT_SYSTEM), HumanMessage(format_extract_user(text))])`。`router` + `resolve_extract_service(request) -> ExtractService`(优先 state.extract_model);`POST /api/extract`,body `ExtractRequest` → `AfterSalesExtract`;服务异常 → `HTTPException(502)`。
- **本任务不验证抽取质量**;质量由 Task 9 标注集跑真模型判定。单测只覆盖路由形状与错误映射。

- [ ] **Step 1: 文档核对(先查后写)**

Context7 `/langchain-ai/docs` 核对当前 `ChatOpenAI.with_structured_output(schema, *, method=..., ...)` 签名与 method 合法取值(`function_calling`/`json_mode`/`json_schema`)、返回 pydantic 对象;确认在你安装的 langchain-openai 版本成立。若有出入,以官方为准改 `extract.py` 并把差异记入 dev-notes。

- [ ] **Step 2: 写失败测试**

```python
# tests/test_extract_route.py
import pytest
from fastapi import HTTPException

from app.main import app
from app.routers.extract import resolve_extract_service
from app.schemas import AfterSalesExtract, RequestType


class _StubOk:
    async def extract(self, text: str) -> AfterSalesExtract:
        return AfterSalesExtract(
            order_no="20260908001",
            request_type=RequestType.RETURN_REFUND,
            desired_solution="上门取件退货",
        )


class _StubBoom:
    async def extract(self, text: str):
        raise RuntimeError("upstream boom")


def test_extract_route_returns_shaped_json(client):
    app.dependency_overrides[resolve_extract_service] = lambda: _StubOk()
    try:
        r = client.post("/api/extract", json={"text": "订单20260908001漏气想退货上门取件"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200
    assert r.json() == {
        "order_no": "20260908001",
        "request_type": "退货退款",
        "desired_solution": "上门取件退货",
    }


def test_extract_route_service_error_maps_to_502(client):
    app.dependency_overrides[resolve_extract_service] = lambda: _StubBoom()
    try:
        r = client.post("/api/extract", json={"text": "我要退货"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 502
```

- [ ] **Step 3: 运行确认失败**

Run: `python -m pytest tests/test_extract_route.py -v`
Expected: FAIL(`404` / import 错误)

- [ ] **Step 4: 实现**

```python
# app/extract.py
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.prompts import EXTRACT_SYSTEM, format_extract_user
from app.schemas import AfterSalesExtract


class ExtractService:
    def __init__(self, model, method: str | None = None):
        method = method or get_settings().extract_method
        self._structured = model.with_structured_output(AfterSalesExtract, method=method)

    async def extract(self, text: str) -> AfterSalesExtract:
        messages = [SystemMessage(EXTRACT_SYSTEM), HumanMessage(format_extract_user(text))]
        return await self._structured.ainvoke(messages)
```

```python
# app/routers/extract.py
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.extract import ExtractService
from app.llm import get_chat_model
from app.schemas import AfterSalesExtract, ExtractRequest

logger = logging.getLogger("mewhelp.extract")

router = APIRouter()


def resolve_extract_service(request: Request) -> ExtractService:
    model = getattr(request.app.state, "extract_model", None)
    if model is None:
        model = get_chat_model()
    return ExtractService(model=model)


@router.post("/api/extract")
async def extract(body: ExtractRequest, service: ExtractService = Depends(resolve_extract_service)) -> AfterSalesExtract:
    try:
        return await service.extract(body.text)
    except Exception:
        logger.exception("extract failed")
        raise HTTPException(status_code=502, detail="抽取服务暂时不可用")
```

`app/main.py` 取消注释 `from app.routers import extract as extract_router` 与 `app.include_router(extract_router.router)`。

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest tests/test_extract_route.py tests/test_healthz.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/extract.py app/routers/extract.py app/main.py tests/test_extract_route.py && git commit -m "feat: structured after-sales extraction endpoint"
```

- [ ] **Step 7: dev-notes 追加 Task 8 段落**

---

### Task 9: 售后抽取标注集 + eval(真实模型,达标验证)

**Files:**
- Create: `eval_data/after_sales_samples.json`
- Create: `scripts/__init__.py`
- Create: `scripts/eval_extract.py`
- Modify: `pyproject.toml`(把 `scripts` 加进可执行区或注明用 `python -m` 运行)

**Interfaces:**
- Consumes: `app.extract.ExtractService`、真实模型(需 `.env` 已填可用 key)。
- Produces: `after_sales_samples.json`(约 10 条标注样例);`python -m scripts.eval_extract`(逐条打印「原文 → 抽取 → 期望 → 命中字段」,末行打印通过率,非 100% 时 exit code 1)。

**判定规则:** request_type 与 order_no 要求**精确命中**;desired_solution 为自由文本,样例里给「关键词表」,抽取结果包含任一关键词即算中;`null` 与 `null` 精确匹配。通过率 = 全字段命中样例数 / 样例总数,迭代 prompt 至 **100%**(争议样本提请用户拍板)。

- [ ] **Step 1: 写标注样例集**

```json
// eval_data/after_sales_samples.json
[
  {
    "text": "你好,我在你家买的猫粮订单20260901001,拆开发现漏气,想退货退款。",
    "expected": {
      "order_no": "20260901001",
      "request_type": "退货退款",
      "desired_solution_keywords": ["退货", "上门取件"]
    }
  },
  {
    "text": "收到的逗猫棒少了一根,订单号是 20260902003,请补发。",
    "expected": {
      "order_no": "20260902003",
      "request_type": "null",
      "desired_solution_keywords": ["补发"]
    }
  },
  {
    "text": "猫窝买来两天就塌了,质量太差,我不想换了直接退钱。订单 20260903007",
    "expected": {
      "order_no": "20260903007",
      "request_type": "仅退款",
      "desired_solution_keywords": ["退钱", "仅退款"]
    }
  },
  {
    "text": "这个自动喂食器不出粮,售后说可以给我换个新的。订单号20260904011。",
    "expected": {
      "order_no": "20260904011",
      "request_type": "换货",
      "desired_solution_keywords": ["换"]
    }
  },
  {
    "text": "跑步机买了半年电机异响,订单20260905002,能维修吗",
    "expected": {
      "order_no": "20260905002",
      "request_type": "维修",
      "desired_solution_keywords": ["维修", "修"]
    }
  },
  {
    "text": "你们这客服电话打不通,我要投诉。",
    "expected": {
      "order_no": "null",
      "request_type": "null",
      "desired_solution_keywords": ["投诉"]
    }
  },
  {
    "text": "退货运费谁出?",
    "expected": {
      "order_no": "null",
      "request_type": "null",
      "desired_solution_keywords": []
    }
  },
  {
    "text": "订单 20260906021 的猫抓板到了但破损,请上门取件给我换个完好的。",
    "expected": {
      "order_no": "20260906021",
      "request_type": "换货",
      "desired_solution_keywords": ["换", "上门取件"]
    }
  }
]
```

- [ ] **Step 2: 写 eval 脚本**

```python
# scripts/eval_extract.py
"""跑真实模型验证售后抽取,直至 100% 通过。用法:python -m scripts.eval_extract"""
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.extract import ExtractService
from app.llm import get_chat_model

SAMPLES = Path(__file__).resolve().parents[1] / "eval_data" / "after_sales_samples.json"
KEYWORDS_FIELD = "desired_solution_keywords"


def _norm(v):
    return (v or "").strip()


def check_one(actual: dict, expected: dict) -> list[str]:
    """返回未命中的字段名列表(空 = 全命中)。"""
    misses = []
    for field in ("order_no", "request_type"):
        exp = _norm(expected.get(field)).lower()
        got = _norm(actual.get(field)).lower()
        if exp == "null":
            if got:
                misses.append(field)
        elif got != exp:
            misses.append(field)
    kw = expected.get(KEYWORDS_FIELD) or []
    sol = _norm(actual.get("desired_solution"))
    if kw and not any(k in sol for k in kw):
        misses.append("desired_solution")
    elif not kw and not sol:
        misses.append("desired_solution")
    return misses


async def main() -> int:
    settings = get_settings()
    if not settings.llm_api_key and settings.llm_provider not in ("ollama",):
        print("请先在 .env 填好 LLM_API_KEY 再跑 eval", file=sys.stderr)
        return 2

    service = ExtractService(get_chat_model())
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    passed = 0
    for i, s in enumerate(samples, 1):
        result = await service.extract(s["text"])
        data = result.model_dump()
        misses = check_one(data, s["expected"])
        ok = not misses
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] #{i} 原文:{s['text'][:30]}")
        print(f"    got  : order_no={data['order_no']} request_type={data['request_type']} desired_solution={data['desired_solution']}")
        print(f"    exp  : order_no={s['expected'].get('order_no')} request_type={s['expected'].get('request_type')} kw={s['expected'].get(KEYWORDS_FIELD)}")
        if misses:
            print(f"    miss : {misses}")
    rate = passed / len(samples)
    print(f"\n通过率: {passed}/{len(samples)} = {rate:.0%}")
    return 0 if rate == 1.0 else 1


if __name__ == "__main__":
    import anyio

    sys.exit(anyio.run(main))
```

- [ ] **Step 3: 跑一次,记录基线**

Run: `python -m scripts.eval_extract`
Expected: 能出结果(无论通过率多少)。把首跑结果与 miss 样例记进 dev-notes 作为基线。

- [ ] **Step 4: 迭代 prompt 到 100%**

改 `app/prompts.py` 的 `EXTRACT_SYSTEM`(或补样例规则),循环跑直到输出 `通过率: 100%`。
- 规则:所有样例通过即停;不要为了过样例把规则写死成样例匹配。
- 若真实模型的 tool-call 行为与期望不符(json 解析失败 / request_type 超出枚举),优先查官方文档调整 method(`EXTRACT_METHOD`)而非改 schema;矛盾处停下问用户。
- 某条样例语义有歧义、模型与人判断不同 → 把双方判断摆出来,请用户拍板后改期望值或 prompt。

- [ ] **Step 5: Commit(仅当 100%)**

```bash
git add eval_data/after_sales_samples.json scripts pyproject.toml && git commit -m "feat: after-sales extraction eval set and runner (100% pass)"
```

- [ ] **Step 6: dev-notes 追加 Task 9 段落(含基线→100% 的迭代记录、prompt 改动、争议样例处理)**

---

### Task 10: README + 演示脚本 + 三连验收

**Files:**
- Create: `README.md`
- Create: `scripts/demo_chat.sh`
- Create: `scripts/demo_extract.sh`

**Interfaces:**
- Consumes: 完整应用(需真实 `.env`)。
- Produces: 演示与验收命令文档;三连验收通过后向 dev-notes 记 finish 前汇总。

- [ ] **Step 1: README 快速开始**

写 `README.md`:依赖安装、`.env` 配置说明(附 OpenAI 协议四家示例:OpenAI/DeepSeek/Ollama/Claude 网关)、启动命令、三连验收命令与预期。

- [ ] **Step 2: 演示脚本**

```bash
# scripts/demo_chat.sh
#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"
echo "== 验收1:流式回复(应逐 token 出现)=="
curl -sN -X POST "$BASE/api/chat" \
  -H 'Content-Type: application/json' \
  -d '{"message":"你好,我的订单一直没发货"}' | tee /tmp/chat1.sse
SID=$(grep -o '"session_id":"[0-9a-f]*"' /tmp/chat1.sse | head -1 | sed 's/.*:"//;s/"//')
echo; echo "== 验收2:同一 session 第二轮(应接住第一轮上下文,含 session_id=$SID)=="
curl -sN -X POST "$BASE/api/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"message\":\"那能帮我查一下大概多久能到吗?\"}"
```

```bash
# scripts/demo_extract.sh
#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"
echo "== 验收3:售后描述 -> 结构化 JSON =="
curl -s -X POST "$BASE/api/extract" \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好,我在你家买的猫粮订单20260901001拆开漏气了,想退货退款,能上门取件吗"}'
```

- [ ] **Step 3: 起服务,跑三连验收(需真实 .env)**

```bash
python -m uvicorn app.main:app --port 8000 &
bash scripts/demo_chat.sh http://127.0.0.1:8000
bash scripts/demo_extract.sh http://127.0.0.1:8000
```
- 验收1:能看到多行 `event: delta` + 文本逐步出现。
- 验收2:第二轮回复与上下文相关(可让模型“复述你第一轮问的订单号”来显式验证)。
- 验收3:返回含 `order_no/request_type/desired_solution` 的 JSON。

- [ ] **Step 4: 全部通过后 Commit**

```bash
git add README.md scripts && git commit -m "docs: quickstart and acceptance demo scripts"
```

- [ ] **Step 5: 向用户汇报验收结果并确认 ch01 finish**

汇报内容:三连验收实际输出、`python -m scripts.eval_extract` 通过率、单测汇总(`pytest` 全绿)、`dev-notes/ch01.md` 路径。待用户确认后追加 finish 段落。

---

## Self-Review 记录

- **Spec 覆盖**:目标/验收(→Task 10);SSE 契约(→Task 6/7);PromptTemplate 客服系统提示(→Task 5);with_structured_output 抽取 + 枚举 + null 兜底(→Task 2/8/9);token 预算裁剪(→Task 3/4);OpenAI 协议统一 + .env(→Task 1/6);TDD/标注集替代规则(→各 Task + Task 9);dev-notes 留痕(→Global Constraints + 各 Task 末步);章节边界(→Global Constraints)。
- **占位扫描**:无 TBD/TODO 实现项;每步含可运行代码。
- **类型/签名一致性**:`ServerSentEvent(event=, data=)` 在 Task 6 产生、Task 7 经 `EventSourceResponse` 消费;`build_messages`/`trim_to_budget`/`estimate_tokens` 签名在 Task 3 定义、Task 6 消费;`get_or_create(session_id)->(Session,bool)` 在 Task 4 定义、Task 6 消费;`with_structured_output(AfterSalesExtract, method=...)` 在 Task 8 定义、Task 9 复用;各 Task Produces 块与消费方一致。
- **已知执行期必须复核点(版本敏感)**:`fastapi.sse` 存在性(Task 1)、GenericFakeChatModel import 路径与 `astream` 行为(Task 6)、`ServerSentEvent` 属性名 `.event/.data`(Task 6 Step 5 提示)、`with_structured_output` method 取值与解析(Task 8)、`chunk.content` 形态(Task 6)。任一对不上 → 先查 Context7 官方文档再改,差异记 dev-notes,不停下自行换设计。
