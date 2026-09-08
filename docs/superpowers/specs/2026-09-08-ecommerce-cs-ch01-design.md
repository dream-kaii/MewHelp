# ch01 电商智能客服 · 纯对话系统设计

- 日期:2026-09-08
- 章节范围:多章项目(电商智能客服系统)的 **ch01** —— 先跑通纯对话,不含工具调用/Agent/持久化/前端
- 相关文档:`dev-notes/ch01.md`(过程留痕)

## 1. 目标与验收

**目标**:做一个能换上游模型的多轮对话客服后端 + 售后诉求结构化抽取小功能。

**验收标准**
1. curl 调对话接口能看到 SSE 流式回复(逐 token)
2. 连续两轮问,第二轮能接住第一轮的上下文(同 session_id)
3. 发一段售后描述,能拿到结构化 JSON

## 2. 技术栈(定死,不自行更换)

- Python 3.12 + FastAPI + LangChain(LangChain OpenAI 集成)
- 模型接入:**应用侧统一说 OpenAI 协议**(`langchain_openai.ChatOpenAI`),`base_url / api_key / model` 全在 `.env`。
  - GPT / DeepSeek / Ollama 原生 OpenAI 协议直连
  - Claude 原生非 OpenAI 协议,经 **OpenAI 兼容网关/中转**接入(用户已确认,不走 Anthropic 专用适配分支)
- 依赖管理:pyproject.toml + `pip install -e .[dev]`(无 uv);测试用 pytest

## 3. 仓库布局

```
d:\MewHelp\
  pyproject.toml
  .env / .env.example
  app/
    main.py            # FastAPI 工厂 + /healthz + 挂路由
    config.py          # pydantic-settings 读 .env
    llm.py             # ChatOpenAI 工厂(唯一建模型的地方)
    prompts.py         # PromptTemplate:客服 System、抽取 System/User
    schemas.py         # 请求/响应模型、RequestType 枚举、AfterSalesExtract
    sessions.py        # 内存会话存储(session_id → 消息历史),带上限
    context.py         # 历史裁剪 + token 预算(纯函数)
    chat.py            # 对话链路:拼 messages → 流式 delta 生成器
    extract.py         # with_structured_output 售后字段抽取
    routers/chat.py    # POST /api/chat
    routers/extract.py # POST /api/extract
  tests/               # pytest 单测(不碰真网络)
  scripts/             # eval_extract.py、demo 脚本
  eval_data/after_sales_samples.json   # 售后抽取标注样例集
  dev-notes/ch01.md
  docs/superpowers/specs/2026-09-08-ecommerce-cs-ch01-design.md
  docs/superpowers/plans/2026-09-08-...-plan.md
```

## 4. 组件职责(单一职责、可独立测试)

| 模块 | 职责 | 依赖 | 测试方式 |
|---|---|---|---|
| `context.py` | 纯函数:输入 system+历史+当前消息+预算 → 输出发给模型的 messages | 无 | TDD 单测 |
| `sessions.py` | session_id → 历史存/取,长度上限、可清空 | 无 | TDD 单测 |
| `llm.py` | 按 config 造 ChatOpenAI(streaming 开) | config | 冒烟 |
| `prompts.py` | PromptTemplate 定义(不调模型) | langchain_core | 渲染断言 |
| `schemas.py` | Pydantic 模型与枚举 | pydantic | TDD 单测 |
| `chat.py` | 取历史→裁剪→astream→逐 token 产出 | context/sessions/llm/prompts | fake model 单测 + 集成冒烟 |
| `extract.py` | with_structured_output 抽取 | llm/prompts/schemas | 标注集 eval(替代单测) |
| routers/* | HTTP 校验与序列化 | 上面对应 service | FastAPI TestClient + fake model |

**分层规则**:router 不碰模型逻辑;service(chat/extract)不 import 对方业务;llm.py 是唯一能建模型的地方(fake model 注入点)。

## 5. 对话链路与 SSE 契约

- `POST /api/chat`,body `{session_id?: str, message: str}`。session_id 缺省由服务端生成。
- 流程:取历史 → `context.build_messages(...)` → `model.astream(messages)` → 逐 chunk 抽 content → SSE。
- 流结束后,把「user 当前消息 + assistant 全文」追加进会话存储。
- SSE(text/event-stream),事件行:

```
event: session   data: {"session_id": "..."}     # 仅新建会话时
event: delta     data: {"content": "你"}
event: done      data: {"session_id":"...","finish":true}
event: error     data: {"message": "..."}        # 上游/内部异常
```

- 多轮语义(已确认假设):**历史由服务端按 session_id 进程内存持有**,客户端只传 id。理由:接近真实客服系统、给后续会话/Langfuse 留身份锚点。上限条目防泄漏。
- 流式实现方式(待查 LangChain 当前版本确认):`chat_model.astream()` vs `astream_events`,ch01 用前者 + per-chunk 手工拼 SSE(不依赖 langserve)。

## 6. Token 预算(最简版)

- `.env` 可配:`HISTORY_BUDGET_TOKENS`(默认 2048)、模型 `max_tokens`(默认 1024)。
- 计量用**启发式估算函数** `estimate_tokens(text)`,默认中文约 1 token/2 字符;函数独立、可测、可换真 tokenizer。已确认假设:估算为近似值,ch01 不绑 tiktoken。
- 裁剪策略:`build_messages` 内,保 system + 当前轮,从最旧整轮丢弃至总估算 ≤ 预算。

## 7. 售后结构化抽取

- `POST /api/extract`,body `{text: str}`,返回 `AfterSalesExtract`。
- 实现:`ChatOpenAI.with_structured_output(AfterSalesExtract)`(依赖 endpoint 支持 tool-call JSON 输出)。
- Schema:

```python
class RequestType(str, Enum):
    RETURN_REFUND = "退货退款"   # 具体取值文案用户已确认:退货退款/仅退款/换货/维修
    REFUND_ONLY = "仅退款"
    EXCHANGE = "换货"
    REPAIR = "维修"

class AfterSalesExtract(BaseModel):
    order_no: str | None          # 缺省 null,不硬编
    request_type: RequestType | None  # 无法归类 null
    desired_solution: str | None  # 自由文本
```

- 关键 prompt:要求只从原文抽取、不编造订单号、无法判断输出 null。

## 8. 错误处理

- 上游异常/鉴权失败:对话 → SSE `error` 事件;抽取 → 502 JSON。均不向客户端泄露堆栈,服务端 logging。
- body 校验失败 → FastAPI 422。

## 9. 测试与验收映射

**TDD(纯逻辑/可注入)**:context 裁剪边界、estimate_tokens、SSE 帧序列化、会话两轮连续性、路由——用 LangChain fake/GenericFake chat model + FastAPI TestClient,不碰真网络。

**标注集替代单测(用户规则,非单测代码)**:售后抽取属"prompt+模型输出",单测测不准 → `eval_data/after_sales_samples.json` 放 ~10 条手工标注售后描述+期望 JSON,`scripts/eval_extract.py` 跑真模型算通过率,迭代 System Prompt 至 **通过率 100%**(争议样本用户拍板)。

**验收三连(真模型,用户填 .env)**:
1. `curl -N` 看 SSE 流式
2. 同 session_id 两轮,第二轮能引用第一轮内容
3. 售后描述 → 结构化 JSON

## 10. 章节边界(不做)

- 不做工具调用 / Agent 循环 / LangGraph(后续章节)
- 不做历史落库持久化(ch01 内存即可)
- 不做前端聊天页(后续 Vibe Coding 章节)
- 实现中发现与上述定死选型矛盾 → 停下来问用户,不自行换方案

## 11. 过程约束(用户要求)

- 全程 Superpowers 流程;库/框架 API 用法先查官方最新文档(Context7 MCP),不凭记忆。
- dev-notes/ch01.md 分阶段追记:brainstorm 定稿、计划评审通过、每任务完成、code review 结论、finish。每段记四样:用户关键原话 / 我的关键产出 / 用户拒绝或纠偏 / 翻车与返工。禁止收尾一次性补记。
