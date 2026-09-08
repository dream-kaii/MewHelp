# MewHelp CS(ch01:电商智能客服 · 纯对话)

多章项目「电商智能客服系统」第一章:**纯对话跑通**。多轮对话 + SSE 流式、PromptTemplate 客服系统提示、售后诉求 `with_structured_output` 结构化抽取。模型走**统一 OpenAI 协议**,GPT / Claude(需 OpenAI 兼容网关)/ DeepSeek / Ollama 换着接,只改 `.env`。

> 本章明确不做:工具调用、Agent 循环、LangGraph、历史落库、前端(后续章节)。详见
> `docs/superpowers/specs/2026-09-08-ecommerce-cs-ch01-design.md` 与
> `docs/superpowers/plans/2026-09-08-ecommerce-cs-ch01-plan.md`。

## 环境

- Python ≥ 3.12(开发机 3.12.3)
- 依赖:`python -m pip install -e ".[dev]"`

## 配置 `.env`

```bash
cp .env.example .env   # 然后填 key
```

统一 OpenAI 协议,四家示例:

| 上游 | LLM_BASE_URL | LLM_MODEL | LLM_API_KEY |
|---|---|---|---|
| DeepSeek | https://api.deepseek.com/v1 | deepseek-chat | sk-… |
| OpenAI | https://api.openai.com/v1 | gpt-4o | sk-… |
| Ollama(本地) | http://localhost:11434/v1 | qwen2.5:7b | 随意(留空也行) |
| Claude(经网关) | 你的 OpenAI 兼容网关地址 | 网关约定模型名 | 网关 key |

`EXTRACT_METHOD`:`function_calling`(默认)/ `json_mode` / `json_schema`,取决于上游对 tool-call 结构化输出的支持。
`HISTORY_BUDGET_TOKENS` 控制多轮历史裁剪预算(启发式约 2 字符/token)。

## 启动

```bash
python -m uvicorn app.main:app --port 8000
```

浏览器打开 <http://127.0.0.1:8000> 即客服聊天页(可爱暖色 + 小猫头像,SSE 逐字渲染、多轮续接)。API 同源:
- `POST /api/chat`(SSE)
- `POST /api/extract`
- `GET /healthz`

健康检查:`curl http://127.0.0.1:8000/healthz`

## 三连验收

```bash
bash scripts/demo_chat.sh http://127.0.0.1:8000    # ①SSE 流式 ②同 session 第二轮接上下文
bash scripts/demo_extract.sh http://127.0.0.1:8000 # ③售后描述 -> 结构化 JSON
```

- ① 应看到多行 `event: delta` 与文本逐步出现,结尾 `event: done`。
- ② 第二轮带 `session_id`,回复能引用第一轮内容。
- ③ 返回 `order_no / request_type / desired_solution` 三字段 JSON。

直接 curl 单发:

```bash
curl -sN -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"你好"}'

curl -s -X POST http://127.0.0.1:8000/api/extract -H 'Content-Type: application/json' \
  -d '{"text":"猫粮订单20260901001漏气,想退货退款,上门取件"}'
```

## 售后抽取标注验证(Task 9)

```bash
python -m scripts.eval_extract   # 跑 8 条标注样例,要求 100% 通过(需 .env 真实 key)
```

## 测试

```bash
python -m pytest -q              # 离线单测(不碰真网络)
```

## 目录速览

```
app/
  main.py           FastAPI 入口 + /healthz + 路由装配
  config.py         Settings(.env → pydantic-settings)
  llm.py            ChatOpenAI 工厂(唯一建模型处)
  prompts.py        PromptTemplate 客服系统提示 / 抽取提示
  schemas.py        请求模型、RequestType 枚举、AfterSalesExtract
  context.py        历史裁剪 + token 预算(纯函数)
  sessions.py       进程内存会话存储(LRU + 轮数上限)
  chat.py           ChatService:拼上下文 → astream → SSE delta 生成器
  extract.py        ExtractService:with_structured_output 售后抽取
  routers/          /api/chat(SSE)、/api/extract
eval_data/          售后抽取标注集
scripts/            eval_extract / demo_chat / demo_extract
dev-notes/ch01.md   分阶段开发留痕(brainstorm/计划/各 Task/code review/finish)
```

## SSE 契约

`POST /api/chat`,`body = {session_id?, message}` → `text/event-stream`,逐事件:

```
event: session   data: {"session_id":"..."}              # 仅新建会话
event: delta     data: {"content":"你"}
event: done      data: {"session_id":"...","finish":true}
event: error     data: {"message":"..."}                  # 上游异常
```

多轮语义:历史由**服务端按 session_id 持有**(进程内存),客户端只传 id。
