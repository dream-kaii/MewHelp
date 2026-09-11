# MewHelp CS(ch01:电商智能客服 · 纯对话)

多章项目「电商智能客服系统」第一章:**纯对话跑通**。多轮对话 + SSE 流式、PromptTemplate 客服系统提示、售后诉求 `with_structured_output` 结构化抽取。模型走**统一 OpenAI 协议**,GPT / Claude(需 OpenAI 兼容网关)/ DeepSeek / Ollama 换着接,只改 `.env`。

> ch01 范围:纯对话跑通(不含工具调用 / Agent 循环 / LangGraph)。后续章节在此基础上扩展:
> ch02 增加了工具调用(单轮往返)、历史落库(MySQL)与聊天前端 —— 见下文「ch02」一节。
> ch03 把 `query_faq` 升级为向量召回的知识库 RAG(Milvus + 离线建库 + 挖矿 job)—— 见下文「ch03」一节。
> ch01 设计与计划存于
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
bash scripts/demo_chat.sh http://127.0.0.1:8000    # ①SSE 流式 ②同一会话第二轮接上下文
bash scripts/demo_extract.sh http://127.0.0.1:8000 # ③售后描述 -> 结构化 JSON
```

- ① 应看到多行 `event: delta` 与文本逐步出现,结尾 `event: done`。
- ② 第二轮复用同一会话:客户端在后续请求回传首轮 `session` 帧给出的 `conversation_id`,
  回复能引用第一轮内容。
- ③ 返回 `order_no / request_type / desired_solution` 三字段 JSON。

直接 curl 单发:

```bash
curl -sN -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","message":"你好"}'

curl -s -X POST http://127.0.0.1:8000/api/extract -H 'Content-Type: application/json' \
  -d '{"text":"猫粮订单20260901001漏气,想退货退款,上门取件"}'
```

## ch02:工具调用(演示)

聊天页现在会展示本轮工具轨迹徽章:调用中 🐾 工具名 → 完成 ✅ / 失败 ⚠(前端见 `web/index.html`)。

启动(与 ch01 同入口):

```bash
python -m uvicorn app.main:app --port 8000
```

先灌 FAQ 种子(验收②「退货政策」需 faq 表有数据;幂等,可反复执行):

```bash
python -m scripts.seed_faq     # 首次「新增 6 条」,重跑则全部「跳过」
```

工具调用演示(物流 / FAQ / 漏召回):

```bash
bash scripts/demo_tools.sh http://127.0.0.1:8000
```

- ① `订单 1001 的物流到哪了` → 见 `event: tool`(`query_logistics`)+ 据物流结果作答。
- ② `退货政策是什么` → `query_faq` 命中并作答(需先灌种子)。
- ③ `邮费是多少` → 选中 `query_faq` 但返回「FAQ 未找到…」(关键词漏召回,**预期**缺口)。

工具选型评测(真模型):

```bash
python -m scripts.eval_tool_selection   # 逐条核对选型;「邮费」为 known_gap,不计入退出码
```

SSE 在 ch01 帧基础上新增工具帧:

```
event: tool   data: {"call_id":"...","name":"query_logistics","args":{...},"status":"running"}
event: tool   data: {"call_id":"...","name":"query_logistics","status":"ok","summary":"..."}
```

## ch03:知识库 RAG(向量召回)

`query_faq` 从「关键词 LIKE 匹配」升级为**向量召回**:文档离线切分 → 双写 MySQL(状态机)+ Milvus(向量)→ 在线按语义检索。

### 依赖服务:Milvus

```bash
bash scripts/milvus.sh up      # 起 standalone(Docker),等 healthy 再继续
bash scripts/milvus.sh status  # healthz + 容器状态
bash scripts/milvus.sh down    # 停并移除容器
```

连接参数在 `.env`:`MILVUS_URI` / `MILVUS_TOKEN` / `KNOWLEDGE_COLLECTION`;
检索侧 `RAG_TOP_K`(默认 5)、`RAG_SCORE_THRESHOLD`(默认 0.5,低于此分视为未命中)。

### 建库(离线:切分 → 双写)

```bash
python -m scripts.build_knowledge                 # knowledge_docs/*.md 全部
python -m scripts.build_knowledge --doc knowledge_docs/运费说明.md
python -m scripts.build_knowledge --limit 50      # 本次最多向量化 50 块(分段执行)
python -m scripts.build_knowledge --skip-embed    # 只写 MySQL 不向量化(演练断点续跑)
```

幂等:按 `content_hash` 去重,重跑「新增 0 块」;中断后重跑本脚本会自动补齐未向量化的块
(`pending` = MySQL 有行但 Milvus 无向量),无需手动清库。

### 挖矿 job(从历史对话沉淀知识)

```bash
python -m scripts.mine_knowledge              # 扫历史会话 → 抽取问答 → 去重入 staging → promote 进知识库
python -m scripts.mine_knowledge --limit 20   # 本轮最多扫 20 个会话
python -m scripts.mine_knowledge --no-promote # 只入 staging 不推进知识库
```

promote 进知识库的行落为 `pending`,**需再跑一次 `python -m scripts.build_knowledge` 才会向量化**(与离线建库同一状态机)。

### 检索质量评测

```bash
python -m scripts.eval_retrieval   # 跑 eval_data/retrieval_samples.json,要求命中率 ≥ 75%
```

### 验收:聊天页 → query_faq → 向量召回

```bash
python -m uvicorn app.main:app --port 8000   # 另开一个终端
bash scripts/demo_rag.sh http://127.0.0.1:8000
```

- ① `邮费是多少`(知识库内)→ 见 `event: tool`(`query_faq`,`status: ok`,summary 内含相似度),
  最终答复引用 `运费说明.md` 内容(按地区收取 / 退换货运费由责任方承担),**不再出现「查不到」**。
- ② `你们店几点开门`(知识库外)→ 应礼貌兜底,不编造营业时间。

### 边界说明

- 在线检索**只跑 dense 单路**(向量召回单路);关键词检索**仅作服务故障降级**——当 Milvus / Embedding
  不可用时才 fallback,不是并行的第二路召回。因此常驻关键词命中场景(如旧版 FAQ 直查)不在 ch03 范围内。
- 判定「未找到」的条件:向量单路干净未命中,或命中分数低于 `RAG_SCORE_THRESHOLD`。

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
  context.py        历史裁剪 + token 预算(纯函数,按轮组整组裁剪)
  chat.py           ChatService:拼上下文 → 单轮工具往返 → SSE delta 生成器
  extract.py        ExtractService:with_structured_output 售后抽取
  db/               引擎/会话、ORM 模型、repository(conversations/messages/… 落库)
  tools/            工具注册表与各工具(物流/FAQ/工单等)
  agent/            tool_runner:第一段流式收集 tool_calls、第二阶段并行执行回灌
  routers/          /api/chat(SSE)、/api/extract
eval_data/          售后抽取标注集
scripts/            eval_* / demo_* / seed_faq / build_knowledge / mine_knowledge / milvus.sh
dev-notes/          分阶段开发留痕(ch01/ch02:brainstorm/计划/各 Task/code review/finish)
```

## SSE 契约

`POST /api/chat`,`body = {user_id, conversation_id?, message}` → `text/event-stream`,逐事件:

```
event: session   data: {"conversation_id": 12}                 # 仅新建会话(未传 conversation_id 或该 id 不属本 user)
event: tool      data: {"call_id":"...","name":"...","status":"running"|"ok"|"error", ...}
event: delta     data: {"content":"你"}
event: done      data: {"conversation_id":12,"finish":true}
event: error     data: {"message":"..."}                        # 上游异常
```

多轮语义:历史由**服务端按 conversation_id 持有**(落在 MySQL 的 `conversations` / `messages` 表),
客户端只传 id;新建会话时 `session` 帧回传服务端生成的 `conversation_id`,后续请求带上它即可续接。
`user_id` 用于会话归属校验:传入不属于该 user 的 `conversation_id` 会被当作新会话处理。
