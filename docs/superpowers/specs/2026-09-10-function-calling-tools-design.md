# ch02 电商智能客服 · Function Calling 工具链设计

- 日期:2026-09-10
- 章节范围:多章项目「电商智能客服系统」的 **ch02** —— 给现有客服聊天接上"查数据"能力(工具调用),**不做** Agent 多轮循环、RAG/向量检索
- 前序:`docs/superpowers/specs/2026-09-08-ecommerce-cs-ch01-design.md`(纯对话)
- 过程留痕:`dev-notes/ch02.md`

## 1. 目标与验收

**目标**:客服聊天接入五个业务工具,模型自行决定调哪个;工具结果回灌后收敛作答;聊天记录(含工具调用与结果)落 MySQL;聊天页显示本轮工具轨迹。

**验收标准**
1. 聊天页问「订单 1001 的物流到哪了」→ 看到模型选中工具(气泡带工具徽章)并按工具返回结果作答
2. 问「退货政策是什么」→ `query_faq` 查到并作答
3. 换个说法问「邮费是多少」→ 关键词查表**查不出来**(漏召回,**预期结果**,记录留给下一步升级)

## 2. 技术栈与依赖(定死)

- FastAPI + **SQLAlchemy 2.0 async**(`create_async_engine` / `async_sessionmaker` / `DeclarativeBase`)+ **MySQL**(用户本机 3306,非 Docker——见 §10 环境说明)
- 异步驱动:**`aiomysql`**(纯 Python,免编译);建表脚本用 `pymysql`
- LangChain 1.x:`langchain.tools.tool` / `ChatOpenAI.bind_tools` / `AIMessage.tool_calls` / `ToolMessage(content, tool_call_id=...)`
- 新增依赖:`sqlalchemy>=2.0`、`aiomysql`、`pymysql`

## 3. 分层骨架

```
app/
  db/
    base.py         # DeclarativeBase + engine + async_sessionmaker(读 .env)
    models.py       # conversations / messages / faq / tickets 四个 2.0 模型(镜像 sql/schema.sql)
    repository.py   # 唯一写 SQL 的地方:会话/消息 CRUD、FAQ LIKE 查询、工单插入
  tools/
    registry.py     # 工具登记表:名字→tool、导出 schema、按名派发
    business.py     # query_order / query_product / query_logistics(mock,确定性)
    kb.py           # query_faq(SQL LIKE)
    ops.py          # create_ticket(写 tickets)
    mockdata.py     # 确定性 mock 数据(hash 播种)
  agent/
    tool_runner.py  # 单轮工具往返:bind_tools → 收集 tool_calls → 执行(超时/重试)→ 产出 ToolMessage
  chat.py           # ChatService 改造:工具往返 + 流式收敛 + 落库
```

**DRY**:工具只调 `repository.py`,不写裸 SQL;`registry.py` 是工具的唯一入口。

## 4. 数据模型与 DDL

- DDL 见 [`sql/schema.sql`](../../../sql/schema.sql)(用户给定,统一加 `IF NOT EXISTS` 以便幂等);建库建表脚本 [`scripts/init_db.py`](../../../scripts/init_db.py)。
- SQLAlchemy 模型**镜像**该 DDL,**不引 Alembic**(ch02 不做迁移)。
- 字段要点:`conversations(id, user_id, status, ...)`;`messages(conversation_id, role∈user/assistant/tool, content, tool_calls JSON, tool_call_id, ...)`;`faq(question, answer, category)`;`tickets(ticket_no 主键, conversation_id, description, ticket_type, status, ...)`。

## 5. 工具层(五个)

| 工具 | 数据源 | 说明 |
|---|---|---|
| `query_order(order_id)` | mock | 订单状态/金额/商品 |
| `query_product(product_id)` | mock | 商品名/价格/库存 |
| `query_logistics(order_id)` | mock | 物流节点/承运商/预计到达 |
| `query_faq(keyword)` | faq 表 | `LIKE %keyword%` 查问题/答案 |
| `create_ticket(description, ticket_type)` | tickets 表 | 写工单,返回 ticket_no |

- 全部 `@tool` 定义;参数 Schema 由 pydantic 签名自动生成并校验;非法参数返回结构化错误(不让模型崩)。
- **mock 确定性**:`random.Random(hash(输入 + MOCK_SEED_SALT))`,同一输入永远同结果(演示/评测可断言)。
- **执行外壳**:`asyncio.wait_for` 超时(默认 8s)+ 重试(默认 2 次,退避);最终失败也回灌"工具执行失败"文本给模型组织话术。
- **上下文注入**:`create_ticket` 需要 `conversation_id` → 按请求构建工具(闭包注入 conversation_id + AsyncSession),不用全局变量。
- **ticket_no 生成**:`T + YYYYMMDD + 3 位当日序号`(查当日最大号 +1),主键冲突则重试。

## 6. 单轮工具调用流程(核心)

**语义硬约束:模型的一次工具决策就到此为止——全程只有一次工具往返。**

```
POST /api/chat {user_id, conversation_id?, message}
 1. 取/建 conversation(首次无 id → 建行,回传 conversation_id);落库 user 消息
 2. 【唯一一次工具往返】model.bind_tools(tools).astream(history)
      · 无 tool_calls → 纯聊天,已流式吐完 → done(ch01 行为不变)
      · 有 tool_calls:
          - 该轮内调用【全部并行执行】,每个推状态帧:
              event: tool  data:{"call_id","name","args","status":"running"}
              event: tool  data:{"call_id","name","status":"ok|error","summary":""}
          - 落库 assistant(tool_calls JSON) + 各条 tool 结果(role=tool, tool_call_id)
 3. 【收敛生成】用【不绑定工具】的同一模型 + 回灌 ToolMessage 的消息,astream 出最终答复
      → 逐 token 推 event: delta;落库 assistant 文本;event: done
```

- step 3 不带工具 → 物理上不可能再调工具,必然收敛,不依赖 prompt 约束。
- step 2 若一次返回多个调用,按用户决策**全执行**,但属**同一轮**。
- 工具执行完成后:
  - `create_ticket` 执行成功 → 将该 `conversations.status` 置为「已转人工」;其余情况保持「进行中」
  - 历史以 `messages` 表为准,复用 ch01 `context.py` 做 token 裁剪。

## 7. SSE 契约(扩展,兼容 ch01)

```
event: session   data:{"conversation_id": 12, "user_id": "..."}   # 新建会话时
event: tool      data:{"call_id","name","args","status":"running"} # 工具开始
event: tool      data:{"call_id","name","status":"ok","summary":""}# 工具结束
event: delta     data:{"content":"你"}
event: done      data:{"conversation_id":12,"finish":true}
event: error     data:{"message":"..."}
```

## 8. 配置(.env 新增)

```
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=...
MYSQL_PASSWORD=...
MYSQL_DATABASE=mewhelp
TOOL_TIMEOUT_SECONDS=8
TOOL_MAX_RETRIES=2
MOCK_SEED_SALT=mewhelp
```

用户身份:聊天页用 `localStorage` 生成并持久化 `user_id`,随请求带上(无登录)。

同步更新 `.env.example`(不写入真实口令)。

## 9. 测试与验收分工

- **TDD**:`repository`(CRUD / LIKE)、mock 确定性、注册表派发与参数校验、超时重试、`tool_runner` 单轮装配、SSE 帧序列 —— 用 fake model 注入 tool_calls,不碰真网络。
- **标注样例替代单测**(用户规则):工具**选型质量**属"模型行为",用 `eval_data/tool_selection_samples.json`(问句 → 期望工具 / 期望不调)跑真模型算准确率;「邮费是多少」漏召回作为**已知缺口**记录。
- **聊天页改造**:Vibe Coding 例外(工具徽章/状态展示),不套 brainstorm/TDD/code review。

## 10. 环境说明(已发生)

原定 "Docker 起 MySQL",实测本机 **Docker 不可用**(无 CLI / 无服务 / Desktop 主程序缺失),WSL Ubuntu 内亦无 docker。**用户确认改用本机已在运行的 MySQL(root@127.0.0.1:3306,库 `mewhelp`)**,DDL 由用户提供。已用 `scripts/init_db.py` 建库建表完成。

## 11. 章节边界(不做)

- Agent 多轮自动循环(只在单轮内执行工具调用,不循环)
- 向量检索 / RAG
- Alembic 迁移;不改用户给定 DDL
- 工具接真实电商/物流接口(演示用 mock)

## 12. 过程约束

- 执行阶段按用户要求走 **Subagent-Driven**(每 Task 独立子代理 + 任务间 review),不用 inline。
- 库/框架 API 用法先查 Context7 官方文档再动手。
- 定死选型走不通 → 停下问用户,不自行换方案。
- `dev-notes/ch02.md` 分阶段追记(四样:用户关键原话 / 我的关键产出 / 用户拒绝或纠偏 / 翻车与返工)。
