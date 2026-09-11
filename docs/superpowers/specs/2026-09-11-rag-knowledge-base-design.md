# ch03 电商智能客服 · 知识库向量检索设计

- 日期:2026-09-11
- 章节范围:多章项目「电商智能客服系统」的 **ch03** —— 把 `query_faq` 的内部实现从关键词查表升级为 **dense 向量语义检索**(BGE-M3 + Milvus)
- 前序:`docs/superpowers/specs/2026-09-08-ecommerce-cs-ch01-design.md`(纯对话)、`docs/superpowers/specs/2026-09-10-function-calling-tools-design.md`(Function Calling 工具链)
- 过程留痕:`dev-notes/ch03.md`

## 1. 目标与验收

**目标**:离线把知识文档与历史对话沉淀成知识块并双写落库(MySQL 原文权威 + Milvus 向量),在线把 `query_faq` 的内部实现换成向量语义检索,**工具入参/出参契约保持不变**。

**验收标准**
1. 「邮费是多少」这类换说法的问题,现在能召回运费说明并答对(关键词查表答不了)
2. 故意中断建库任务再重跑,漏向量化的块能被捡起补齐

## 2. 技术栈(定死)

- 嵌入模型:**BGE-M3**,**本地权重**(`FlagEmbedding.BGEM3FlagModel('BAAI/bge-m3')`,dense 向量 **1024 维**)
- 向量库:**Milvus**(standalone,Docker;`pymilvus.MilvusClient`)
- MySQL:知识原文与元数据权威源(沿用 ch02 的 SQLAlchemy 2.0 async + aiomysql)
- 只跑 **dense 单路**:不做关键词召回 / 混合检索 / 重排

## 3. 环境前提(已确认/进行中)

- 本机原无 Docker;用户同意**安装 Docker Desktop**,由 docker compose 起 Milvus standalone(`19530`)。
- 本机无 torch/FlagEmbedding,首次运行会下载依赖与权重(~2.3GB;可用 hf-mirror)。

## 4. 分层骨架

```
app/rag/
  chunker.py     # 结构感知切分(Markdown 标题层级 / 表格 / 超长递归 / 重叠裁句)
  embedder.py    # BGE-M3 封装:encode(list[str]) -> list[list[float]] (1024 维)
  store.py       # Milvus 封装:ensure_collection / upsert / search / delete;集合名 knowledge
  retriever.py   # 在线检索:query → 向量 → Top-K + 阈值 → 回 MySQL 取 answer 拼装
app/db/
  models.py                 # 追加 KnowledgeChunk / KnowledgeStaging(镜像新 DDL)
  repository_knowledge.py   # 两张表的读写与状态机(唯一写 SQL 处,沿用 ch02 约定)
scripts/
  build_knowledge.py        # 离线建库:文档 → 切分 → 双写(可重跑、可 --limit/--doc)
  mine_knowledge.py         # 挖知识 job:conversations/messages → LLM 抽 QA → 暂存 → 去重 → 入库
knowledge_docs/             # 样例 Markdown(退货政策 / 运费说明 / 商品FAQ / 售后手册,含多级标题与大表格)
sql/schema.sql              # 追加两张表 DDL(不改既有四张表)
```

`app/tools/kb.py::query_faq(keyword)` 只换内部实现(改调 `retriever`),**签名与返回文本风格不变**(无命中仍返回含「未找到」的提示,保留 ch02 语义)。

## 5. 切分策略(参数可配)

- 按 **Markdown 标题层级**做结构感知切分;块内文本 = `category + questions + answer` 拼装,该拼装文本用于向量化。
- 超长内容**递归切**;相邻块之间**加重叠**(默认 `CHUNK_OVERLAP=120` 字符),重叠边界**回退到最近句号**,不留半截话。
- **大表格按行切,每块复制表头**,保证每块自洽。
- 字段来源规则:
  - 商品 FAQ / 挖出的问答对:`questions` 填**真实问法**。
  - 政策手册这类没有天然问题的:`questions` 填**所在章节标题**,`category` 填**上级标题路径**。
- 四类元数据 —— **章节路径、内容类型、是否关键条款、前后块指针** —— 只存 MySQL(**不进向量**)。

## 6. 落库结构

### 6.1 MySQL `knowledge_chunks`(原文权威源)

| 列 | 说明 |
|---|---|
| `id` | 自增主键;**同时作为 Milvus 主键** |
| `doc_id` | 来源文档标识(文件相对路径或挖矿批次标识) |
| `category` | 分类(章节上级路径 / 挖矿所得) |
| `questions` | 问法(真实问法或章节标题) |
| `answer` | 答案正文 |
| `text` | `category + questions + answer` 拼装文本(向量化输入) |
| `section_path` | 章节路径元数据(不进向量) |
| `content_type` | 内容类型元数据(如 政策 / FAQ / 挖矿QA) |
| `is_key_clause` | 是否关键条款元数据 |
| `prev_id` / `next_id` | 前后块指针元数据 |
| `content_hash` | 归一化内容哈希(去重/幂等) |
| `vector_id` | 向量化成功后回填 |
| `status` | `pending`(待向量化) / `embedded`(已向量化) |
| `created_at` / `updated_at` | 时间戳 |

### 6.2 MySQL `knowledge_staging`(挖矿暂存)

挖知识 job 的中间表:LLM 抽出的 QA 原文、来源会话/消息 id、`dedupe_hash`、处理状态(`staged` / `deduped` / `promoted` / `dropped`)、时间戳。

### 6.3 Milvus 集合 `knowledge`

**只放两个字段**:`id`(int64 主键,对应 `knowledge_chunks.id`)+ `vector`(FLOAT_VECTOR,1024 维,`metric_type="COSINE"`)。文本与元数据一律回 MySQL 取。

## 7. 双写与幂等(验收②的实现基础)

1. 离线建库/挖矿产出 → 写 MySQL,`status='pending'`;
2. 批量取 `pending` 或 `vector_id IS NULL` 的块 → BGE-M3 编码;
3. Milvus `upsert(id, vector)`(**按主键幂等**);
4. 回填 `vector_id`,状态转 `embedded`。

**断点续跑**:重跑只扫未完成的块(不重算已 `embedded` 的);`build_knowledge.py` 支持 `--limit` / `--doc` 便于分段执行;**中断后重跑即可补齐**(验收②以真实中断验证:跑到一半终止进程 → 重跑 → `pending` 清零)。

## 8. 挖知识 job(命令可启动,自行部署)

`python -m scripts.mine_knowledge [--since <date>] [--limit N] [--batch N]`

- 数据源:ch02 已落库的 `conversations` / `messages`;
- 分批喂 LLM(沿用 ch02 的 OpenAI 协议 `ChatOpenAI`)抽取问答对 → 写 `knowledge_staging`;
- **整体去重**:归一化哈希精确去重 + BGE-M3 向量近重复检测(余弦 > `DEDUPE_SIM_THRESHOLD` 丢弃);
- 存活项入 `knowledge_chunks`(`content_type='挖矿QA'`,`status='pending'`),交建库流程向量化;
- **纯 CLI**,无内置常驻调度:由 cron / Windows 任务计划调用,或手动直跑(README 给示例命令)。

## 9. 在线检索

`retriever.search(query, top_k)`:
query → BGE-M3 编码 → Milvus `search(limit=top_k)` → 按 `RAG_SCORE_THRESHOLD` 过滤 → 按 id 回 MySQL 取 `answer`(与 `category`/`questions` 组织成可读文本)→ 返回字符串。

`query_faq` 无命中(全低于阈值或集合为空)时返回**含「未找到」**的提示,保留 ch02 的漏召回语义。

## 10. 配置(.env 新增)

```
MILVUS_URI=http://127.0.0.1:19530
MILVUS_TOKEN=
KNOWLEDGE_COLLECTION=knowledge
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

## 11. 测试与验收分工

- **TDD(纯逻辑/可注入)**:`chunker`(标题层级、表格复制表头、超长递归、重叠裁句)、去重(哈希 + 向量近重)、`repository_knowledge` 状态机与断点续跑(用 fake embedder + 真 MySQL 测试库)、`retriever` 装配/阈值/`未找到` 分支(fake Milvus client)。
- **标注集替代单测**(用户规则):`eval_data/retrieval_samples.json`(问句 → 期望命中的 doc/章节,含「邮费是多少」),跑**真 BGE-M3 + 真 Milvus** 算命中率;ch02 的 `tool_selection_samples` 评测保持通过。
- **验收②**以真实中断验证(pipeline 中途终止再重跑)。

## 12. 章节边界(不做)

- 关键词召回 / 混合检索 / 重排(rerank)—— 本章只跑 dense 单路
- 不改 `query_faq` 工具契约(入参/出参保持一致)
- 不引入 LangChain 的向量库抽象层(直接用 `pymilvus`)
- 不改 ch02 已建的四张表;仅新增两张知识表

## 13. 过程约束

- 执行阶段走 **Subagent-Driven**(每 Task 独立子代理 + 任务间评审)
- 库/框架 API 用法先查 **Context7** 官方文档(已核:`MilvusClient.create_collection(dimension, metric_type="COSINE")`、`CollectionSchema/FieldSchema/DataType`、`insert/search/query/delete`;`BGEM3FlagModel('BAAI/bge-m3').encode(..., return_dense=True)['dense_vecs']`)
- 定死选型走不通 → 停下问用户,不自行换方案
- `dev-notes/ch03.md` 分阶段追记(四样:用户关键原话 / 我的关键产出 / 用户拒绝或纠偏 / 翻车与返工)
