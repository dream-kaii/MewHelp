-- =============================================================
-- ch02 · Function Calling 工具链 · 建表 DDL
-- 本章新建:faq / conversations / messages / tickets 四张表
-- 商品、订单、物流走工具内 mock,不建表
-- 全库统一 ENGINE=InnoDB、CHARSET=utf8mb4
-- 建表顺序:先 conversations,再依赖它的 messages / tickets
-- 说明:相较原始 DDL 统一加了 IF NOT EXISTS,保证脚本可重复执行
-- =============================================================

-- 会话壳:一通对话的统一身份,messages / tickets 都引用它
CREATE TABLE IF NOT EXISTS conversations (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '会话主键',
  user_id     VARCHAR(64)     NOT NULL                COMMENT '用户标识',
  status      ENUM('进行中','已转人工','已结束') NOT NULL DEFAULT '进行中' COMMENT '处理状态',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '开启时间',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客服会话';

-- 消息流水:一通会话底下挂 N 条,role 对齐 Chat Completions 协议
CREATE TABLE IF NOT EXISTS messages (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '消息主键',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '所属会话',
  role            ENUM('user','assistant','tool') NOT NULL COMMENT '角色:用户/助手/工具结果',
  content         TEXT            NULL                     COMMENT '消息正文,assistant 纯工具调用时可为空',
  tool_calls      JSON            NULL                     COMMENT 'assistant 消息带的工具调用申请单',
  tool_call_id    VARCHAR(64)     NULL                     COMMENT 'tool 消息对应的申请单 id,回灌时对号入座',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '产生时间',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_messages_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话消息流水';

-- FAQ 问答对:query_faq 的数据源;ch03 起检索改走向量库,这张表退居原始录入
CREATE TABLE IF NOT EXISTS faq (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'FAQ 主键',
  question    VARCHAR(512)    NOT NULL                COMMENT '问题',
  answer      TEXT            NOT NULL                COMMENT '答案',
  category    VARCHAR(64)     NOT NULL                COMMENT '分类',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='常见问答';

-- 人工工单:create_ticket 落地,工单号当业务主键
CREATE TABLE IF NOT EXISTS tickets (
  ticket_no       VARCHAR(32)     NOT NULL                COMMENT '工单号,如 T20260701008',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '关联会话,可倒查当时聊了什么',
  description     TEXT            NOT NULL                COMMENT '问题描述',
  ticket_type     ENUM('售后','投诉','咨询') NOT NULL     COMMENT '工单类型',
  status          ENUM('待处理','已处理') NOT NULL DEFAULT '待处理' COMMENT '处理状态',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (ticket_no),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_tickets_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='人工工单';

-- =============================================================
-- ch03 · 知识库 · 建表 DDL
-- 本章新建:knowledge_chunks(知识块,原文权威源) / knowledge_staging(挖矿暂存)
-- 向量库(Milvus)只存向量,原文与关系一律以 MySQL 为准
-- =============================================================

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
