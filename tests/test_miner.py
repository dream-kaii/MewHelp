import logging

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app.db import repository_knowledge as rk
from app.db.models import Conversation, KnowledgeChunk, Message
from app.rag import miner as miner_mod
from app.rag.chunker import build_text
from app.rag.embedder import FakeEmbedder
from app.rag.miner import QABatch, QAPair, PROMPT, dedupe_pairs, extract_qa, mine, promote_staged
from scripts import mine_knowledge as mk

pytestmark = pytest.mark.anyio


class _Structured:
    """替身:`with_structured_output(...)` 的返回值,只实现 extract_qa 用到的那一个方法。"""

    def __init__(self, batches, fail_on=None):
        self._batches = list(batches)
        self._fail_on = fail_on
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        text = messages[-1]["content"]
        if self._fail_on and self._fail_on in text:
            raise ValueError("模拟抽取失败(上游返回了不合法结构 / 网络抖动)")
        # 只给一个 batch 时每个会话都用它;给多个则按调用顺序消耗(便于按会话给不同结果)
        return self._batches.pop(0) if len(self._batches) > 1 else self._batches[0]


class FakeLLM:
    """绝不打真实 LLM:记录用过的 schema 和喂进去的消息。"""

    def __init__(self, batch, fail_on=None):
        self._batches = batch if isinstance(batch, list) else [batch]
        self._fail_on = fail_on
        self.schemas = []
        self.structured = None

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        self.structured = _Structured(self._batches, self._fail_on)
        return self.structured


class _SameEmbedder(FakeEmbedder):
    """所有文本都给同一向量,模拟"换说法但语义相同"。"""

    def encode(self, texts):
        return [[1.0] + [0.0] * 1023 for _ in texts]


class _FakeStore:
    """VectorStore 替身:记录 search 调用,返回构造时给的命中。"""

    def __init__(self, hits):
        self._hits = hits
        self.queries = []
        self.ensured = 0

    def ensure_collection(self):
        self.ensured += 1

    def search(self, vector, top_k=5):
        self.queries.append((vector, top_k))
        return list(self._hits)


async def _seed_conversation(db_session, *, user_id: str, user_msg="邮费怎么算", reply="按地区收取") -> int:
    conv = Conversation(user_id=user_id)
    db_session.add(conv)
    await db_session.flush()
    db_session.add(Message(conversation_id=conv.id, role="user", content=user_msg))
    db_session.add(Message(conversation_id=conv.id, role="assistant", content=reply))
    await db_session.commit()
    return conv.id


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


async def test_extract_qa_uses_structured_output_with_batch_schema():
    batch = QABatch(pairs=[QAPair(questions=["邮费怎么算"], answer="按地区收取", category="运费")])
    llm = FakeLLM(batch)

    out = await extract_qa(llm, "user: 邮费怎么算\nassistant: 按地区收取")

    assert llm.schemas == [QABatch]  # 用 QABatch 做结构化输出
    assert [len(out), out[0].answer, out[0].category] == [1, "按地区收取", "运费"]
    assert [m["role"] for m in llm.structured.messages] == ["system", "user"]
    assert llm.structured.messages[0]["content"] == PROMPT


async def test_mine_stages_then_promotes_pending_chunks(session_factory, db_session):
    conv_id = await _seed_conversation(db_session, user_id="u1")
    batch = QABatch(pairs=[QAPair(questions=["邮费怎么算", "运费多少"], answer="按地区收取", category="运费")])

    stats = await mine(session_factory, FakeLLM(batch), FakeEmbedder(), lookback_days=30)

    assert stats == {"conversations": 1, "staged": 1, "failed": 0}
    async with session_factory() as s:
        staged = await rk.staging_list(s)
    assert [(r["conversation_id"], r["category"], r["answer"]) for r in staged] == [(conv_id, "运费", "按地区收取")]
    # dedupe_hash = content_hash(sorted(questions) 的 JSON + "|" + answer),注意 sorted 按码点排序
    assert staged[0]["dedupe_hash"] == rk.content_hash('["运费多少", "邮费怎么算"]|按地区收取')

    pr = await promote_staged(session_factory)

    assert pr == {"promoted": 1}
    async with session_factory() as s:
        chunks = (await s.execute(select(KnowledgeChunk))).scalars().all()
        remaining = await rk.staging_list(s)
    assert len(chunks) == 1
    row = chunks[0]
    assert (row.doc_id, row.category, row.content_type, row.status, row.vector_id) == (
        f"mine:{conv_id}", "运费", "挖矿QA", "pending", None,
    )
    assert row.questions == "邮费怎么算、运费多少"
    assert row.text == build_text("运费", "邮费怎么算、运费多少", "按地区收取")
    assert remaining == []  # 暂存行已标记 promoted,不再回到待提升队列


async def test_mine_rerun_and_promote_rerun_are_idempotent(session_factory, db_session):
    await _seed_conversation(db_session, user_id="u2")
    batch = QABatch(pairs=[QAPair(questions=["邮费怎么算"], answer="按地区收取", category="运费")])

    first = await mine(session_factory, FakeLLM(batch), FakeEmbedder())
    again = await mine(session_factory, FakeLLM(batch), FakeEmbedder())

    assert [first["staged"], again["staged"]] == [1, 0]  # 哈希去重,不重复入暂存
    assert (await promote_staged(session_factory))["promoted"] == 1
    assert (await promote_staged(session_factory))["promoted"] == 0  # 已 promoted,不重复入库


async def test_mine_dedupes_within_one_conversation(session_factory, db_session):
    await _seed_conversation(db_session, user_id="u3")
    batch = QABatch(pairs=[
        QAPair(questions=["邮费怎么算"], answer="按地区收取", category="运费"),
        QAPair(questions=["邮费怎么算"], answer="按地区收取", category="运费"),
        QAPair(questions=["怎么开发票"], answer="下单时勾选", category="发票"),
    ])

    stats = await mine(session_factory, FakeLLM(batch), FakeEmbedder())

    assert stats["staged"] == 2  # 第三条重复被去重


async def test_mine_skips_conversations_outside_lookback(session_factory, db_session):
    conv_id = await _seed_conversation(db_session, user_id="u4")
    # 把会话挪到 90 天前(照 --lookback 30 天窗应扫不到)
    await db_session.execute(
        text("UPDATE conversations SET created_at = NOW() - INTERVAL 90 DAY WHERE id = :i"), {"i": conv_id}
    )
    await db_session.commit()
    batch = QABatch(pairs=[QAPair(questions=["邮费怎么算"], answer="按地区收取")])

    stats = await mine(session_factory, FakeLLM(batch), FakeEmbedder(), lookback_days=30)

    assert stats == {"conversations": 0, "staged": 0, "failed": 0}


# ---------- Critical:空 questions 不能把无人值守 job 卡死 ----------


def test_dedupe_skips_pairs_without_questions():
    pairs = [
        QAPair(questions=[], answer="按地区收取", category="运费"),
        QAPair(questions=["邮费怎么算"], answer="按地区收取", category="运费"),
    ]
    kept = dedupe_pairs(pairs, embedder=FakeEmbedder(), existing_vectors=[])
    assert [p.questions for p in kept] == [["邮费怎么算"]]


def test_dedupe_skips_empty_questions_without_embedder():
    # embedder=None 走的是另一条分支;空 questions 同样不能炸
    assert dedupe_pairs([QAPair(questions=[], answer="按地区收取")]) == []


async def test_mine_survives_empty_questions_pair(session_factory, db_session):
    """旧实现:questions=[] → `p.questions[0]` IndexError → 整个 job 中断。"""
    await _seed_conversation(db_session, user_id="u5")
    batch = QABatch(pairs=[QAPair(questions=[], answer="按地区收取", category="运费")])

    stats = await mine(session_factory, FakeLLM(batch), FakeEmbedder())

    assert stats == {"conversations": 1, "staged": 0, "failed": 0}


async def test_mine_continues_after_one_conversation_fails(session_factory, db_session):
    """单个会话抽取失败只跳过它:其余会话照常处理,函数正常返回。"""
    await _seed_conversation(db_session, user_id="u6", user_msg="邮费怎么算")
    ok_id = await _seed_conversation(db_session, user_id="u7", user_msg="怎么开发票")
    batch = QABatch(pairs=[QAPair(questions=["怎么开发票"], answer="下单时勾选", category="发票")])

    stats = await mine(session_factory, FakeLLM(batch, fail_on="邮费怎么算"), FakeEmbedder())

    assert stats == {"conversations": 2, "staged": 1, "failed": 1}
    async with session_factory() as s:
        staged = await rk.staging_list(s)
    assert [r["conversation_id"] for r in staged] == [ok_id]  # 坏会话没拦住好会话


async def test_mine_survives_staging_write_failure(session_factory, db_session, monkeypatch):
    """入库本身报错也只跳过该会话,不虚报 staged。"""
    await _seed_conversation(db_session, user_id="u14")
    batch = QABatch(pairs=[QAPair(questions=["邮费怎么算"], answer="按地区收取")])

    async def boom(session, items):
        raise RuntimeError("模拟写库失败")

    monkeypatch.setattr(miner_mod.rk, "staging_insert", boom)

    stats = await mine(session_factory, FakeLLM(batch), FakeEmbedder())

    assert stats == {"conversations": 1, "staged": 0, "failed": 1}


async def test_extract_qa_warns_on_unexpected_shape(caplog):
    class WeirdLLM:
        def with_structured_output(self, schema):
            class S:
                async def ainvoke(self, messages):
                    return None  # 上游没按 QABatch 返回

            return S()

    with caplog.at_level(logging.WARNING, logger="mewhelp.rag.miner"):
        assert await extract_qa(WeirdLLM(), "user: 邮费怎么算") == []

    assert "意外形状" in caplog.text  # 不静默


# ---------- Important 3:整体去重(既有向量池 + 跨会话 + 既有知识库) ----------


def test_dedupe_uses_existing_vectors_as_input_pool():
    pairs = [QAPair(questions=["邮费怎么算"], answer="按地区收取", category="运费")]
    pool = [[1.0] + [0.0] * 1023]

    assert dedupe_pairs(pairs, embedder=_SameEmbedder(), existing_vectors=pool, sim_threshold=0.95) == []
    assert len(pool) == 1  # 被丢弃的对不写回池


def test_dedupe_appends_kept_vectors_to_pool():
    """保留项的向量要就地追加进池子,调用方才能跨批次累积。"""
    pairs = [QAPair(questions=["邮费怎么算"], answer="按地区收取", category="运费")]
    pool = []

    assert len(dedupe_pairs(pairs, embedder=_SameEmbedder(), existing_vectors=pool)) == 1

    assert [len(pool), pool[0][0]] == [1, 1.0]


def test_dedupe_drops_pair_already_in_knowledge_store():
    pairs = [QAPair(questions=["邮费怎么算"], answer="按地区收取", category="运费")]
    store = _FakeStore([(7, 0.99)])

    assert dedupe_pairs(pairs, embedder=_SameEmbedder(), existing_vectors=[], store=store) == []

    assert [len(store.queries), store.queries[0][1]] == [1, 1]  # 只查 top_k=1


def test_dedupe_keeps_pair_when_store_has_no_hit():
    pairs = [QAPair(questions=["邮费怎么算"], answer="按地区收取", category="运费")]

    low = dedupe_pairs(pairs, embedder=_SameEmbedder(), existing_vectors=[], store=_FakeStore([(7, 0.10)]))
    none = dedupe_pairs(pairs, embedder=_SameEmbedder(), existing_vectors=[], store=_FakeStore([]))

    assert [len(low), len(none)] == [1, 1]  # 分数不够 / 空库 → 保留


async def test_mine_skips_pairs_found_in_store(session_factory, db_session):
    await _seed_conversation(db_session, user_id="u10")
    batch = QABatch(pairs=[QAPair(questions=["邮费怎么算"], answer="按地区收取", category="运费")])
    store = _FakeStore([(1, 0.99)])

    stats = await mine(session_factory, FakeLLM(batch), FakeEmbedder(), store=store)

    assert stats["staged"] == 0  # 既有知识库里已有 → 不入暂存
    assert [store.ensured, len(store.queries)] == [1, 1]  # 首跑也 ensure_collection,按对查询


async def test_mine_dedupes_near_duplicates_across_conversations(session_factory, db_session):
    """旧实现:每次 dedupe 都新建 kept_vecs → 跨会话近重漏网。"""
    await _seed_conversation(db_session, user_id="u8", user_msg="邮费怎么算")
    await _seed_conversation(db_session, user_id="u9", user_msg="运费如何计算")
    batches = [
        QABatch(pairs=[QAPair(questions=["邮费怎么算"], answer="按地区收取", category="运费")]),
        QABatch(pairs=[QAPair(questions=["运费如何计算"], answer="按地区收取运费", category="运费")]),
    ]

    stats = await mine(session_factory, FakeLLM(batches), _SameEmbedder(), sim_threshold=0.95)

    assert (stats["conversations"], stats["staged"], stats["failed"]) == (2, 1, 0)  # 同向量 → 只留第一对


async def test_mine_limit_zero_or_none_means_unlimited(session_factory, db_session):
    """limit<=0 不能退化成「扫 0 条」,负数更不能炸在 DB 上。"""
    await _seed_conversation(db_session, user_id="u12")
    batch = QABatch(pairs=[QAPair(questions=["邮费怎么算"], answer="按地区收取")])

    zero = await mine(session_factory, FakeLLM(batch), FakeEmbedder(), limit=0)
    none = await mine(session_factory, FakeLLM(batch), FakeEmbedder(), limit=None)

    assert [zero["conversations"], none["conversations"]] == [1, 1]


async def test_cli_rejects_non_positive_limit():
    for bad in ("0", "-3"):
        with pytest.raises(SystemExit) as exc:
            await mk.main(["--limit", bad])
        assert exc.value.code == 2  # argparse 直接报错退出,不连 DB、不空转


# ---------- Important 2:promote_staged 的 savepoint 容错 ----------


def _integrity_error(mark: str) -> IntegrityError:
    return IntegrityError("INSERT INTO knowledge_chunks ...", {}, Exception(f"Duplicate entry 'x' for key '{mark}'"))


async def _stage_two_pairs(session_factory, db_session) -> None:
    await _seed_conversation(db_session, user_id="u13")
    batch = QABatch(pairs=[
        QAPair(questions=["邮费怎么算"], answer="按地区收取", category="运费"),
        QAPair(questions=["怎么开发票"], answer="下单时勾选", category="发票"),
    ])
    await mine(session_factory, FakeLLM(batch), FakeEmbedder())


async def test_promote_absorbs_duplicate_content_hash_conflict(session_factory, db_session, monkeypatch):
    """并发 cron 下 insert_chunks 撞 uk_content_hash:只跳过这一行,其余照常入库。"""
    await _stage_two_pairs(session_factory, db_session)
    real_insert = rk.insert_chunks
    calls = {"n": 0}

    async def flaky(session, rows):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _integrity_error("uk_content_hash")
        return await real_insert(session, rows)

    monkeypatch.setattr(miner_mod.rk, "insert_chunks", flaky)

    assert (await promote_staged(session_factory))["promoted"] == 1  # 冲突那行被吸收,另一行入库
    async with session_factory() as s:
        chunks = (await s.execute(select(KnowledgeChunk))).scalars().all()
        remaining = await rk.staging_list(s)
    assert [len(chunks), remaining] == [1, []]  # 两行都已离开暂存队列


async def test_promote_reraises_non_duplicate_integrity_error(session_factory, db_session, monkeypatch):
    """非 uk_content_hash 的完整性错误必须原样上抛,且什么都不标 promoted。"""
    await _stage_two_pairs(session_factory, db_session)

    async def broken(session, rows):
        raise _integrity_error("Column 'text' cannot be null")

    monkeypatch.setattr(miner_mod.rk, "insert_chunks", broken)

    with pytest.raises(IntegrityError):
        await promote_staged(session_factory)

    async with session_factory() as s:
        assert len(await rk.staging_list(s)) == 2  # 整批回滚,两行都还留在暂存队列(没有被静默标记 promoted)
