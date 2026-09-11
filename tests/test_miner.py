import pytest
from sqlalchemy import select, text

from app.db import repository_knowledge as rk
from app.db.models import Conversation, KnowledgeChunk, Message
from app.rag.chunker import build_text
from app.rag.embedder import FakeEmbedder
from app.rag.miner import QABatch, QAPair, PROMPT, dedupe_pairs, extract_qa, mine, promote_staged

pytestmark = pytest.mark.anyio


class _Structured:
    """替身:`with_structured_output(...)` 的返回值,只实现 extract_qa 用到的那一个方法。"""

    def __init__(self, batch):
        self._batch = batch

    async def ainvoke(self, messages):
        self.messages = messages
        return self._batch


class FakeLLM:
    """绝不打真实 LLM:记录用过的 schema 和喂进去的消息。"""

    def __init__(self, batch: QABatch):
        self._batch = batch
        self.schemas = []
        self.structured = None

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        self.structured = _Structured(self._batch)
        return self.structured


async def _seed_conversation(db_session, *, user_id: str) -> int:
    conv = Conversation(user_id=user_id)
    db_session.add(conv)
    await db_session.flush()
    db_session.add(Message(conversation_id=conv.id, role="user", content="邮费怎么算"))
    db_session.add(Message(conversation_id=conv.id, role="assistant", content="按地区收取"))
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

    assert stats == {"conversations": 1, "staged": 1}
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

    assert stats == {"conversations": 0, "staged": 0}
