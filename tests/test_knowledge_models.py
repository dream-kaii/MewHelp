from sqlalchemy import inspect

from app.db.models import KnowledgeChunk, KnowledgeStaging


def test_knowledge_chunk_columns():
    cols = {c.name for c in inspect(KnowledgeChunk).columns}
    assert {
        "id", "doc_id", "category", "questions", "answer", "text",
        "section_path", "content_type", "is_key_clause", "prev_id", "next_id",
        "content_hash", "vector_id", "status", "created_at", "updated_at",
    } <= cols
    assert KnowledgeChunk.__table__.c.status.type.enums == ["pending", "embedded"]


def test_knowledge_staging_columns():
    cols = {c.name for c in inspect(KnowledgeStaging).columns}
    assert {"id", "conversation_id", "source_message_ids", "questions", "answer",
            "category", "dedupe_hash", "status", "created_at", "updated_at"} <= cols
    assert KnowledgeStaging.__table__.c.status.type.enums == [
        "staged", "promoted", "dropped"
    ]
