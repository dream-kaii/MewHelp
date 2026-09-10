from sqlalchemy import inspect

from app.db.models import Conversation, Faq, Message, Ticket


def test_table_names_match_ddl():
    assert Conversation.__tablename__ == "conversations"
    assert Message.__tablename__ == "messages"
    assert Faq.__tablename__ == "faq"
    assert Ticket.__tablename__ == "tickets"


def test_message_columns():
    cols = {c.name for c in inspect(Message).columns}
    assert {"conversation_id", "role", "content", "tool_calls", "tool_call_id", "created_at"} <= cols


def test_role_enum_values():
    role = Message.__table__.c.role.type
    assert set(role.enums) == {"user", "assistant", "tool"}


def test_ticket_primary_key_and_enums():
    assert [c.name for c in Ticket.__table__.primary_key.columns] == ["ticket_no"]
    assert set(Ticket.__table__.c.ticket_type.type.enums) == {"售后", "投诉", "咨询"}
    assert set(Ticket.__table__.c.status.type.enums) == {"待处理", "已处理"}
