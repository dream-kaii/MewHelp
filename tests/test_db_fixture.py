import pytest
from sqlalchemy import text

pytestmark = pytest.mark.anyio


async def test_db_session_points_to_test_database(db_session):
    r = await db_session.execute(text("SELECT DATABASE()"))
    assert r.scalar() == "mewhelp_test"


async def test_tables_exist_in_test_db(db_session):
    r = await db_session.execute(text("SHOW TABLES"))
    tables = {row[0] for row in r.fetchall()}
    assert {"conversations", "messages", "faq", "tickets"} <= tables
