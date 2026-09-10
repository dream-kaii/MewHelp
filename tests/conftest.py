import os

# 必须最先执行:.env 里 MYSQL_DATABASE=mewhelp,这里用环境变量覆盖到测试库
# (pydantic-settings 中环境变量优先级高于 .env 文件)
os.environ.setdefault("MYSQL_DATABASE", "mewhelp_test")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.base import build_database_url
from app.main import app
from tests.db_utils import apply_schema, create_database_if_missing, truncate_all


@pytest.fixture(scope="session")
def _test_db() -> str:
    s = get_settings()
    create_database_if_missing(s.mysql_test_database)
    apply_schema(s.mysql_test_database)
    return s.mysql_test_database


@pytest.fixture
def client(_test_db):
    """同步 TestClient。依赖 _test_db 保证测试库有表,并在每个用例前清空。"""
    truncate_all(_test_db)
    app.state.chat_model = None
    app.state.extract_model = None
    with TestClient(app) as c:
        yield c
    truncate_all(_test_db)


@pytest.fixture
async def session_factory(_test_db):
    s = get_settings()
    engine = create_async_engine(build_database_url(s, database=s.mysql_test_database))
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


@pytest.fixture
async def db_session(session_factory):
    truncate_all(get_settings().mysql_test_database)
    async with session_factory() as session:
        yield session
