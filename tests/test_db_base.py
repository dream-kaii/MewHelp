from urllib.parse import quote_plus

from app.config import get_settings
from app.db.base import Base, build_database_url


def test_build_database_url_uses_aiomysql_and_charset():
    s = get_settings()
    url = build_database_url(s)
    assert url.startswith("mysql+aiomysql://")
    assert "charset=utf8mb4" in url
    # 查询串 charset 必须位于库名之后,故取 "?" 前的主干判断库名
    assert url.split("?")[0].endswith("/" + s.mysql_database)


def test_build_database_url_can_target_other_database():
    s = get_settings()
    url = build_database_url(s, database="mewhelp_test")
    assert url.split("?")[0].endswith("/mewhelp_test")


def test_build_database_url_escapes_password():
    from app.config import Settings

    s = Settings(mysql_user="u", mysql_password="p@ss:word/1", mysql_host="h", mysql_port=3307, mysql_database="d")
    url = build_database_url(s)
    assert quote_plus("p@ss:word/1") in url and "h:3307" in url


def test_base_metadata_null():
    assert Base.metadata is not None
