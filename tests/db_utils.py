"""测试库工具:建库、按 sql/schema.sql 建表、清空表。全部走 pymysql(同步,供 sync/async fixture 共用)。"""
from pathlib import Path

import pymysql

from app.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "sql" / "schema.sql"
_TABLES = ("messages", "tickets", "conversations", "faq")  # 清空顺序:先子表(有外键)


def _connect(database: str | None = None):
    s = get_settings()
    return pymysql.connect(
        host=s.mysql_host, port=s.mysql_port, user=s.mysql_user, password=s.mysql_password,
        database=database, charset="utf8mb4", autocommit=True,
    )


def create_database_if_missing(name: str) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{name}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        conn.close()


def apply_schema(name: str) -> None:
    text = SCHEMA.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("--")]
    stmts = [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
    conn = _connect(name)
    try:
        with conn.cursor() as cur:
            for stmt in stmts:
                cur.execute(stmt)
    finally:
        conn.close()


def truncate_all(name: str) -> None:
    conn = _connect(name)
    try:
        with conn.cursor() as cur:
            cur.execute("SET FOREIGN_KEY_CHECKS=0")
            for t in _TABLES:
                cur.execute(f"TRUNCATE TABLE `{t}`")
            cur.execute("SET FOREIGN_KEY_CHECKS=1")
    finally:
        conn.close()
