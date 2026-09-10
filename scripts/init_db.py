"""建库 + 建表(幂等)。用法:python -m scripts.init_db

读 .env 的 MYSQL_* 连接本地 MySQL,建库(若不存在)后执行 sql/schema.sql。
只用 PyMySQL,不依赖应用代码;可重复执行。
"""
import sys
from pathlib import Path

import pymysql
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "sql" / "schema.sql"


class DBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "mewhelp"


def _statements(sql_text: str) -> list[str]:
    """去掉 -- 行注释后按 ; 切分成可执行语句。"""
    lines = [ln for ln in sql_text.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def main() -> int:
    s = DBSettings()
    conn = pymysql.connect(
        host=s.mysql_host,
        port=s.mysql_port,
        user=s.mysql_user,
        password=s.mysql_password,
        charset="utf8mb4",
        autocommit=True,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{s.mysql_database}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            print(f"[ok] database `{s.mysql_database}` 就绪")
            cur.execute(f"USE `{s.mysql_database}`")
            for stmt in _statements(SCHEMA.read_text(encoding="utf-8")):
                cur.execute(stmt)
            cur.execute("SHOW TABLES")
            tables = [r[0] for r in cur.fetchall()]
            print("[ok] tables:", ", ".join(tables))
            for t in ("conversations", "messages", "faq", "tickets"):
                cur.execute(f"SELECT COUNT(*) FROM `{t}`")
                print(f"     {t}: {cur.fetchone()[0]} rows")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
