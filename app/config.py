from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全部来自 .env / 环境变量;缺省值面向本地 Ollama 与离线单测。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = ""
    llm_model: str = "qwen2.5:7b"
    llm_provider: str = "ollama"
    temperature: float = 0.7
    max_tokens: int = 1024
    history_budget_tokens: int = 2048
    session_max_turns: int = 30
    session_max_count: int = 200
    extract_method: str = "function_calling"
    cs_shop_name: str = "MewHelp"
    cs_staff_name: str = "小喵"

    # MySQL
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "mewhelp"
    mysql_test_database: str = "mewhelp_test"
    # 工具执行
    tool_timeout_seconds: float = 8.0
    tool_max_retries: int = 2
    mock_seed_salt: str = "mewhelp"


@lru_cache
def get_settings() -> Settings:
    return Settings()
