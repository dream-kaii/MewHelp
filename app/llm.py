from langchain_openai import ChatOpenAI

from app.config import get_settings


def get_chat_model() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key or "not-needed",  # 本地 Ollama 等不需要真实 key 的上游
        model=s.llm_model,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        streaming=True,
        request_timeout=120,
    )
