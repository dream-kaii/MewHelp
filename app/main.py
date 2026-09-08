from fastapi import FastAPI

from app.config import get_settings


def create_app() -> FastAPI:
    app = FastAPI(title="MewHelp CS ch01")
    app.state.chat_model = None
    app.state.extract_model = None

    @app.get("/healthz")
    async def healthz():
        s = get_settings()
        return {"status": "ok", "provider": s.llm_provider, "model": s.llm_model}

    return app


app = create_app()
