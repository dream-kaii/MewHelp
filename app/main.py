from fastapi import FastAPI

from app.config import get_settings
from app.routers import chat as chat_router
from app.routers import extract as extract_router

app = FastAPI(title="MewHelp CS ch01")
app.state.chat_model = None
app.state.extract_model = None


@app.get("/healthz")
async def healthz():
    s = get_settings()
    return {"status": "ok", "provider": s.llm_provider, "model": s.llm_model}


app.include_router(chat_router.router)
app.include_router(extract_router.router)

