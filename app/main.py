from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import get_settings
from app.routers import chat as chat_router
from app.routers import extract as extract_router

app = FastAPI(title="MewHelp CS ch01")
app.state.chat_model = None
app.state.extract_model = None

_WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_WEB_INDEX)


@app.get("/healthz")
async def healthz():
    s = get_settings()
    return {"status": "ok", "provider": s.llm_provider, "model": s.llm_model}


app.include_router(chat_router.router)
app.include_router(extract_router.router)

