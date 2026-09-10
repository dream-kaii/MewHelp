from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import get_settings
from app.db.base import dispose_engine
from app.routers import chat as chat_router
from app.routers import extract as extract_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # 关闭时释放连接池:连接绑定在创建它的事件循环上,
    # 若不释放,同一进程里换个循环再用池化连接会炸(TestClient 每例一个新循环)。
    await dispose_engine()


app = FastAPI(title="MewHelp CS ch01", lifespan=lifespan)
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

