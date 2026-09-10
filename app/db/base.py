from urllib.parse import quote_plus

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


def build_database_url(settings: Settings, database: str | None = None) -> str:
    db = database or settings.mysql_database
    pwd = quote_plus(settings.mysql_password)
    return (
        f"mysql+aiomysql://{settings.mysql_user}:{pwd}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{db}?charset=utf8mb4"
    )


_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            build_database_url(get_settings()),
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def dispose_engine() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


async def create_all(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
