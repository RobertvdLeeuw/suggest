import sys
from typing import AsyncGenerator
import os

from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from logger import LOGGER
from models import Base

from dotenv import load_dotenv
load_dotenv()


def create_database_url(unittest: bool = False) -> URL:
    """
    """
    url = URL.create(
        drivername='postgresql+asyncpg',
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["POSTGRES_PORT"]),
    )
    return url


def create_async_engine_and_session(url: str | URL) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """
    """
    try:
        # 数据库引擎
        engine = create_async_engine(
            url,
            future=True,
            # 中等并发
            #pool_size=10,  # 低：- 高：+
            #max_overflow=20,  # 低：- 高：+
            #pool_timeout=30,  # 低：+ 高：-
            #pool_recycle=3600,  # 低：+ 高：-
            #pool_pre_ping=True,  # 低：False 高：True
            #pool_use_lifo=False,  # 低：False 高：True
        )
    except Exception as e:
        LOGGER.info(f"Could not create session/connection with error: {e}")
        sys.exit()
    else:
        db_session = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            autoflush=False,  # 禁用自动刷新
            expire_on_commit=False,  # 禁用提交时过期
        )
        return engine, db_session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_db_session() as session:
        yield session


async def create_table() -> None:
    async with async_engine.begin() as coon:
        await coon.run_sync(Base.metadata.create_all)


SQLALCHEMY_DATABASE_URL = create_database_url()
async_engine, async_db_session = create_async_engine_and_session(SQLALCHEMY_DATABASE_URL)
