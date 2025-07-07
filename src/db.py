import os
import asyncio

from sqlalchemy import URL, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from logger import LOGGER
import traceback

from models import Base

from dotenv import load_dotenv
load_dotenv()

def create_database_url() -> URL:
    return URL.create(drivername='postgresql+asyncpg',
                      username=os.environ["POSTGRES_USER"],
                      password=os.environ["POSTGRES_PASSWORD"],
                      host=os.environ["POSTGRES_HOST"],
                      port=int(os.environ["DB_PORT"]))

ENGINE = None    
SESSION_FACTORY = None
async def init_db():
    global ENGINE, SESSION_FACTORY

    try:
        ENGINE = create_async_engine(create_database_url(), 
                                     future=True, 
                                     echo=True, 
                                     pool_pre_ping=True)
    except Exception as e:
        LOGGER.error(f"Could not create session/connection: {traceback.format_exc()}")
        exit()

async def setup_tables():
    async with ENGINE.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)

    SESSION_FACTORY = async_sessionmaker(bind=ENGINE,
                                         class_=AsyncSession,
                                         expire_on_commit=False)  # Keep objects usable after commit.

async def get_session() -> AsyncSession:
    if ENGINE is None:
        # TODO: Make this wait with timeout error?
        raise Exception("Tried to get session before DB engine was initialized.")

    return SESSION_FACTORY()


LOGGER.info("Initializing DB engine.")
asyncio.run(init_db())

LOGGER.info("Setting up tables.")
asyncio.run(setup_tables())

LOGGER.info("DB connection initialized.")

