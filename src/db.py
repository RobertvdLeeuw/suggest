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
                                     echo=False, 
                                     pool_pre_ping=True)

        # logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
        # logging.getLogger('sqlalchemy.dialects').setLevel(logging.WARNING)
        # logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
        # logging.getLogger('sqlalchemy.orm').setLevel(logging.WARNING)

        SESSION_FACTORY = async_sessionmaker(bind=ENGINE,
                                             class_=AsyncSession,
                                             expire_on_commit=False)  # Keep objects usable after commit.
    except Exception as e:
        LOGGER.error(f"Could not create session/connection: {traceback.format_exc()}")
        raise Exception(f"Could not create session/connection: {traceback.format_exc()}")

async def setup_tables():
    await init_db()

    async with ENGINE.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE;"))
        await conn.execute(text("CREATE SCHEMA public;"))

        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)


def get_session() -> AsyncSession:
    if ENGINE is None:
        # TODO: Make this wait with timeout error?
        raise Exception("Tried to get session before DB engine was initialized.")

    return SESSION_FACTORY()


async def setup():
    LOGGER.info("Initializing DB engine.")
    await init_db()
    LOGGER.info("DB connection initialized.")


if __name__ == "__main__":
    LOGGER.info("Setting up tables.")

    if not input("This will remove all tables and data from the DB. Are you sure? ").lower().startswith("y"):
        exit()

    asyncio.run(setup_tables())


