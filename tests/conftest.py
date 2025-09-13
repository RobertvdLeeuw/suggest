import pytest
import asyncio
import os
import uuid
from pytest_postgresql import factories
from pytest_postgresql.janitor import DatabaseJanitor
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import text

from models import Base

# Single PostgreSQL instance for the entire test session
postgresql_proc = factories.postgresql_proc(port=None)

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
async def isolated_test_db(postgresql_proc):
    """Create isolated database per test (much faster than separate instances)."""
    # Generate unique database name
    test_id = str(uuid.uuid4())[:8]
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'main')
    db_name = f"test_db_{worker_id}_{test_id}"
    
    # Create the test database
    janitor = DatabaseJanitor(
        user=postgresql_proc.user,
        host=postgresql_proc.host,
        port=postgresql_proc.port,
        dbname=db_name,
        version=postgresql_proc.version,
    )
    janitor.init()
    
    connection_str = (
        f"postgresql+asyncpg://{postgresql_proc.user}:@"
        f"{postgresql_proc.host}:{postgresql_proc.port}/{db_name}"
    )
    
    # Set environment for your DatabaseManager
    old_test_mode = os.environ.get("TEST_MODE")
    old_db_name = os.environ.get("TEST_DATABASE_NAME")
    old_host = os.environ.get("POSTGRES_HOST")
    old_port = os.environ.get("POSTGRES_PORT") 
    old_user = os.environ.get("POSTGRES_USER")
    
    os.environ["TEST_MODE"] = "true"
    os.environ["TEST_DATABASE_NAME"] = db_name
    os.environ["POSTGRES_HOST"] = postgresql_proc.host
    os.environ["POSTGRES_PORT"] = str(postgresql_proc.port)
    os.environ["POSTGRES_USER"] = postgresql_proc.user
    
    # Reset DatabaseManager to pick up new environment
    from db import DatabaseManager
    await DatabaseManager.cleanup_all_instances()
    
    # Create engine and set up schema
    engine = create_async_engine(connection_str, poolclass=NullPool, echo=False)
    
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine, db_name
    
    # Cleanup
    await engine.dispose()
    await DatabaseManager.cleanup_all_instances()
    janitor.drop()
    
    # Restore environment
    if old_test_mode is not None:
        os.environ["TEST_MODE"] = old_test_mode
    else:
        os.environ.pop("TEST_MODE", None)
        
    if old_db_name is not None:
        os.environ["TEST_DATABASE_NAME"] = old_db_name
    else:
        os.environ.pop("TEST_DATABASE_NAME", None)
        
    # Restore other env vars...
    for key, old_val in [("POSTGRES_HOST", old_host), 
                         ("POSTGRES_PORT", old_port),
                         ("POSTGRES_USER", old_user)]:
        if old_val is not None:
            os.environ[key] = old_val

@pytest.fixture(scope="function")  
async def db_session(isolated_test_db):
    """Create async session for tests."""
    engine, db_name = isolated_test_db
    
    async_session = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session

# Convenience fixture that just sets up the test database
@pytest.fixture
async def test_db(isolated_test_db):
    """Simple test database fixture."""
    engine, db_name = isolated_test_db
    yield db_name
