import pytest
import asyncio

from typing import Optional, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from db import DatabaseManager
from models import Base

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_db_manager():
    """Create a test database manager for the entire test session."""
    db_manager = DatabaseManager()
    
    await db_manager.initialize()
    
    # Create test database if it doesn't exist
    async with db_manager.get_engine().begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
    
    yield db_manager
    
    await db_manager.cleanup()

@asynccontextmanager
async def get_clean_session(test_db_manager):
    """Context manager that provides a clean database session for each use.
    
    This can be used in tests that need fresh database state for each iteration,
    particularly useful with Hypothesis property-based tests.
    """
    # Ensure we're using the correct event loop
    loop = asyncio.get_event_loop()
    
    # Start a transaction
    engine = await test_db_manager.get_engine()
    connection = await engine.connect()
    transaction = await connection.begin()
    
    # Create session bound to this transaction
    session = AsyncSession(bind=connection, expire_on_commit=False)
    test_db_manager.set_test_session(session)
    
    try:
        yield session
    finally:
        test_db_manager.set_test_session(None)
        await session.close()
        await transaction.rollback()
        await connection.close()

# Keep this fixture for tests that don't use Hypothesis
@pytest.fixture(scope="function")
async def clean_session(test_db_manager, event_loop) -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a clean database session for regular tests (non-Hypothesis).
    Uses transaction rollback for fast cleanup.
    """
    async with get_clean_session(test_db_manager) as session:
        yield session
