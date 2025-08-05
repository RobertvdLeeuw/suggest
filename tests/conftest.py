import pytest
import asyncio
from contextlib import asynccontextmanager
import sys
import os
from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseManager
from models import Base

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)

    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_db_manager(event_loop):
    """Create a test database manager for the entire test session."""
    DatabaseManager._instance = None
    DatabaseManager._engine = None
    DatabaseManager._session_factory = None
    DatabaseManager._initialized = False

    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    # Create test database if it doesn't exist
    engine = await db_manager.get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
    
    yield db_manager
    
    # Cleanup after all tests
    await db_manager.cleanup()
    DatabaseManager._instance = None

@asynccontextmanager
async def get_clean_session(test_db_manager):
    """Context manager that provides a clean database session for each use.
    
    This can be used in tests that need fresh database state for each iteration,
    particularly useful with Hypothesis property-based tests.
    """
    # Ensure we're using the correct event loop
    # loop = asyncio.get_event_loop()
    
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
