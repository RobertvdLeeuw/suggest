import pytest
import asyncio
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
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_db_manager():
    """Create a test database manager for the entire test session."""
    DatabaseManager._instance = None
    DatabaseManager._engine = None
    DatabaseManager._session_factory = None
    DatabaseManager._initialized = False

    db_manager = DatabaseManager()
    
    original_create_url = db_manager.create_database_url
    
    def test_create_url():
        url = original_create_url()
        return url.set(database="test_db")
    
    db_manager.create_database_url = test_create_url
    
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

@pytest.fixture(scope="function")
async def clean_session(test_db_manager) -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a clean database session for each test function.
    Uses transaction rollback for fast cleanup.
    """
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
