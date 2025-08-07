import pytest
import asyncio
from contextlib import asynccontextmanager
import sys
import os
import uuid
from typing import AsyncGenerator
from sqlalchemy import text, URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import Base

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def unique_test_db_name():
    """Generate a unique database name for this test."""
    # Include test node info for parallel test support
    test_id = str(uuid.uuid4())[:8]
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'main')
    return f"test_db_{worker_id}_{test_id}"


@pytest.fixture(scope="function") 
async def admin_engine():
    """Create an engine connected to the default postgres database for admin operations."""
    admin_url = URL.create(
        drivername='postgresql+asyncpg',
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["DB_PORT"]),
        database='postgres'  # Connect to default postgres db for admin operations
    )
    
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    yield engine
    await engine.dispose()


@asynccontextmanager
async def get_isolated_test_db():
    """Context manager that creates an isolated test database for each use.
    
    This is designed for use with Hypothesis property-based tests where each
    generated test case needs its own fresh database.
    """
    # Generate unique database name
    test_id = str(uuid.uuid4())[:8]
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'main')
    db_name = f"test_db_{worker_id}_{test_id}"
    
    # Set environment variable so production code uses test database
    old_test_db = os.environ.get("TEST_DATABASE_NAME")
    os.environ["TEST_DATABASE_NAME"] = db_name
    
    # Create admin engine for database operations
    admin_url = URL.create(
        drivername='postgresql+asyncpg',
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["DB_PORT"]),
        database='postgres'
    )
    
    admin_engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    
    try:
        # Create the test database
        async with admin_engine.connect() as conn:
            # Terminate any existing connections to the database
            await conn.execute(text(f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = '{db_name}' AND pid <> pg_backend_pid()
            """))
            
            # Drop database if it exists (cleanup from failed previous run)
            await conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))
            
            # Create new database
            await conn.execute(text(f"CREATE DATABASE {db_name}"))
        
        # Import db_manager AFTER setting the environment variable
        from db import db_manager, DatabaseManager
        
        # Reset the global instance to pick up the new database name
        await db_manager.cleanup()
        DatabaseManager._instance = None
        DatabaseManager._initialized = False
        
        # Initialize with the test database
        await db_manager.initialize()
        
        # Set up the database schema
        engine = await db_manager.get_engine()
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.run_sync(Base.metadata.create_all)
        
        yield db_manager
        
    finally:
        # Cleanup: close all connections and drop the database
        from collecter.embedders import end_processes
        end_processes()
        await db_manager.cleanup()
        
        # Restore environment variable
        if old_test_db is not None:
            os.environ["TEST_DATABASE_NAME"] = old_test_db
        else:
            os.environ.pop("TEST_DATABASE_NAME", None)
        
        # Drop the test database
        async with admin_engine.connect() as conn:
            # Terminate any remaining connections
            await conn.execute(text(f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = '{db_name}' AND pid <> pg_backend_pid()
            """))
            
            # Drop the database
            await conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))
        
        await admin_engine.dispose()

