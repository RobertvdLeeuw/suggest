import pytest

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
    
    original_create_url = db_manager.create_database_url
    
    def test_create_url():
        url = original_create_url()
        return url.set(database="test_collecter")
    
    db_manager.create_database_url = test_create_url
    
    await db_manager.initialize()
    
    # Create test database if it doesn't exist
    async with db_manager.get_engine().begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
    
    yield db_manager
    
    # Cleanup after all tests
    await db_manager.cleanup()

@pytest.fixture(scope="function")
async def clean_session(test_db_manager) -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a clean database session for each test function.
    Uses transaction rollback for fast cleanup.
    """
    # Start a transaction
    connection = await test_db_manager.get_engine().connect()
    transaction = await connection.begin()
    
    # Create session bound to this transaction
    session = AsyncSession(bind=connection, expire_on_commit=False)
    
    try:
        yield session
    finally:
        await session.close()
        # Rollback the transaction - this undoes all changes
        await transaction.rollback()
        await connection.close()
