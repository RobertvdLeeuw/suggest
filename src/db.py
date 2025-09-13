import os
import sys
import asyncio
import contextlib
from typing import Optional, AsyncGenerator, Dict

from sqlalchemy import URL, text, event, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from sqlalchemy.events import PoolEvents

import traceback
import logging
LOGGER = logging.getLogger(__name__)

import numpy as np

from models import Base

from dotenv import load_dotenv
load_dotenv()

class DatabaseManager:
    """Process-local singleton database manager with proper connection handling"""
    
    _instances: Dict[int, 'DatabaseManager'] = {}  # keyed by process ID
    
    def __new__(cls) -> 'DatabaseManager':
        pid = os.getpid()
        if pid not in cls._instances:
            instance = super().__new__(cls)
            # Initialize instance attributes
            instance._engine: Optional[AsyncEngine] = None
            instance._session_factory: Optional[async_sessionmaker] = None
            instance._initialized: bool = False
            cls._instances[pid] = instance
        return cls._instances[pid]
    
    def create_database_url(self) -> URL:
        if test_db := os.environ.get("TEST_DATABASE_NAME"):
            database_name = test_db
        elif os.getenv("TEST_MODE"):
            database_name = "test_db"
        else:
            database_name = os.environ.get("POSTGRES_DB", "db")

        LOGGER.info(f"Using database '{database_name}' (PID: {os.getpid()}).")

        return URL.create(
            drivername='postgresql+asyncpg',
            username=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            host=os.environ["POSTGRES_HOST"],
            port=int(os.environ["DB_PORT"]),
            database=database_name
        )
    
    async def initialize(self) -> None:
        """Initialize database engine and session factory with proper configuration"""
        if self._initialized:
            LOGGER.debug(f"Database already initialized for PID {os.getpid()}")
            return
            
        LOGGER.info(f"Initializing DB engine for PID {os.getpid()}")
        try:
            url = self.create_database_url()
            # Enhanced engine configuration
            self._engine = create_async_engine(
                url,
                # Connection pool settings
                poolclass=AsyncAdaptedQueuePool,
                pool_size=10,                    # Number of connections to maintain
                max_overflow=20,                 # Additional connections allowed
                pool_pre_ping=True,              # Validate connections before use
                pool_recycle=3600,               # Recycle connections every hour
                pool_timeout=30,                 # Timeout for getting connection
                
                # Engine settings
                echo=False,                      # Set to True for SQL debugging
                future=True,
                
                # Connection arguments for asyncpg
                connect_args={
                    "server_settings": {
                        "application_name": f"audio_processing_service_pid_{os.getpid()}",
                        "jit": "off"  # Disable JIT for better performance on small queries
                    },
                    "command_timeout": 60,
                    "statement_cache_size": 0,  # Disable statement cache if having issues
                }
            )
            
            # Add connection pool event listeners for monitoring
            @event.listens_for(self._engine.sync_engine, "connect")
            def receive_connect(dbapi_connection, connection_record):
                LOGGER.debug(f"New database connection established (PID: {os.getpid()})")
            
            @event.listens_for(self._engine.sync_engine, "checkout")
            def receive_checkout(dbapi_connection, connection_record, connection_proxy):
                LOGGER.debug(f"Connection checked out from pool (PID: {os.getpid()})")
            
            # Test the connection
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
                LOGGER.info(f"Database connection test to '{url.database}' successful (PID: {os.getpid()})")
            
            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,          # Keep objects usable after commit
                autoflush=True,                  # Auto-flush before queries
                autocommit=False                 # Explicit transaction control
            )
            
            self._initialized = True
            LOGGER.info(f"Database engine and session factory initialized successfully (PID: {os.getpid()})")
            
        except Exception as e:
            LOGGER.error(f"Could not initialize database (PID: {os.getpid()}): {traceback.format_exc()}")
            await self.cleanup()
            raise Exception(f"Database initialization failed: {str(e)}")

    @contextlib.asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Context manager for database sessions with proper cleanup"""
        if not self._initialized:
            LOGGER.info(f"DB not initialized yet for PID {os.getpid()}, doing that now.")
            await self.initialize()
        
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            LOGGER.error(f"Session error, rolling back (PID: {os.getpid()}): {traceback.format_exc()}")
            raise
        finally:
            await session.close()
    
    async def get_engine(self) -> AsyncEngine:
        """Get the database engine"""
        if not self._initialized:
            LOGGER.info(f"DB not initialized yet for PID {os.getpid()}, doing that now.")
            await self.initialize()
        return self._engine
    
    async def cleanup(self) -> None:
        """Cleanup database resources for this process"""
        if self._engine:
            await self._engine.dispose()
            LOGGER.info(f"Database engine disposed (PID: {os.getpid()})")
        
        self._engine = None
        self._session_factory = None
        self._initialized = False
        
        # Remove this process's instance from the class dict
        pid = os.getpid()
        if pid in self._instances:
            del self._instances[pid]

    async def create_tables_with_alembic(self) -> None:
        """Create tables using Alembic migrations instead of direct creation"""
        import subprocess
        import sys
        
        await self.initialize()
        
        # Run alembic upgrade
        try:
            result = subprocess.run([
                sys.executable, "-m", "alembic", "upgrade", "head"
            ], check=True, capture_output=True, text=True)
            LOGGER.info(f"Alembic upgrade completed: {result.stdout}")
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Alembic upgrade failed: {e.stderr}")
            raise

    async def setup_tables(self) -> None:
        """Setup database tables using Alembic (safer than direct creation)"""
        await self.initialize()
        
        # For development/testing - you might still want the nuclear option
        if os.getenv("FORCE_RECREATE_TABLES"):
            async with self._engine.begin() as conn:
                await conn.execute(text("DROP SCHEMA public CASCADE;"))
                await conn.execute(text("CREATE SCHEMA public;"))
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        await self.create_tables_with_alembic()
        
        LOGGER.info(f"Database setup completed (PID: {os.getpid()})")


    @classmethod
    async def cleanup_all_instances(cls) -> None:
        """Cleanup all database instances across all processes (for test cleanup)"""
        for pid, instance in list(cls._instances.items()):
            await instance.cleanup()
        cls._instances.clear()

# Global instance getter
_db_manager = None

def get_db_manager():
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def get_session():
    """Get session context manager"""
    return get_db_manager().get_session()

async def get_embeddings(emb_type) -> np.ndarray:  # Structured array (basically a better dict.)
    async with get_session() as s:
        results = await s.execute(select(emb_type))
        items = results.scalars().all()

    embedding_dim = len(items[0].embedding) if results else 0
    dtype = [
        ('song_id', 'i4'),
        ('chunk_id', 'i4'), 
        ('embedding', f'f4', (embedding_dim,))
    ]

    data = np.fromiter(
        ((x.song_id, x.chunk_id, x.embedding) for x in items),
        dtype=dtype
    )

    return data

if __name__ == "__main__":
    async def main():
        LOGGER.info("Setting up tables.")
        
        prompt = "This will remove all PROD tables and data from the DB. Are you sure? "
        if "-t" in sys.argv or "--test" in sys.argv:
            os.environ["TEST_MODE"] = "true"
            prompt = "This will remove all TEST tables and data from the DB. Are you sure? "

        if not input(prompt).lower().startswith("y"):
            return
        
        try:
            await get_db_manager().setup_tables()
        finally:
            await get_db_manager().cleanup()
    
    asyncio.run(main())
