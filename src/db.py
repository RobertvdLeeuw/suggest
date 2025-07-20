import os
import asyncio
import contextlib
from typing import Optional, AsyncGenerator

from sqlalchemy import URL, text, event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from sqlalchemy.events import PoolEvents

from logger import LOGGER
import traceback

from models import Base

from dotenv import load_dotenv
load_dotenv()

class DatabaseManager:
    """Singleton database manager with proper connection handling"""
    
    _instance: Optional['DatabaseManager'] = None
    _engine: Optional[AsyncEngine] = None
    _session_factory: Optional[async_sessionmaker] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'DatabaseManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def create_database_url(self) -> URL:
        # Add database name and ensure all required components
        return URL.create(
            drivername='postgresql+asyncpg',
            username=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            host=os.environ["POSTGRES_HOST"],
            port=int(os.environ["DB_PORT"]),
            database=os.environ.get("POSTGRES_DB", "postgres")  # Add database name
        )
    
    async def initialize(self) -> None:
        """Initialize database engine and session factory with proper configuration"""
        if self._initialized:
            LOGGER.debug("Database already initialized")
            return
            
        try:
            # Enhanced engine configuration
            self._engine = create_async_engine(
                self.create_database_url(),
                # Connection pool settings
                poolclass= AsyncAdaptedQueuePool,
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
                        "application_name": "audio_processing_service",
                        "jit": "off"  # Disable JIT for better performance on small queries
                    },
                    "command_timeout": 60,
                    "statement_cache_size": 0,  # Disable statement cache if having issues
                }
            )
            
            # Add connection pool event listeners for monitoring
            @event.listens_for(self._engine.sync_engine, "connect")
            def receive_connect(dbapi_connection, connection_record):
                LOGGER.debug("New database connection established")
            
            @event.listens_for(self._engine.sync_engine, "checkout")
            def receive_checkout(dbapi_connection, connection_record, connection_proxy):
                LOGGER.debug("Connection checked out from pool")
            
            # Test the connection
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
                LOGGER.info("Database connection test successful")
            
            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,          # Keep objects usable after commit
                autoflush=True,                  # Auto-flush before queries
                autocommit=False                 # Explicit transaction control
            )
            
            self._initialized = True
            LOGGER.info("Database engine and session factory initialized successfully")
            
        except Exception as e:
            LOGGER.error(f"Could not initialize database: {traceback.format_exc()}")
            await self.cleanup()
            raise Exception(f"Database initialization failed: {str(e)}")
    
    @contextlib.asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Context manager for database sessions with proper cleanup"""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            LOGGER.error(f"Session error, rolling back: {str(e)}")
            raise
        finally:
            await session.close()
    
    async def get_engine(self) -> AsyncEngine:
        """Get the database engine"""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._engine
    
    async def cleanup(self) -> None:
        """Cleanup database resources"""
        if self._engine:
            await self._engine.dispose()
            LOGGER.info("Database engine disposed")
        
        self._engine = None
        self._session_factory = None
        self._initialized = False
    
    async def setup_tables(self) -> None:
        """Setup database tables (dangerous - drops all data!)"""
        await self.initialize()
        
        async with self._engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE;"))
            await conn.execute(text("CREATE SCHEMA public;"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.run_sync(Base.metadata.create_all)
        
        LOGGER.info("Database tables created successfully")

# Global instance
db_manager = DatabaseManager()

# Convenience functions that maintain backward compatibility
async def init_db():
    """Initialize database - wrapper for backward compatibility"""
    await db_manager.initialize()

async def setup_tables():
    """Setup tables - wrapper for backward compatibility"""
    await db_manager.setup_tables()

def get_session():
    """Get session context manager"""
    return db_manager.get_session()

async def setup():
    """Setup database connection"""
    LOGGER.info("Initializing DB engine.")
    await db_manager.initialize()
    LOGGER.info("DB connection initialized.")

# Graceful shutdown function
async def shutdown_db():
    """Gracefully shutdown database connections"""
    await db_manager.cleanup()

if __name__ == "__main__":
    async def main():
        LOGGER.info("Setting up tables.")
        
        if not input("This will remove all tables and data from the DB. Are you sure? ").lower().startswith("y"):
            return
        
        try:
            await setup_tables()
        finally:
            await shutdown_db()
    
    asyncio.run(main())
