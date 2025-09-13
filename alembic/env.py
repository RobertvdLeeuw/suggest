import os
import sys
from logging.config import fileConfig
from sqlalchemy import create_engine, pool
from alembic import context
from dotenv import load_dotenv

# Add your project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

# Import your models
from models import Base

# this is the Alembic Config object
config = context.config

# Set database URL from environment variables
def get_database_url():
    """Get sync database URL for Alembic (uses psycopg2, not asyncpg)"""
    if test_db := os.environ.get("TEST_DATABASE_NAME"):
        database_name = test_db
    elif os.getenv("TEST_MODE"):
        database_name = "test_db"
    else:
        database_name = os.environ.get("POSTGRES_DB", "db")
    
    # Use postgresql:// (sync) instead of postgresql+asyncpg:// (async)
    return f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_HOST']}:{os.environ['DB_PORT']}/{database_name}"

# Set the database URL
config.set_main_option("sqlalchemy.url", get_database_url())

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate support
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = create_engine(
        config.get_main_option("sqlalchemy.url"),
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
