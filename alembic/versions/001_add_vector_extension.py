"""add vector extension

Revision ID: 001
Create Date: 2025-01-XX XX:XX:XX.XXXXXX

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Create vector extension if it doesn't exist
    op.execute('CREATE EXTENSION IF NOT EXISTS vector;')

def downgrade() -> None:
    # Note: We don't drop the extension in case other things depend on it
    pass
