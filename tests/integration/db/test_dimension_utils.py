"""Test dimension utilities."""

import pytest
from sqlalchemy import text

from rag.config import TEST_DB_NAME, get_db_url
from rag.db.dimension_utils import ensure_vector_dimension


def test_ensure_vector_dimension_same(test_db):
    """Test when current dimension matches desired dimension."""
    engine = test_db

    # Set initial dimension
    with engine.connect() as conn:
        conn.execute(
            text(
                """
            ALTER TABLE chunks
            ALTER COLUMN embedding TYPE vector(1536)
            USING embedding::vector(1536);
        """
            )
        )
        conn.commit()

    # Call ensure_vector_dimension with same dimension
    ensure_vector_dimension(engine, 1536)

    # Verify dimension is still 1536
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass
            AND attname = 'embedding';
        """
            )
        )
        assert result.scalar() == 1536


def test_ensure_vector_dimension_different(test_db):
    """Test when current dimension differs from desired dimension."""
    engine = test_db

    # Set initial dimension
    with engine.connect() as conn:
        conn.execute(
            text(
                """
            ALTER TABLE chunks
            ALTER COLUMN embedding TYPE vector(1536)
            USING embedding::vector(1536);
        """
            )
        )
        conn.commit()

    # Change to new dimension
    ensure_vector_dimension(engine, 512)

    # Verify dimension is updated
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass
            AND attname = 'embedding';
        """
            )
        )
        assert result.scalar() == 512
