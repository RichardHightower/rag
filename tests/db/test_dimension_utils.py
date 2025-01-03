"""Test dimension utilities."""

import pytest
from sqlalchemy import text
from rag.db.dimension_utils import ensure_vector_dimension
from rag.config import get_db_url, TEST_DB_NAME


def test_ensure_vector_dimension_same(test_db):
    """Test when current dimension matches desired dimension."""
    engine = test_db
    
    # Set initial dimension
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE chunks
            ALTER COLUMN embedding TYPE vector(3)
            USING embedding::vector(3);
        """))
        conn.commit()
    
    # Call ensure_vector_dimension with same dimension
    ensure_vector_dimension(engine, 3)
    
    # Verify dimension is still 3
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass
            AND attname = 'embedding';
        """))
        assert result.scalar() == 3


def test_ensure_vector_dimension_different(test_db):
    """Test when current dimension differs from desired dimension."""
    engine = test_db
    
    # Set initial dimension
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE chunks
            ALTER COLUMN embedding TYPE vector(3)
            USING embedding::vector(3);
        """))
        conn.commit()
    
    # Change to different dimension
    ensure_vector_dimension(engine, 5)
    
    # Verify dimension was changed
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass
            AND attname = 'embedding';
        """))
        assert result.scalar() == 5
