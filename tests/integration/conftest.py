"""Pytest configuration file."""

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from rag.embeddings import MockEmbedder


@pytest.fixture
def mock_embedder():
    """Create a mock embedder for testing."""
    return MockEmbedder(dimension=4)  # Small dimension for testing


@pytest.fixture(scope="session")
def test_db():
    """Create a test database engine."""
    # Get database credentials from environment variables
    password = os.environ.get("POSTGRES_PASSWORD", "postgres").strip()
    username = os.environ.get("POSTGRES_USER", "postgres").strip()
    host = os.environ.get("POSTGRES_HOST", "localhost").strip()
    port = int(os.environ.get("POSTGRES_PORT", "5433").strip())
    database = "vectordb_test"
    db_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"

    print(f"POSTGRES_USER={username}")
    print(f"POSTGRES_PASSWORD={password}")
    print(f"POSTGRES_HOST={host}")
    print(f"POSTGRES_PORT={port}")
    print(f"Database URL={db_url}")

    # Create the engine with minimal pooling
    engine = create_engine(
        db_url, poolclass=NullPool, connect_args={"connect_timeout": 5}
    )

    # Drop all tables first to ensure clean state
    with engine.connect() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(
            text("CREATE EXTENSION IF NOT EXISTS vector")
        )  # Add pgvector extension
        conn.commit()

    # Create all tables
    from rag.db.models import Base

    Base.metadata.create_all(engine)

    yield engine

    # Drop all tables after tests
    Base.metadata.drop_all(engine)
