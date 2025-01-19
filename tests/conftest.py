"""Pytest configuration file."""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from vector_rag.config import get_db_url
from vector_rag.embeddings import MockEmbedder


def create_database():
    """Create a test database if it doesn't exist."""
    template_db_url = get_db_url("postgres")
    test_db_url = get_db_url("vectordb_test")

    engine = create_engine(template_db_url)
    try:
        with engine.connect() as conn:
            # Disconnect all users from the database we will drop
            conn.execute(
                text(
                    f"""
                SELECT pg_terminate_backend(pid) 
                FROM pg_stat_activity 
                WHERE datname = 'vectordb_test'
                AND pid <> pg_backend_pid()
                """
                )
            )
            conn.execute(text("commit"))

            # Drop and recreate the database
            conn.execute(text("DROP DATABASE IF EXISTS vectordb_test"))
            conn.execute(text("commit"))
            conn.execute(text("CREATE DATABASE vectordb_test"))
            conn.execute(text("commit"))

            # Enable pgvector extension
            test_engine = create_engine(test_db_url)
            with test_engine.connect() as test_conn:
                test_conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                test_conn.execute(text("commit"))
    except OperationalError as e:
        pytest.fail(
            f"Failed to connect to database. Please check your environment variables and ensure the database is running: {e}"
        )


@pytest.fixture
def mock_embedder():
    """Create a mock embedder for testing."""
    return MockEmbedder(dimension=4)  # Small dimension for testing


@pytest.fixture(scope="function")
def test_db():
    """Create a test database."""
    # Connect to default postgres database to create/drop test database
    default_engine = create_engine(
        f"postgresql://postgres:postgres@localhost:5433/postgres"
    )

    test_db_name = "vectordb_test"

    # Drop test database if it exists and create it fresh
    with default_engine.connect() as conn:
        # Terminate existing connections
        conn.execute(
            text(
                f"""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = '{test_db_name}'
            AND pid <> pg_backend_pid()
            """
            )
        )
        conn.execute(text("commit"))

        # Drop and recreate database
        conn.execute(text(f"DROP DATABASE IF EXISTS {test_db_name}"))
        conn.execute(text("commit"))
        conn.execute(text(f"CREATE DATABASE {test_db_name}"))
        conn.execute(text("commit"))

    # Create engine for test database
    test_engine = create_engine(get_db_url(test_db_name))

    # Create vector extension and tables
    with test_engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("commit"))

        # Import and create tables
        from vector_rag.db.db_model import DbBase

        DbBase.metadata.create_all(test_engine)
        conn.execute(text("commit"))

    yield test_engine

    # Cleanup: drop test database
    with default_engine.connect() as conn:
        conn.execute(
            text(
                f"""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = '{test_db_name}'
            AND pid <> pg_backend_pid()
            """
            )
        )
        conn.execute(text("commit"))
        conn.execute(text(f"DROP DATABASE IF EXISTS {test_db_name}"))
        conn.execute(text("commit"))
