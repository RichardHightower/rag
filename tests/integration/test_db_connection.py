"""Simple script to test database connection."""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL

from rag.db.models import Base


def test_psycopg2_connection():
    """Test connection using psycopg2 directly."""
    print("\nTesting psycopg2 connection...")
    try:
        conn = psycopg2.connect(
            dbname="vectordb_test",
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5433")),
        )
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        print(f"psycopg2 connection successful: {result}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"psycopg2 connection failed: {str(e)}")


def test_sqlalchemy_connection():
    """Test connection using SQLAlchemy."""
    print("\nTesting SQLAlchemy connection...")
    try:
        # Use the same URL construction as conftest.py
        db_url = URL.create(
            drivername="postgresql",
            username=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5433")),
            database="vectordb_test",
        )
        print(f"Connection URL: {db_url}")
        engine = create_engine(db_url, echo=True)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            print(f"SQLAlchemy connection successful: {result}")
    except Exception as e:
        print(f"SQLAlchemy connection failed: {str(e)}")


def test_create_tables():
    """Test creating tables."""
    print("\nTesting table creation...")
    try:
        db_url = URL.create(
            drivername="postgresql",
            username=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5433")),
            database="vectordb_test",
        )
        engine = create_engine(db_url, echo=True)

        # Drop all tables first
        Base.metadata.drop_all(engine)
        print("Dropped all tables")

        # Create all tables
        Base.metadata.create_all(engine)
        print("Created all tables")

        # Verify tables exist
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """
                )
            ).fetchall()
            print("Tables created:", [r[0] for r in result])

    except Exception as e:
        print(f"Table creation failed: {str(e)}")


if __name__ == "__main__":
    test_psycopg2_connection()
    test_sqlalchemy_connection()
    test_create_tables()
