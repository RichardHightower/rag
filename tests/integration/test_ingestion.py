"""Integration tests for file ingestion (using the same connection style as test_db_connection)."""

import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

# Load environment variables
env_file = Path(project_root) / ".env"
load_dotenv(env_file)

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.engine import URL

from rag.db.models import Project, File, Chunk
from rag.db.db_file_handler import DBFileHandler
from rag.embeddings.mock_embedder import MockEmbedder


def _get_db_url() -> str:
    """
    Build a DB URL string using environment variables,
    matching the style from test_db_connection.py
    """
    password = os.environ.get("POSTGRES_PASSWORD", "postgres").strip()
    username = os.environ.get("POSTGRES_USER", "postgres").strip()
    host = os.environ.get("POSTGRES_HOST", "localhost").strip()
    port = int(os.environ.get("POSTGRES_PORT", "5433").strip())
    database = "vectordb_test"
    db_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"


    print(f"test ingestion POSTGRES_USER={username}")
    print(f"test ingestion POSTGRES_PASSWORD={password}")
    print(f"test ingestion POSTGRES_HOST={host}")
    print(f"test ingestion POSTGRES_PORT={port}")
    print(f"test ingestion Database URL={db_url}")
    db_url2 = str(
        URL.create(
            drivername="postgresql",
            username=username,
            password=password,
            host=host,
            port=port,
            database=database,
        )
    )
    print(f"test ingestion Database URL={db_url2}")
    return db_url


def test_file_ingestion():
    """Test basic file ingestion flow."""
    db_url = _get_db_url()
    handler = DBFileHandler(db_url, MockEmbedder())

    # Create project
    project = handler.get_or_create_project(f"test_project_{uuid.uuid4()}")

    # Test file ingestion
    content = "This is a test file.\nIt has multiple lines.\nEach line is different."
    file, is_updated = handler.process_file(project.id, "/test/file.txt", content)

    assert not is_updated
    assert file.filename == "file.txt"
    assert file.file_path == "/test/file.txt"

    # Verify chunks were created
    engine = create_engine(db_url)
    with Session(engine) as session:
        chunks = session.query(Chunk).filter(Chunk.file_id == file.id).all()
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.embedding is not None


def test_project_uniqueness():
    """Test project name uniqueness."""
    db_url = _get_db_url()
    handler = DBFileHandler(db_url, MockEmbedder())

    # Create project with unique name
    name = f"test_project_{uuid.uuid4()}"
    project1 = handler.get_or_create_project(name)

    # Try to create project with the same name
    project2 = handler.get_or_create_project(name)

    # Should return the same project
    assert project1.id == project2.id
    assert project1.name == project2.name


def test_multiple_projects():
    """Test creating multiple projects."""
    db_url = _get_db_url()
    handler = DBFileHandler(db_url, MockEmbedder())

    # Create multiple projects with unique names
    names = [f"test_project_{uuid.uuid4()}" for _ in range(3)]
    projects = [handler.get_or_create_project(name) for name in names]

    # Verify all projects were created with unique IDs
    project_ids = [p.id for p in projects]
    assert len(project_ids) == len(set(project_ids))
    assert len(project_ids) == len(names)


def test_file_deduplication():
    """Test file deduplication logic."""
    db_url = _get_db_url()
    handler = DBFileHandler(db_url, MockEmbedder())
    project = handler.get_or_create_project(f"test_project_{uuid.uuid4()}")

    # Initial file ingestion
    content1 = "Initial content"
    file1, is_updated = handler.process_file(project.id, "/test/file.txt", content1)
    assert not is_updated

    # Same content - should not update
    file2, is_updated = handler.process_file(project.id, "/test/file.txt", content1)
    assert not is_updated
    assert file1.id == file2.id
    assert file1.crc == file2.crc

    # Modified content - should update
    content2 = "Modified content"
    file3, is_updated = handler.process_file(project.id, "/test/file.txt", content2)
    assert is_updated
    assert file1.id == file3.id
    assert file1.crc != file3.crc

    # Verify chunk update
    engine = create_engine(db_url)
    with Session(engine) as session:
        chunks = session.query(Chunk).filter(Chunk.file_id == file3.id).all()
        assert len(chunks) > 0
        for chunk in chunks:
            assert "Modified" in chunk.content


def test_file_path_uniqueness():
    """Test file path uniqueness constraints."""
    db_url = _get_db_url()
    handler = DBFileHandler(db_url, MockEmbedder())

    # Create two projects
    project1 = handler.get_or_create_project(f"test_project_{uuid.uuid4()}")
    project2 = handler.get_or_create_project(f"test_project_{uuid.uuid4()}")

    # Same file path in different projects - should work
    content = "Test content"
    file1, _ = handler.process_file(project1.id, "/test/file_unique.txt", content)
    file2, _ = handler.process_file(project2.id, "/test/file_unique.txt", content)

    assert file1.id != file2.id

    # Verify both files exist
    engine = create_engine(db_url)
    with Session(engine) as session:
        files = session.query(File).filter(File.file_path == "/test/file_unique.txt").all()
        assert len(files) == 2
