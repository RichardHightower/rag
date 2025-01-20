"""Integration test for file ingestion."""

import uuid
from pathlib import Path

import pytest

from vector_rag.config import Config
from vector_rag.db.db_file_handler import DBFileHandler
from vector_rag.embeddings.mock_embedder import MockEmbedder
from vector_rag.model import File

config = Config()
DB_URL = config.DB_URL


@pytest.fixture
def sample_file():
    """Create a temporary sample file."""
    content = "This is a test file.\nIt has multiple lines.\nEach line will be chunked."
    return File(
        name="test.txt", path="/path/to/test.txt", crc="test123", content=content
    )


@pytest.fixture
def unique_name():
    """Generate a unique project name."""
    return f"Test Project {uuid.uuid4()}"


def test_file_ingestion(test_db, sample_file, unique_name):
    """Test complete file ingestion flow."""
    # Create handler with mock embedder
    handler = DBFileHandler(config, MockEmbedder())

    # Create project
    project = handler.create_project(unique_name)
    assert project is not None
    assert project.name == unique_name

    # Add file
    file = handler.add_file(project.id, sample_file)
    assert file is not None

    # Verify chunks were created
    with handler.session_scope() as session:
        # Count chunks
        chunk_count = session.query(handler.Chunk).filter_by(file_id=file.id).count()
        assert chunk_count > 0

        # Verify embeddings exist
        chunks = session.query(handler.Chunk).filter_by(file_id=file.id).all()
        for chunk in chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == MockEmbedder().get_dimension()


def test_project_uniqueness(test_db, unique_name):
    """Test project name uniqueness constraints."""
    handler = DBFileHandler(config, MockEmbedder())

    # Create initial project
    project1 = handler.create_project(unique_name)
    assert project1 is not None
    assert project1.name == unique_name

    # Try to create another project with the same name
    with pytest.raises(
        ValueError, match=f"Project with name '{unique_name}' already exists"
    ):
        handler.create_project(unique_name)

    # Test get_or_create_project
    project2 = handler.get_or_create_project(unique_name)
    assert project2.id == project1.id
    assert project2.name == unique_name

    # Test description update
    new_description = "Updated description"
    project3 = handler.get_or_create_project(unique_name, new_description)
    assert project3.id == project1.id
    assert project3.name == unique_name
    assert project3.description == new_description


def test_multiple_projects(test_db):
    """Test creating multiple projects with different names."""
    handler = DBFileHandler(config, MockEmbedder())

    # Create multiple projects
    names = [f"Project {i}" for i in range(3)]
    projects = []

    for name in names:
        project = handler.create_project(name)
        assert project is not None
        assert project.name == name
        projects.append(project)

    # Verify all projects exist and have unique IDs
    project_ids = {p.id for p in projects}
    assert len(project_ids) == len(names)

    for project in projects:
        handler.delete_project(project.id)


def teardown_module(module):
    """Clean up temporary files after tests."""
    # Clean up any .txt files in the current directory
    for item in Path().glob("*.txt"):
        if item.is_file() and item.suffix == ".txt":
            try:
                item.unlink()
            except OSError:
                pass
