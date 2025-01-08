"""Test database file handler."""

import hashlib
import os
import tempfile
from logging import debug
from pathlib import Path

import pytest

from rag.config import TEST_DB_NAME, get_db_url
from rag.db.db_file_handler import DBFileHandler
from rag.embeddings.mock_embedder import MockEmbedder
from rag.model import File as FileModel


@pytest.fixture
def embedder():
    """Create a mock embedder for testing."""
    return MockEmbedder()


def create_test_file(content="Line 1\nLine 2\nLine 3\n"):
    """Create a FileModel instance for testing."""
    return FileModel(
        name="test.txt",
        path="/path/to/test.txt",
        crc=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        content=content,
        meta_data={},
    )


def test_create_project(test_db):
    """Test creating a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME))

    project = handler.create_project("Test Project", "Test Description")

    # Test project was created with correct attributes
    assert project.id is not None
    assert project.name == "Test Project"
    assert project.description == "Test Description"

    # Verify project exists in database
    with handler.session_scope() as session:
        db_project = session.get(handler.Project, project.id)
        assert db_project is not None
        assert db_project.name == "Test Project"
        assert db_project.description == "Test Description"


def test_get_project(test_db):
    """Test retrieving a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME))

    # Create a project
    created = handler.create_project("Test Project", "Test Description")

    # Get the project
    project = handler.get_project(created.id)
    assert project is not None
    assert project.id == created.id
    assert project.name == created.name
    assert project.description == created.description

    # Try to get a non-existent project
    assert handler.get_project(999) is None


def test_delete_project(test_db):
    """Test deleting a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME))

    # Create project
    project = handler.create_project("Test Project")
    assert project.id is not None

    # Verify project exists
    with handler.session_scope() as session:
        assert session.get(handler.Project, project.id) is not None

    # Delete project
    assert handler.delete_project(project.id) is True

    # Verify project is deleted
    with handler.session_scope() as session:
        assert session.get(handler.Project, project.id) is None


def test_delete_nonexistent_project(test_db):
    """Test deleting a project that doesn't exist."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME))
    assert handler.delete_project(999) is False


def test_add_file(test_db, embedder):
    """Test adding a file to a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)

    # Create project
    project = handler.create_project("Test Project")

    # Create test file
    file_model = create_test_file()

    debug(file_model)

    # Add file
    success = handler.add_file(project.id, file_model)
    assert success is not None

    # Verify file was added
    with handler.session_scope() as session:
        file = session.query(handler.File).filter_by(filename=file_model.name).first()
        assert file is not None
        assert file.project_id == project.id
        assert file.filename == file_model.name
        assert file.file_path == file_model.path
        assert file.created_at is not None

        # Verify chunks were created
        chunks = session.query(handler.Chunk).filter_by(file_id=file.id).all()
        assert len(chunks) > 0

        # Verify chunk content and embeddings
        for chunk in chunks:
            assert chunk.content is not None
            assert chunk.embedding is not None
            assert len(chunk.embedding) == embedder.get_dimension()


def test_add_file_to_nonexistent_project(test_db, embedder):
    """Test adding a file to a non-existent project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    file_model = create_test_file()

    # Try to add file to non-existent project
    success = handler.add_file(999, file_model)
    assert success is None


def test_remove_file(test_db, embedder):
    """Test removing a file from a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)

    # Create project and add file
    project = handler.create_project("Test Project")
    file_model = create_test_file()
    success = handler.add_file(project.id, file_model)
    assert success is not None

    # Get file ID
    with handler.session_scope() as session:
        file = session.query(handler.File).filter_by(filename=file_model.name).first()
        assert file is not None
        file_id = file.id

    # Remove file
    assert handler.remove_file(project.id, file_id) is True

    # Verify file is removed
    with handler.session_scope() as session:
        assert session.get(handler.File, file_id) is None


def test_remove_nonexistent_file(test_db, embedder):
    """Test removing a non-existent file."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)

    # Create project
    project = handler.create_project("Test Project")

    # Try to remove non-existent file
    assert handler.remove_file(project.id, 999) is False


def test_remove_file_wrong_project(test_db, embedder):
    """Test removing a file from the wrong project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)

    # Create two projects
    project1 = handler.create_project("Project 1")
    project2 = handler.create_project("Project 2")

    # Add file to project1
    file_model = create_test_file()
    file = handler.add_file(project1.id, file_model)
    assert file is not None

    # Get file ID
    with handler.session_scope() as session:
        file = session.query(handler.File).filter_by(filename=file_model.name).first()
        assert file is not None
        file_id = file.id

    # Try to remove file from project2
    assert handler.remove_file(project2.id, file_id) is False


def test_delete_file_success(test_db, embedder):
    """Test successful deletion of a file and its chunks."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME))

    # Create a project and add a file
    project = handler.create_project("Test Project")
    file_model = create_test_file()
    handler.add_file(project.id, file_model)

    # Get the file from DB to get its ID
    with handler.session_scope() as session:
        file = (
            session.query(handler.File)
            .filter(handler.File.file_path == file_model.path)
            .first()
        )
        file_id = file.id

        # Verify chunks exist
        chunks = (
            session.query(handler.Chunk).filter(handler.Chunk.file_id == file_id).all()
        )
        assert len(chunks) > 0

    # Delete the file
    result = handler.delete_file(file_id)
    assert result is True

    # Verify file and chunks are gone
    with handler.session_scope() as session:
        file = session.get(handler.File, file_id)
        assert file is None

        chunks = (
            session.query(handler.Chunk).filter(handler.Chunk.file_id == file_id).all()
        )
        assert len(chunks) == 0


def test_delete_nonexistent_file(test_db):
    """Test attempting to delete a non-existent file."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME))
    result = handler.delete_file(999999)  # Non-existent ID
    assert result is False


def test_get_file(test_db, embedder):
    """Test getting a file by project ID, path and name."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)

    # Create a project and add a file
    project = handler.create_project("Test Project")
    file_model = create_test_file("Test content")
    added_file = handler.add_file(project.id, file_model)
    assert added_file is not None

    # Test successful lookup
    found_file = handler.get_file(
        project_id=project.id, file_path=file_model.path, filename=file_model.name
    )
    assert found_file is not None
    assert found_file.id == added_file.id
    assert found_file.project_id == project.id
    assert found_file.filename == file_model.name
    assert found_file.file_path == file_model.path

    # Test lookup with wrong project ID
    wrong_project = handler.create_project("Wrong Project")
    not_found = handler.get_file(
        project_id=wrong_project.id, file_path=file_model.path, filename=file_model.name
    )
    assert not_found is None

    # Test lookup with wrong path
    not_found = handler.get_file(
        project_id=project.id, file_path="/wrong/path.txt", filename=file_model.name
    )
    assert not_found is None

    # Test lookup with wrong filename
    not_found = handler.get_file(
        project_id=project.id, file_path=file_model.path, filename="wrong.txt"
    )
    assert not_found is None


def teardown_module(module):
    """Clean up temporary files after tests."""
    # Clean up any .txt files in the current directory
    for item in Path().glob("*.txt"):
        if item.is_file() and item.suffix == ".txt":
            try:
                item.unlink()
            except OSError:
                pass
