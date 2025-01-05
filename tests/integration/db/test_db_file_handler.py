"""Test DB file handler."""

import tempfile
import uuid
from pathlib import Path

import pytest

from rag.config import TEST_DB_NAME, get_db_url
from rag.db.db_file_handler import DBFileHandler
from rag.embeddings.mock_embedder import MockEmbedder


@pytest.fixture
def embedder():
    """Create mock embedder for testing."""
    return MockEmbedder()


@pytest.fixture
def sample_text_file():
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Line 1\nLine 2\nLine 3\n")
        return Path(f.name)


def test_get_or_create_project(test_db, embedder):
    """Test creating a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    project = handler.get_or_create_project("Test Project", "Test Description")
    
    # Test project was created with correct attributes
    assert project.id is not None
    assert project.name == "Test Project"
    assert project.description == "Test Description"
    assert project.created_at is not None
    assert project.updated_at is not None
    
    # Verify project exists in database
    with handler.session_scope() as session:
        db_project = session.get(handler.Project, project.id)
        assert db_project is not None


def test_get_or_create_project_existing(test_db, embedder):
    """Test retrieving an existing project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)

    # Create a project
    created = handler.get_or_create_project("Test Project", "Test Description")
    
    # Get the same project
    project = handler.get_or_create_project("Test Project")
    assert project is not None
    assert project.id == created.id
    assert project.name == created.name
    assert project.description == created.description


def test_process_file(test_db, embedder, sample_text_file):
    """Test adding a file to a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Create project
    project = handler.get_or_create_project("Test Project")
    
    # Add file
    with open(sample_text_file, 'r') as f:
        content = f.read()
    file, is_updated = handler.process_file(project.id, str(sample_text_file), content)
    
    # Verify file was added
    assert file is not None
    assert file.project_id == project.id
    assert file.filename == sample_text_file.name
    assert file.file_path == str(sample_text_file)
    assert file.created_at is not None
    assert not is_updated
    
    # Verify chunks were created
    with handler.session_scope() as session:
        db_file = session.get(handler.File, file.id)
        assert db_file is not None
        assert len(db_file.chunks) > 0
        
        # Verify chunk content and embeddings
        for chunk in db_file.chunks:
            assert chunk.content is not None
            assert chunk.embedding is not None


def test_process_file_nonexistent_project(test_db, embedder, sample_text_file):
    """Test adding a file to a non-existent project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Try to add file to non-existent project
    with open(sample_text_file, 'r') as f:
        content = f.read()
    file, is_updated = handler.process_file(999, str(sample_text_file), content)
    assert file is None
    assert not is_updated


def test_file_deduplication(test_db, embedder, sample_text_file):
    """Test file deduplication."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Create project and add file
    project = handler.get_or_create_project("Test Project")
    with open(sample_text_file, 'r') as f:
        content = f.read()
    
    # Add same file twice
    file1, is_updated1 = handler.process_file(project.id, str(sample_text_file), content)
    file2, is_updated2 = handler.process_file(project.id, str(sample_text_file), content)
    
    assert file1.id == file2.id  # Same file
    assert not is_updated1  # First time not an update
    assert not is_updated2  # Second time no changes, so no update


def test_file_update(test_db, embedder, sample_text_file):
    """Test updating a file."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Create project and add file
    project = handler.get_or_create_project("Test Project")
    with open(sample_text_file, 'r') as f:
        content = f.read()
    
    # Add file first time
    file1, is_updated1 = handler.process_file(project.id, str(sample_text_file), content)
    
    # Update with modified content
    modified_content = content + "\nNew line"
    file2, is_updated2 = handler.process_file(project.id, str(sample_text_file), modified_content)
    
    assert file1.id == file2.id  # Same file
    assert not is_updated1  # First time not an update
    assert is_updated2  # Second time with changes is an update


def teardown_module(module):
    """Clean up temporary files after tests."""
    # Clean up the temporary file
    for item in Path().glob("*.txt"):
        if item.is_file() and item.suffix == ".txt":
            try:
                item.unlink()
            except OSError:
                pass
