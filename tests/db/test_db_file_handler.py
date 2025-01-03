"""Test database file handler."""

import os
import tempfile
from pathlib import Path

import pytest
from rag.db.db_file_handler import DBFileHandler
from rag.embeddings.mock_embedder import MockEmbedder
from rag.config import get_db_url, TEST_DB_NAME


@pytest.fixture
def embedder():
    """Create a mock embedder for testing."""
    return MockEmbedder()


@pytest.fixture
def sample_text_file():
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write('Line 1\nLine 2\nLine 3\n')
        return Path(f.name)


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


def test_add_file(test_db, embedder, sample_text_file):
    """Test adding a file to a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Create project
    project = handler.create_project("Test Project")
    
    # Add file
    file = handler.add_file(project.id, str(sample_text_file))
    
    # Verify file was added
    assert file is not None
    assert file.project_id == project.id
    assert file.filename == sample_text_file.name
    assert file.file_path == str(sample_text_file)
    assert file.created_at is not None
    
    # Verify chunks were created
    with handler.session_scope() as session:
        db_file = session.get(handler.File, file.id)
        assert db_file is not None
        assert len(db_file.chunks) > 0
        
        # Verify chunk content and embeddings
        for chunk in db_file.chunks:
            assert chunk.content is not None
            assert chunk.embedding is not None


def test_add_file_to_nonexistent_project(test_db, embedder, sample_text_file):
    """Test adding a file to a non-existent project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Try to add file to non-existent project
    file = handler.add_file(999, str(sample_text_file))
    assert file is None


def test_remove_file(test_db, embedder, sample_text_file):
    """Test removing a file from a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Create project and add file
    project = handler.create_project("Test Project")
    file = handler.add_file(project.id, str(sample_text_file))
    
    # Remove file
    assert handler.remove_file(project.id, file.id) is True
    
    # Verify file is removed
    with handler.session_scope() as session:
        assert session.get(handler.File, file.id) is None


def test_remove_nonexistent_file(test_db, embedder):
    """Test removing a non-existent file."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Create project
    project = handler.create_project("Test Project")
    
    # Try to remove non-existent file
    assert handler.remove_file(project.id, 999) is False


def test_remove_file_wrong_project(test_db, embedder, sample_text_file):
    """Test removing a file from the wrong project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Create two projects
    project1 = handler.create_project("Project 1")
    project2 = handler.create_project("Project 2")
    
    # Add file to project1
    file = handler.add_file(project1.id, str(sample_text_file))
    
    # Try to remove file from project2
    assert handler.remove_file(project2.id, file.id) is False


def test_embedder_dimension_handling(test_db, embedder):
    """Test embedder dimension handling."""
    # First handler initializes tables
    handler1 = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Second handler should work with same dimension
    handler2 = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    assert handler2 is not None


def teardown_module(module):
    """Clean up temporary files after tests."""
    # Clean up the temporary file
    for item in Path().glob('*.txt'):
        if item.is_file() and item.suffix == '.txt':
            try:
                item.unlink()
            except OSError:
                pass
