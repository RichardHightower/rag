"""Test database file handler."""

import hashlib
import os
import tempfile
from logging import debug
from pathlib import Path

import pytest

from vector_rag.config import TEST_DB_NAME, get_db_url
from vector_rag.db.db_file_handler import DBFileHandler
from vector_rag.embeddings.mock_embedder import MockEmbedder
from vector_rag.model import File as FileModel


@pytest.fixture
def embedder():
    """Create a mock embedder for testing."""
    return MockEmbedder()


def create_test_file(content="Test content", name="test.txt", path="/path/to/test.txt"):
    """Create a test file model."""
    return FileModel(
        name=name,
        path=path,
        crc=str(hash(content)),  # Simple hash for testing
        content=content,
        meta_data={},
    )


def test_add_duplicate_file_same_crc(test_db):
    """Test adding the same file twice with same CRC."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), MockEmbedder())
    project = handler.create_project("Test Project")

    # Create and add file first time
    file_model = create_test_file()
    file1 = handler.add_file(project.id, file_model)
    assert file1 is not None

    # Add same file again
    file2 = handler.add_file(project.id, file_model)
    assert file2 is not None
    assert file2.id == file1.id  # Should return same file

    # Verify only one file exists in DB
    with handler.session_scope() as session:
        file_count = session.query(handler.File).count()
        assert file_count == 1

        chunk_count = session.query(handler.Chunk).count()
        assert chunk_count > 0  # Should have original chunks


def test_add_duplicate_file_different_project(test_db):
    """Test adding same file to different projects."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), MockEmbedder())

    # Create two projects
    project1 = handler.create_project("Project 1")
    project2 = handler.create_project("Project 2")

    # Create test file
    file_model = create_test_file()

    # Add to first project
    file1 = handler.add_file(project1.id, file_model)
    assert file1 is not None

    # Add to second project
    file2 = handler.add_file(project2.id, file_model)
    assert file2 is not None
    assert file2.id != file1.id  # Should be different files

    # Verify both files exist
    with handler.session_scope() as session:
        # Should be two files total
        file_count = session.query(handler.File).count()
        assert file_count == 2

        # Each project should have its chunks
        chunks1 = session.query(handler.Chunk).filter_by(file_id=file1.id).count()
        chunks2 = session.query(handler.Chunk).filter_by(file_id=file2.id).count()
        assert chunks1 > 0
        assert chunks2 > 0
        assert chunks1 == chunks2  # Same content, so same number of chunks


def test_add_duplicate_file_different_crc(test_db):
    """Test adding same file with different content (different CRC)."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), MockEmbedder())
    project = handler.create_project("Test Project")

    # Add original file
    original_content = "Original content"
    file1 = handler.add_file(project.id, create_test_file(original_content))
    assert file1 is not None
    file1_id = file1.id

    # Get original chunk count
    with handler.session_scope() as session:
        original_chunk_count = (
            session.query(handler.Chunk).filter_by(file_id=file1_id).count()
        )

    # Add modified version of same file
    modified_content = "Modified content"
    handler.add_file(
        project.id,
        create_test_file(modified_content, name="test.txt", path="/path/to/test.txt"),
    )

    # Verify database state
    with handler.session_scope() as session:
        # Should be exactly one file
        file_count = session.query(handler.File).count()
        assert file_count == 1

        # Get the current file and verify its content changed
        current_file = session.query(handler.File).first()
        assert current_file is not None

        # Verify chunks were updated
        current_chunks = session.query(handler.Chunk).all()
        assert len(current_chunks) > 0
        # At least one chunk should contain the new content
        assert any(modified_content in chunk.content for chunk in current_chunks)


def test_file_versioning_workflow(test_db):
    """Integration test for complete file versioning workflow."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), MockEmbedder())
    project = handler.create_project("Test Project")

    # Initial version
    content1 = "Version 1\nThis is the first version of the file."
    file1 = handler.add_file(project.id, create_test_file(content1))
    assert file1 is not None

    # Get initial chunk count
    with handler.session_scope() as session:
        chunks1 = session.query(handler.Chunk).filter_by(file_id=file1.id).all()
        chunk_count1 = len(chunks1)
        assert chunk_count1 > 0

    # Modified version
    content2 = "Version 2\nThis is the modified version with more content.\nExtra line."
    file2 = handler.add_file(
        project.id,
        create_test_file(content2, name="test.txt", path="/path/to/test.txt"),
    )
    assert file2 is not None
    assert file2.id != file1.id

    # Verify database state
    with handler.session_scope() as session:
        # Should only be one file
        files = session.query(handler.File).all()
        assert len(files) == 1

        # Check it's the new version
        assert files[0].id == file2.id
        assert files[0].crc == file2.crc

        # Verify chunks
        chunks2 = session.query(handler.Chunk).filter_by(file_id=file2.id).all()
        assert len(chunks2) > 0
        # More content should mean more or equal chunks
        assert len(chunks2) >= chunk_count1

        # Verify old chunks are gone
        old_chunks = session.query(handler.Chunk).filter_by(file_id=file1.id).all()
        assert len(old_chunks) == 0


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
    assert found_file.name == file_model.name
    assert found_file.path == file_model.path

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


def test_get_projects(test_db):
    """Test getting project listings."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME))

    # Create some test projects
    project_names = ["Project A", "Project B", "Project C"]
    created_projects = []

    for name in project_names:
        project = handler.create_project(name, f"Description for {name}")
        created_projects.append(project)

    # Test getting all projects
    all_projects = handler.get_projects()
    assert len(all_projects) == len(project_names)

    # Verify projects are ordered by creation date (newest first)
    for i in range(len(all_projects) - 1):
        assert all_projects[i].created_at >= all_projects[i + 1].created_at

    # Test limit
    limited_projects = handler.get_projects(limit=2)
    assert len(limited_projects) == 2
    assert limited_projects[0].name == project_names[-1]  # Most recent project

    # Test offset
    offset_projects = handler.get_projects(offset=1)
    assert len(offset_projects) == 2
    assert offset_projects[0].name == project_names[-2]  # Second most recent project

    # Test limit and offset together
    paged_projects = handler.get_projects(limit=1, offset=1)
    assert len(paged_projects) == 1
    assert paged_projects[0].name == project_names[-2]


def test_get_projects_empty(test_db):
    """Test getting project listings when there are no projects."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME))

    projects = handler.get_projects()
    assert len(projects) == 0

    # Test with limit and offset on empty database
    assert len(handler.get_projects(limit=10)) == 0
    assert len(handler.get_projects(offset=5)) == 0


def test_list_files(test_db, embedder):
    """Test listing files in a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)

    # Create a project
    project = handler.create_project("Test Project")
    assert project is not None

    # Create and add multiple files
    files_data = [
        ("file1.txt", "Content of file 1\nMore content"),
        ("file2.md", "# Markdown file\n## Section"),
        ("file3.py", "def hello():\n    print('Hello')"),
    ]

    original_files = []
    for filename, content in files_data:
        file_model = FileModel(
            name=filename,
            path=f"/test/path/{filename}",
            crc=str(hash(content)),
            content=content,
            meta_data={"type": filename.split(".")[-1]},
        )
        result = handler.add_file(project.id, file_model)
        assert result is not None
        original_files.append(file_model)

    # Test listing files
    listed_files = handler.list_files(project.id)

    # Verify the correct number of files
    assert len(listed_files) == len(files_data)

    # Verify each file matches the original
    for orig_file, listed_file in zip(original_files, listed_files):
        assert listed_file.name == orig_file.name
        assert listed_file.path == orig_file.path
        assert listed_file.crc == orig_file.crc
        assert listed_file.content == orig_file.content
        assert listed_file.meta_data["type"] == orig_file.meta_data["type"]
        assert listed_file.size == len(orig_file.content)

    # Test listing files for non-existent project
    empty_list = handler.list_files(999)
    assert len(empty_list) == 0

    # Test listing files for project with no files
    empty_project = handler.create_project("Empty Project")
    empty_result = handler.list_files(empty_project.id)
    assert len(empty_result) == 0

    # Test after deleting a file
    file_to_delete = handler.get_file(
        project_id=project.id,
        file_path=f"/test/path/{files_data[0][0]}",
        filename=files_data[0][0],
    )
    assert file_to_delete is not None
    handler.delete_file(file_to_delete.id)

    # Verify file was removed from list
    updated_files = handler.list_files(project.id)
    assert len(updated_files) == len(files_data) - 1
    assert all(f.name != files_data[0][0] for f in updated_files)

    # Test after deleting project
    handler.delete_project(project.id)
    final_list = handler.list_files(project.id)
    assert len(final_list) == 0
