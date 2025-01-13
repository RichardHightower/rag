import pytest

from rag.config import get_db_url
from rag.db.db_file_handler import DBFileHandler
from rag.embeddings import MockEmbedder
from rag.model import Chunk, File


@pytest.fixture
def db_handler(test_db):
    """Create a DBFileHandler with test database."""
    return DBFileHandler(get_db_url("vectordb_test"), embedder=MockEmbedder())


def test_add_chunk(db_handler):
    # Create a project
    project = db_handler.create_project("Test Project")

    # Create a file
    file_model = File(name="test.txt", path="/test/test.txt", crc="123", content="Test")
    file = db_handler.add_file(project.id, file_model)

    # Create a chunk
    chunk = Chunk(target_size=100, content="Test chunk content", index=0)

    # Add the chunk
    result = db_handler.add_chunk(file.id, chunk)

    # Assert the chunk was added successfully
    assert result is not None
    assert isinstance(result, Chunk)
    assert result.content == chunk.content
    assert result.index == chunk.index


def test_add_chunk_nonexistent_file(db_handler):
    # Try to add a chunk to a non-existent file
    chunk = Chunk(target_size=100, content="Test chunk content", index=0)
    result = db_handler.add_chunk(999, chunk)

    # Assert the operation failed
    assert result is None


def test_add_chunk_with_metadata(db_handler):
    # Create a project
    project = db_handler.create_project("Test Project")

    # Create a file
    file_model = File(
        name="test.txt", path="/test/test.txt", crc="123", content="Test content"
    )
    file = db_handler.add_file(project.id, file_model)

    # Create a chunk with metadata
    metadata = {"source": "test_file", "page": 1, "importance": "high"}
    chunk = Chunk(
        target_size=100, content="Test chunk content", index=0, meta_data=metadata
    )

    # Add the chunk
    result = db_handler.add_chunk(file.id, chunk)

    # Assert the chunk was added successfully
    assert result is not None
    assert isinstance(result, Chunk)
    assert result.content == chunk.content
    assert result.index == chunk.index
    assert result.meta_data == metadata
