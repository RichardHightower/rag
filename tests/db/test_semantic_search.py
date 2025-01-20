"""Test semantic search functionality."""

import numpy as np
import pytest
from sqlalchemy import text

from vector_rag.chunking import LineChunker
from vector_rag.config import Config
from vector_rag.db.db_file_handler import DBFileHandler
from vector_rag.db.dimension_utils import ensure_vector_dimension
from vector_rag.embeddings import MockEmbedder
from vector_rag.model import File

config = Config()
EMBEDDING_DIM = config.EMBEDDING_DIM
TEST_DB_NAME = config.TEST_DB_NAME
TEST_DB_URL = config.TEST_DB_URL


@pytest.fixture
def mock_embedder():
    """Create a mock embedder with consistent embeddings for testing."""

    class TestEmbedder(MockEmbedder):
        def embed_texts(self, texts):
            # Return predictable embeddings with correct dtype and shape
            return [
                np.array(
                    [float(len(t.content)) / 100] * self.dimension, dtype=">f4"
                )  # big-endian float32
                for t in texts
            ]

    return TestEmbedder(dimension=EMBEDDING_DIM)


@pytest.fixture
def test_files():
    """Create test files with different content lengths."""
    return [
        File(
            name=f"test{i}.txt",
            path=f"/path/to/test{i}.txt",
            crc=f"crc{i}",
            content=f"Test content {'x\n' * (i * 10)}" * (i + 1),
            meta_data={"type": "test"},
        )
        for i in range(5)  # Creates 5 files of increasing size
    ]


@pytest.fixture
def populated_handler(test_db, mock_embedder, test_files):
    """Create a handler with test data."""
    # Ensure vector dimensions match
    ensure_vector_dimension(test_db, EMBEDDING_DIM)

    handler = DBFileHandler.create(
        TEST_DB_NAME, mock_embedder, chunker=LineChunker.create(5, 0)
    )
    project = handler.create_project("Test Project")

    # Add test files
    for file in test_files:
        file_record = handler.add_file(project.id, file)
        assert file_record is not None

    return handler, project.id


def test_search_by_text_basic(populated_handler):
    """Test basic text search functionality."""
    handler, project_id = populated_handler

    results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        page=1,
        page_size=10,
        similarity_threshold=0.0,  # Accept all results for testing
    )

    assert results is not None
    assert len(results.results) > 0
    assert all(0 <= r.score <= 1 for r in results.results)
    assert results.total_count > 0
    assert results.page == 1


def test_search_by_embedding_basic(populated_handler, mock_embedder):
    """Test basic embedding search functionality."""
    handler, project_id = populated_handler

    # Create a test embedding with correct dtype
    test_embedding = np.array([0.5] * mock_embedder.get_dimension(), dtype=">f4")

    results = handler.search_chunks_by_embedding(
        project_id=project_id,
        embedding=test_embedding,
        page=1,
        page_size=10,
        similarity_threshold=0.0,
    )

    assert results is not None
    assert len(results.results) > 0
    assert all(0 <= r.score <= 1 for r in results.results)


def test_pagination(populated_handler):
    """Test pagination functionality."""
    handler, project_id = populated_handler

    # Get first page
    page1 = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        page=1,
        page_size=2,
        similarity_threshold=0.0,
    )

    # Get second page
    page2 = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        page=2,
        page_size=2,
        similarity_threshold=0.0,
    )

    assert len(page1.results) == 2
    assert page1.page == 1
    assert page1.has_next

    assert len(page2.results) > 0
    assert page2.page == 2
    assert page2.has_previous

    # Ensure different results on different pages
    page1_ids = {r.chunk.index for r in page1.results}
    page2_ids = {r.chunk.index for r in page2.results}
    # assert not page1_ids.intersection(page2_ids) TODO fix this


def test_search_with_threshold(populated_handler):
    """Test search with similarity threshold."""
    handler, project_id = populated_handler

    # Search with high threshold
    high_threshold_results = handler.search_chunks_by_text(
        project_id=project_id, query_text="Test query", similarity_threshold=0.9
    )

    # Search with low threshold
    low_threshold_results = handler.search_chunks_by_text(
        project_id=project_id, query_text="Test query", similarity_threshold=0.1
    )

    assert len(high_threshold_results.results) <= len(low_threshold_results.results)
    assert all(r.score >= 0.9 for r in high_threshold_results.results)


def test_search_invalid_project(populated_handler):
    """Test search with invalid project ID."""
    handler, _ = populated_handler

    results = handler.search_chunks_by_text(
        project_id=99999, query_text="Test query"  # Invalid project ID
    )

    assert results.total_count == 0
    assert len(results.results) == 0


def test_search_invalid_page(populated_handler):
    """Test search with invalid page parameters."""
    handler, project_id = populated_handler

    # Test invalid page number
    with pytest.raises(ValueError, match="Page number must be greater than 0"):
        handler.search_chunks_by_text(
            project_id=project_id,
            query_text="Test query",
            page=0,  # Invalid page number
        )

    # Test invalid page size
    with pytest.raises(ValueError, match="Page size must be greater than 1"):
        handler.search_chunks_by_text(
            project_id=project_id,
            query_text="Test query",
            page_size=0,  # Invalid page size
        )


def test_ordering(populated_handler):
    """Test that results are properly ordered by similarity."""
    handler, project_id = populated_handler

    results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        page_size=10,
        similarity_threshold=0.0,
    )

    # Check that scores are in descending order
    scores = [r.score for r in results.results]
    assert scores == sorted(scores, reverse=True)


def test_empty_results(populated_handler):
    """Test handling of empty results."""
    handler, project_id = populated_handler

    # Search with impossibly high threshold
    results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        similarity_threshold=2.0,  # No results will match this threshold
    )

    assert results.total_count == 0
    assert len(results.results) == 0
    assert not results.has_next
    assert not results.has_previous
    assert results.total_pages == 0


def test_last_page(populated_handler):
    """Test behavior of last page."""
    handler, project_id = populated_handler

    # Get total count with a large page size
    full_results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        page_size=100,
        similarity_threshold=0.0,
    )

    total_pages = full_results.total_pages

    # Get last page
    last_page = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        page=total_pages,
        page_size=2,
        similarity_threshold=0.0,
    )

    # assert not last_page.has_next TODO fix this
    assert last_page.has_previous if total_pages > 1 else not last_page.has_previous
    assert last_page.page == total_pages

    # Try to get page beyond the last page
    # with pytest.raises(ValueError, match="Page number must be greater than 0"):
    #     handler.search_chunks_by_text(
    #         project_id=project_id,
    #         query_text="Test query",
    #         page=total_pages + 1,
    #         page_size=2,
    #         similarity_threshold=0.0
    #     )
