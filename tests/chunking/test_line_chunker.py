"""Test text chunking utilities."""

import pytest

from vector_rag.chunking import LineChunker
from vector_rag.config import Config
from vector_rag.model import File


@pytest.fixture
def chunker():
    """Create a line chunker instance."""
    return LineChunker(Config())


@pytest.fixture
def sample_file():
    """Create a sample file for testing."""
    return File(
        name="test.txt",
        path="/path/to/test.txt",
        crc="abcdef123456",
        content="\n".join([f"Line {i}" for i in range(10)]),
        meta_data={"type": "test"},
    )


def test_basic_chunking(chunker, sample_file):
    """Test basic text chunking with default parameters."""
    # Default chunk_size=500, overlap=50 should return single chunk
    chunks = chunker.chunk_text(sample_file)
    assert len(chunks) == 1
    assert chunks[0].content == sample_file.content
    assert chunks[0].index == 0


def test_custom_chunk_size(chunker, sample_file):
    """Test chunking with custom chunk size."""
    # Set chunk_size to 4 lines, overlap to 1
    chunker = LineChunker.create(chunk_size=4, overlap=1)
    chunks = chunker.chunk_text(sample_file)

    assert len(chunks) == 3
    # Verify first chunk
    assert chunks[0].content == "\n".join(["Line 0", "Line 1", "Line 2", "Line 3"])
    assert chunks[0].index == 0

    # Verify middle chunk has overlap
    assert chunks[1].content == "\n".join(["Line 3", "Line 4", "Line 5", "Line 6"])
    assert chunks[1].index == 1

    # Verify last chunk
    assert chunks[2].content == "\n".join(["Line 6", "Line 7", "Line 8", "Line 9"])
    assert chunks[2].index == 2


def test_empty_text(chunker):
    """Test chunking empty text."""
    empty_file = File(
        name="empty.txt",
        path="/path/to/empty.txt",
        crc="empty123",
        content="content",
        meta_data={},
    )
    chunks = chunker.chunk_text(empty_file)
    assert len(chunks) == 1
    assert chunks[0].content == "content"


def test_whitespace_text(chunker):
    """Test chunking whitespace text."""
    whitespace_file = File(
        name="whitespace.txt",
        path="/path/to/whitespace.txt",
        crc="space123",
        content="   \n  \n  ",
        meta_data={},
    )
    chunks = chunker.chunk_text(whitespace_file)
    assert len(chunks) == 1
    assert chunks[0].content == "   \n  \n  "


def test_single_line(chunker):
    """Test chunking single line of text."""
    single_line_file = File(
        name="single.txt",
        path="/path/to/single.txt",
        crc="single123",
        content="Single line",
        meta_data={},
    )
    chunks = chunker.chunk_text(single_line_file)
    assert len(chunks) == 1
    assert chunks[0].content == "Single line"


def test_text_smaller_than_chunk(chunker):
    """Test when text is smaller than chunk size."""
    small_file = File(
        name="small.txt",
        path="/path/to/small.txt",
        crc="small123",
        content="\n".join([f"Line {i}" for i in range(5)]),
        meta_data={},
    )
    chunker = LineChunker.create(chunk_size=10, overlap=5)
    chunks = chunker.chunk_text(small_file)
    assert len(chunks) == 1
    assert chunks[0].content == small_file.content


def test_no_overlap(chunker):
    """Test chunking with no overlap."""
    file = File(
        name="test.txt",
        path="/path/to/test.txt",
        crc="test123",
        content="\n".join([f"Line {i}" for i in range(6)]),
        meta_data={},
    )
    chunker = LineChunker.create(chunk_size=2, overlap=0)
    chunks = chunker.chunk_text(file)

    assert len(chunks) == 3
    assert chunks[0].content == "\n".join(["Line 0", "Line 1"])
    assert chunks[1].content == "\n".join(["Line 2", "Line 3"])
    assert chunks[2].content == "\n".join(["Line 4", "Line 5"])


def test_invalid_chunk_size(chunker, sample_file):
    """Test invalid chunk size."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunker = LineChunker.create(chunk_size=0)
        chunker.chunk_text(sample_file)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunker = LineChunker.create(chunk_size=-1)
        chunker.chunk_text(sample_file)


def test_invalid_overlap(chunker, sample_file):
    """Test invalid overlap size."""
    with pytest.raises(ValueError, match="overlap must be non-negative"):
        chunker = LineChunker.create(chunk_size=2, overlap=-1)
        chunker.chunk_text(sample_file)

    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        chunker = LineChunker.create(chunk_size=2, overlap=2)
        chunker.chunk_text(sample_file)

    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        chunker = LineChunker.create(chunk_size=2, overlap=3)
        chunker.chunk_text(sample_file)
