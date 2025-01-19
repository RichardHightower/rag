import pytest

from vector_rag.chunking import SizeChunker, debug_chunker
from vector_rag.model import File


@pytest.fixture
def sample_file():
    return File(
        name="test.txt",
        path="/path/to/test.txt",
        crc="abcdef123456",
        content="This is a test file. It contains multiple sentences. "
        "We will use it to test the SizeChunker class. "
        "The chunker should split this text based on character size.",
        meta_data={"type": "test"},
    )


def test_size_chunker_init():
    chunker = SizeChunker(chunk_size=50, overlap=10)
    assert chunker.chunk_size == 50
    assert chunker.overlap == 10


def test_size_chunker_invalid_init():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        SizeChunker(chunk_size=0)

    with pytest.raises(ValueError, match="overlap must be non-negative"):
        SizeChunker(chunk_size=50, overlap=-1)

    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        SizeChunker(chunk_size=50, overlap=50)


def test_size_chunker_chunk_text(sample_file):
    chunker = SizeChunker(chunk_size=50, overlap=10)

    print("\nDebug output:")
    debug_chunker(chunker, sample_file)

    chunks = chunker.chunk_text(sample_file)

    assert len(chunks) == 5
    assert chunks[0].content == "This is a test file. It contains multiple"
    assert chunks[1].content == "s multiple sentences. We will use it to test the"
    assert chunks[2].content == "o test the SizeChunker class. The chunker should"
    assert chunks[3].content == "ker should split this text based on character"
    assert chunks[4].content == " character size."


def test_size_chunker_small_content():
    small_file = File(
        name="small.txt",
        path="/path/to/small.txt",
        crc="small123",
        content="Short text",
        meta_data={},
    )
    chunker = SizeChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk_text(small_file)

    assert len(chunks) == 1
    assert chunks[0].content == "Short text"


def test_size_chunker_empty_content():
    empty_file = File(
        name="empty.txt",
        path="/path/to/empty.txt",
        crc="empty123",
        content="",
        meta_data={},
    )
    chunker = SizeChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk_text(empty_file)

    assert len(chunks) == 0
