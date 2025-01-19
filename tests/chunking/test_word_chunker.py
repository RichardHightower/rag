import pytest

from vector_rag.chunking import WordChunker, debug_chunker
from vector_rag.model import File


@pytest.fixture
def sample_file():
    return File(
        id=None,
        name="test.txt",
        path="/path/to/test.txt",
        crc="abcdef123456",
        content="This is a test file with multiple words. It has several sentences. "
        "This content will definitely span across multiple chunks when we use a small chunk size.",
        meta_data={"type": "test"},
        file_size=None,
    )


def test_custom_chunk_size(sample_file):
    """Test chunking with custom chunk size."""
    chunker = WordChunker(chunk_size=10, overlap=2)

    print("\nDebug output:")
    debug_chunker(chunker, sample_file)

    chunks = chunker.chunk_text(sample_file)

    assert len(chunks) == 4, f"Expected 4 chunks, but got {len(chunks)}"

    # Check the content of each chunk
    assert chunks[0].content == "This is a test file with multiple words. It has"
    assert (
        chunks[1].content
        == "It has several sentences. This content will definitely span across"
    )
    assert chunks[2].content == "span across multiple chunks when we use a small chunk"
    assert chunks[3].content == "small chunk size."

    # Check for overlap
    assert chunks[0].content.split()[-2:] == chunks[1].content.split()[:2]
    assert chunks[1].content.split()[-2:] == chunks[2].content.split()[:2]
    assert chunks[2].content.split()[-2:] == chunks[3].content.split()[:2]


def test_overlap(sample_file):
    """Test chunking with overlap."""
    chunker = WordChunker(chunk_size=20, overlap=5)

    print("\nDebug output:")
    debug_chunker(chunker, sample_file)

    chunks = chunker.chunk_text(sample_file)

    assert len(chunks) > 1, "Expected multiple chunks"

    # Check for overlap
    for i in range(len(chunks) - 1):
        overlap_words = set(chunks[i].content.split()[-5:]).intersection(
            set(chunks[i + 1].content.split()[:5])
        )
        assert len(overlap_words) > 0, f"No overlap found between chunk {i} and {i+1}"


def test_small_content(sample_file):
    """Test chunking with content smaller than chunk size."""
    small_content = "Short text."
    small_file = File(**{**sample_file.__dict__, "content": small_content})
    chunker = WordChunker(chunk_size=20, overlap=2)

    print("\nDebug output:")
    debug_chunker(chunker, small_file)

    chunks = chunker.chunk_text(small_file)

    assert len(chunks) == 1, f"Expected 1 chunk, but got {len(chunks)}"
    assert chunks[0].content == small_content


if __name__ == "__main__":
    pytest.main([__file__])
