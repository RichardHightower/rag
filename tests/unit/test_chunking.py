"""Tests for text chunking functionality."""

import pytest

from rag.chunking import split_text_into_chunks


def test_empty_text():
    """Test chunking empty text."""
    chunks = split_text_into_chunks("")
    assert len(chunks) == 0


def test_text_smaller_than_chunk_size():
    """Test when text is smaller than chunk size."""
    text = "This is a short text"
    chunks = split_text_into_chunks(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].start == 0
    assert chunks[0].end == len(text)


def test_text_larger_than_chunk_size():
    """Test when text is larger than chunk size."""
    text = "This is a longer text that should be split into multiple chunks based on the specified chunk size."
    chunk_size = 20
    overlap = 5
    
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
    
    assert len(chunks) > 1
    
    # Verify each chunk's properties
    for i, chunk in enumerate(chunks):
        # Verify chunk boundaries
        assert chunk.start >= 0
        assert chunk.end <= len(text)
        assert chunk.start < chunk.end
        
        # Verify content matches position
        assert chunk.content == text[chunk.start:chunk.end]
        
        # Verify overlap with next chunk if not last
        if i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            overlap_text = text[chunk.end - overlap:chunk.end]
            assert overlap_text in next_chunk.content
    
    # Verify complete text is covered
    reconstructed = ""
    last_end = 0
    for chunk in chunks:
        if chunk.start > last_end:
            # Add any text between chunks
            reconstructed += text[last_end:chunk.start]
        reconstructed += chunk.content
        last_end = chunk.end
    
    # Add any remaining text
    if last_end < len(text):
        reconstructed += text[last_end:]
    
    assert text == reconstructed


def test_chunk_overlap():
    """Test that chunks overlap correctly."""
    text = "This is a test of the chunking overlap functionality"
    chunk_size = 20
    overlap = 10
    
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
    
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        
        # Verify overlap size
        overlap_end = text[current_chunk.end - overlap:current_chunk.end]
        overlap_start = text[next_chunk.start:next_chunk.start + len(overlap_end)]
        assert overlap_end == overlap_start
        
        # Verify no gaps between chunks
        assert next_chunk.start <= current_chunk.end


def test_no_tiny_final_chunk():
    """Test that the final chunk is not too small."""
    text = "This is a test text that should not create a tiny final chunk"
    chunk_size = 20
    overlap = 5
    
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
    
    # The last chunk should be larger than the overlap
    last_chunk = chunks[-1]
    assert len(last_chunk.content) > overlap
    
    # Verify the last chunk is properly merged if it would have been tiny
    if len(chunks) > 1:
        second_to_last = chunks[-2]
        # Check there's no tiny gap between second to last and last chunk
        gap_size = last_chunk.start - second_to_last.end
        assert gap_size <= 0  # Should overlap, not gap


def test_word_boundary_respect():
    """Test that chunks try to split at word boundaries."""
    text = "This is a test of word boundary splitting"
    chunk_size = 10
    overlap = 2
    
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
    
    for chunk in chunks:
        # First character should not be in the middle of a word
        if chunk.start > 0:
            assert text[chunk.start - 1].isspace()
            
        # Last character position should be end of word or text
        if chunk.end < len(text):
            assert text[chunk.end].isspace()


def test_validation():
    """Test input validation."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        split_text_into_chunks("test", chunk_size=0)
        
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        split_text_into_chunks("test", chunk_size=-1)
        
    with pytest.raises(ValueError, match="overlap must be non-negative"):
        split_text_into_chunks("test", chunk_size=10, overlap=-1)
        
    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        split_text_into_chunks("test", chunk_size=10, overlap=10)
        
    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        split_text_into_chunks("test", chunk_size=10, overlap=11)
